from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from joint_model.jointsequence import JointSequence
from model.crf import CRF
from utils.log import logger


class JointModel(nn.Module):
    def __init__(self, data):
        super(JointModel, self).__init__()
        self.data = data
        self.use_crf = data.use_crf
        logger.info("build network...")
        logger.info("word feature extractor: %s" % data.word_feature_extractor)
        logger.info("use_cuda: %s" % data.HP_gpu)
        self.gpu = data.HP_gpu
        logger.info("use_crf: %s" % data.use_crf)
        self.average_batch = data.average_batch_loss

        label_size = data.label_alphabet_size
        sentence_size = data.sentence_alphabet_size
        self.word_hidden = JointSequence(data)
        if self.use_crf:
            self.word_crf = CRF(label_size, batch_first=True)
            self.sent_crf = CRF(sentence_size, batch_first=True)
            if self.gpu:
                self.word_crf = self.word_crf.cuda()
                self.sent_crf = self.sent_crf.cuda()

    def neg_log_likelihood_loss(self, word_inputs, word_tensor, word_seq_lengths, batch_label, batch_sent_type, mask,
                                sent_mask,
                                input_label_seq_tensor, input_sent_type_tensor,
                                batch_word_recover, word_perm_idx, need_cat=True, need_embedding=True):
        words_outs, sent_out = self.word_hidden(word_inputs, word_tensor, word_seq_lengths,
                                                input_label_seq_tensor, input_sent_type_tensor,
                                                batch_word_recover, word_perm_idx, batch_sent_type, need_cat,
                                                need_embedding)
        batch_size = words_outs.size(0)
        seq_len = words_outs.size(1)
        if self.use_crf:
            # e_out(batch_size,sentence_length,tag_size)
            words_loss = (-self.word_crf(words_outs, batch_label, mask)) / (
                    len(word_seq_lengths) * seq_len)
            words_tag_seq = self.word_crf.decode(words_outs, mask)
            sent_total_loss = -self.sent_crf(sent_out,
                                             batch_sent_type[batch_word_recover].view(batch_size, 1),
                                             sent_mask.view(batch_size, 1).byte()) / len(sent_mask)
            sent_tag_seq = self.sent_crf.decode(sent_out, sent_mask.view(batch_size, 1).byte())
        else:
            loss_function = nn.NLLLoss()
            words_outs = words_outs.view(batch_size * seq_len, -1)
            words_score = F.log_softmax(words_outs, 1)
            words_loss = loss_function(words_score, batch_label.contiguous().view(batch_size * seq_len))
            _, words_tag_seq = torch.max(words_score, 1)
            words_tag_seq = words_tag_seq.view(batch_size, seq_len)

            sent_out = sent_out.view(batch_size, -1)
            sent_score = F.log_softmax(sent_out, 1)
            sent_total_loss = loss_function(sent_score, batch_sent_type[batch_word_recover].view(batch_size))
            _, sent_tag_seq = torch.max(sent_score, 1)
        return words_loss, words_tag_seq, sent_total_loss, sent_tag_seq

    def evaluate(self, word_inputs, word_tensor, word_seq_lengths, batch_sent_type, mask, sent_mask,
                 input_label_seq_tensor, input_sent_type_tensor,
                 batch_word_recover, word_perm_idx, need_cat=True, need_embedding=True):
        words_out, sent_out = self.word_hidden(word_inputs, word_tensor, word_seq_lengths,
                                               input_label_seq_tensor,
                                               input_sent_type_tensor,
                                               batch_word_recover, word_perm_idx, batch_sent_type, need_cat,
                                               need_embedding)
        batch_size = words_out.size(0)
        seq_len = words_out.size(1)
        if self.use_crf:
            sent_tag_seq = self.sent_crf.decode(sent_out, sent_mask.view(batch_size, 1).byte())
            # 由于sentence在预测分类时已经恢复了顺序，后面的word顺序还没有恢复，所以此时要继续打乱顺序
            sent_tag_seq = torch.tensor(sent_tag_seq)[word_perm_idx]
            if self.gpu:
                sent_tag_seq = sent_tag_seq.cpu().data.numpy().tolist()
            else:
                sent_tag_seq = sent_tag_seq.data.numpy().tolist()
            words_tag_seq = self.word_crf.decode(words_out, mask)
        else:
            sent_out = sent_out.view(batch_size, -1)
            _, sent_tag_seq = torch.max(sent_out, 1)
            # 由于sentence在预测分类时已经恢复了顺序，后面的word顺序还没有恢复，所以此时要继续打乱顺序
            sent_tag_seq = sent_tag_seq[word_perm_idx]

            words_out = words_out.view(batch_size * seq_len, -1)
            _, words_tag_seq = torch.max(words_out, 1)
            words_tag_seq = mask.long() * words_tag_seq.view(batch_size, seq_len)
        return words_tag_seq, sent_tag_seq

    def forward(self, word_inputs, word_tensor, word_seq_lengths, mask, sent_mask,
                input_label_seq_tensor, input_sent_type_tensor,
                batch_word_recover, word_perm_idx,
                need_cat=True, need_embedding=True):
        batch_size = word_tensor.size(0)
        seq_len = word_tensor.size(1)
        lstm_out, hidden, sent_out, label_embs = self.word_hidden.evaluate_sentence(word_inputs,
                                                                                    word_tensor,
                                                                                    word_seq_lengths,
                                                                                    input_label_seq_tensor,
                                                                                    input_sent_type_tensor,
                                                                                    batch_word_recover,
                                                                                    need_cat, need_embedding)
        lstm_out = torch.cat(
            [lstm_out, sent_out[word_perm_idx].expand([lstm_out.size(0), lstm_out.size(1), sent_out.size(-1)])], -1)
        words_outs = self.word_hidden.evaluate_word(
            lstm_out, hidden,
            word_seq_lengths,
            label_embs
        )
        if self.use_crf:
            sent_tag_seq = self.sent_crf.decode(sent_out, sent_mask.view(batch_size, 1).byte())
            # 由于sentence在预测分类时已经恢复了顺序，后面的word顺序还没有恢复，所以此时要继续打乱顺序
            sent_tag_seq = torch.tensor(sent_tag_seq)[word_perm_idx]
            words_tag_seq = self.word_crf.decode(words_outs, mask)
        else:
            sent_out = sent_out.view(batch_size, -1)
            _, sent_tag_seq = torch.max(sent_out, 1)
            # 由于sentence在预测分类时已经恢复了顺序，后面的word顺序还没有恢复，所以此时要继续打乱顺序
            sent_tag_seq = sent_tag_seq[word_perm_idx]
            words_outs = words_outs.view(batch_size * seq_len, -1)
            _, words_tag_seq = torch.max(words_outs, 1)
            words_tag_seq = mask.long() * words_tag_seq.view(batch_size, seq_len)
        return words_tag_seq, sent_tag_seq
