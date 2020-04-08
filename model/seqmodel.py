from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.crf import CRF
from .wordsequence import WordSequence


class SeqModel(nn.Module):
    def __init__(self, data):
        super(SeqModel, self).__init__()
        self.data = data
        self.use_crf = data.use_crf
        print("build network...")
        print("word feature extractor: ", data.word_feature_extractor)

        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        # opinion 和 evidence 分开抽
        label_size = data.label_alphabet_size
        self.word_hidden = WordSequence(data)
        if self.use_crf:
            self.word_crf = CRF(label_size, batch_first=True)
            if self.gpu:
                self.word_crf = self.word_crf.cuda()

    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, batch_label, mask,
                                input_label_seq_tensor):
        lstm_outs = self.word_hidden(word_inputs, word_seq_lengths, input_label_seq_tensor)
        # lstm_outs(batch_size,sentence_length,tag_size)
        batch_size = word_inputs.size(0)
        if self.use_crf:
            mask = mask.byte()
            loss = (-self.word_crf(lstm_outs, batch_label, mask))
            tag_seq = self.word_crf.decode(lstm_outs, mask)
        else:
            loss_function = nn.NLLLoss()
            seq_len = lstm_outs.size(1)
            lstm_outs = lstm_outs.view(batch_size * seq_len, -1)
            score = F.log_softmax(lstm_outs, 1)
            loss = loss_function(score, batch_label.contiguous().view(batch_size * seq_len))
            _, tag_seq = torch.max(score, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
        return loss, tag_seq

    def evaluate(self, word_inputs, word_seq_lengths, mask, input_label_seq_tensor):
        lstm_outs = self.word_hidden(word_inputs, word_seq_lengths, input_label_seq_tensor)
        if self.use_crf:
            mask = mask.byte()
            tag_seq = self.word_crf.decode(lstm_outs, mask)
        else:
            batch_size = word_inputs.size(0)
            seq_len = lstm_outs.size(1)
            lstm_outs = lstm_outs.view(batch_size * seq_len, -1)
            _, tag_seq = torch.max(lstm_outs, 1)
            tag_seq = mask.long() * tag_seq.view(batch_size, seq_len)
        return tag_seq

    def forward(self, word_inputs, word_seq_lengths, mask, input_label_seq_tensor):
        return self.evaluate(word_inputs, word_seq_lengths, mask, input_label_seq_tensor)
