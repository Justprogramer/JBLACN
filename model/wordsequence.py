import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .wordrep import WordRep
from model.lstm_attention import LSTM_attention, multihead_attention


class WordSequence(nn.Module):
    def __init__(self, data):
        super(WordSequence, self).__init__()
        print("build word sequence feature extractor: %s..." % (data.word_feature_extractor))
        self.data = data
        self.gpu = data.HP_gpu
        # self.batch_size = data.HP_batch_size
        # self.hidden_dim = data.HP_hidden_dim
        self.droplstm = nn.Dropout(data.HP_dropout)
        self.bilstm_flag = data.HP_bilstm
        self.num_of_lstm_layer = data.HP_lstm_layer
        # word embedding
        self.wordrep = WordRep(data)

        self.input_size = data.word_emb_dim
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim

        self.word_feature_extractor = data.word_feature_extractor
        # word
        self.lstm_first = nn.LSTM(self.input_size, lstm_hidden, num_layers=1, batch_first=True,
                                  bidirectional=self.bilstm_flag)
        # LAN module
        self.self_attention_first = multihead_attention(data.HP_hidden_dim, num_heads=data.num_attention_head,
                                                        dropout_rate=data.HP_dropout, gpu=self.gpu)

        self.lstm_attention_stack = nn.ModuleList(
            [LSTM_attention(lstm_hidden, self.bilstm_flag, data) for _ in range(int(self.num_of_lstm_layer) - 2)])

        self.lstm_last = nn.LSTM(lstm_hidden * 4, lstm_hidden, num_layers=1, batch_first=True,
                                 bidirectional=self.bilstm_flag)
        # self.lstm_last = nn.LSTM(lstm_hidden * 2, lstm_hidden, num_layers=1, batch_first=True,
        #                          bidirectional=self.bilstm_flag)
        # self.hidden2label = nn.Linear(lstm_hidden * 2, data.label_alphabet_size)
        self.self_attention_last = multihead_attention(data.HP_hidden_dim, num_heads=1, dropout_rate=0,
                                                       gpu=self.gpu)
        # DO NOT Add dropout at last layer
        # highway encoding
        # self.highway_encoding = HighwayEncoding(data,data.HP_hidden_dim,activation_function=F.relu)

        if self.gpu:
            self.droplstm = self.droplstm.cuda()
            self.lstm_first = self.lstm_first.cuda()
            self.lstm_last = self.lstm_last.cuda()
            # self.hidden2label = self.hidden2label.cuda()

    def embedding(self, word_inputs, word_seq_lengths, input_label_seq_tensor):
        lstm_out, label_embs = self.wordrep(word_inputs, input_label_seq_tensor)
        # word_represent shape [batch_size, seq_length, word_embedding_dim+char_hidden_dim]
        # word_embs (batch_size, seq_len, embed_size)
        # label_embs = self.highway_encoding(label_embs)
        """
        First LSTM layer (input word only)
        """
        # lstm_out = word_represent
        lstm_out = pack_padded_sequence(input=lstm_out, lengths=word_seq_lengths.cpu().numpy(), batch_first=True)
        hidden = None
        lstm_out, hidden = self.lstm_first(lstm_out, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        # shape [seq_len, batch, hidden_size]
        lstm_out = self.droplstm(lstm_out.transpose(1, 0))
        return lstm_out, hidden, label_embs

    # def sentence_classify(self, hidden, batch_word_recover, sent_type_embs):
    #     # sentence module
    #     # (batch * length * embedding)
    #     sentence_represent = torch.cat([hidden[0][0], hidden[0][1]], -1).squeeze()
    #     if len(sentence_represent.size()) == 2:
    #         sentence_represent = sentence_represent.unsqueeze(1)
    #     if len(sentence_represent.size()) == 1:
    #         sentence_represent = sentence_represent.unsqueeze(0).unsqueeze(0)
    #     sentence_represent = sentence_represent[batch_word_recover]
    #     attention_label_sent = self.sent_attention_first(sentence_represent, sent_type_embs, sent_type_embs)
    #     sent_out = torch.cat([sentence_represent, attention_label_sent], -1)
    #     # LAN layer
    #     # for layer in self.sentence_lstm_attention_stack:
    #     #     sent_out = layer(sent_out, sent_type_embs, evidence_word_length, hidden)
    #     sent_out, _ = self.sent_layer(sent_out)
    #     sent_out = self.sent_attention_last(sent_out, sent_type_embs, sent_type_embs, True)
    #     return sent_out

    def word_layer(self, hidden, word_seq_lengths, lstm_out, label_embs):
        attention_label = self.self_attention_first(lstm_out, label_embs, label_embs)
        # shape [batch_size, seq_length, embedding_dim]
        lstm_out = torch.cat([lstm_out, attention_label], -1)
        # shape [batch_size, seq_length, embedding_dim + label_embeeding_dim]
        """
        Last Layer 
        Attention weight calculate loss
        """
        # LAN layer
        for layer in self.lstm_attention_stack:
            lstm_out = layer(lstm_out, label_embs, word_seq_lengths, hidden)

        lstm_out = pack_padded_sequence(input=lstm_out,
                                        lengths=word_seq_lengths.cpu().numpy(),
                                        batch_first=True)
        lstm_out, hidden = self.lstm_last(lstm_out, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        lstm_out = self.droplstm(lstm_out.transpose(1, 0))
        # lstm_out = self.hidden2label(lstm_out)
        lstm_out = self.self_attention_last(lstm_out, label_embs, label_embs, True)
        return lstm_out

    def forward(self, word_inputs, word_seq_lengths, input_label_seq_tensor):
        """
            input:
                word_inputs: (batch_size, sent_len)
                word_seq_lengths: list of batch_size, (batch_size,1)
                label_size: nubmer of label
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        lstm_out, hidden, label_embs = self.embedding(word_inputs, word_seq_lengths, input_label_seq_tensor)
        lstm_out = self.word_layer(hidden, word_seq_lengths, lstm_out, label_embs)
        return lstm_out

    # def evaluate_sentence(self, word_inputs,
    #                       word_seq_lengths,
    #                       input_evidence_label_seq_tensor,
    #                       input_opinion_label_seq_tensor,
    #                       input_sent_type_tensor,
    #                       batch_word_recover):
    #     lstm_out, hidden, evidence_label_embs, \
    #     opinion_label_embs, sent_type_embs = self.embedding(word_inputs,
    #                                                         word_seq_lengths,
    #                                                         input_evidence_label_seq_tensor,
    #                                                         input_opinion_label_seq_tensor,
    #                                                         input_sent_type_tensor)
    #     sent_out = self.sentence_classify(hidden, batch_word_recover, sent_type_embs)
    #     # sent_out此时已经恢复了顺序
    #     return lstm_out, hidden, sent_out, evidence_label_embs, opinion_label_embs

    def evaluate_word(self, lstm_out, hidden, sent_out, word_seq_lengths, label_embs):
        lstm_out = self.word_layer(hidden, word_seq_lengths, lstm_out, label_embs)
        return lstm_out
