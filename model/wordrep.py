from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import numpy as np


class WordRep(nn.Module):
    def __init__(self, data):
        super(WordRep, self).__init__()
        print("build word representation...")
        self.gpu = data.HP_gpu
        self.batch_size = data.HP_batch_size

        self.embedding_dim = data.word_emb_dim
        self.drop = nn.Dropout(data.HP_dropout)
        # word embedding
        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)

        if data.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.embedding_dim)))

        # label embedding
        self.label_dim = data.HP_hidden_dim
        self.label_embedding = nn.Embedding(data.label_alphabet_size, self.label_dim)

        self.label_embedding.weight.data.copy_(torch.from_numpy(
            self.random_embedding_label(data.label_alphabet_size, self.label_dim, data.label_embedding_scale)))

        # whether to use gpu
        if self.gpu:
            self.drop = self.drop.cuda()
            self.word_embedding = self.word_embedding.cuda()
            self.label_embedding = self.label_embedding.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def random_embedding_label(self, vocab_size, embedding_dim, scale):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        # scale = np.sqrt(3.0 / embedding_dim)
        # scale = 0.025
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, word_inputs, input_label_seq_tensor):
        """
            input:
                word_inputs: (batch_size, sent_len)
                word_seq_lengths: list of batch_size, (batch_size,1)
                input_label_seq_tensor: (batch_size, number of label)
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        word_embs = self.word_embedding(word_inputs)
        # label embedding
        label_embs = self.label_embedding(input_label_seq_tensor)
        word_represent = self.drop(word_embs)
        # label_embs = self.drop(label_embs)
        return word_represent, label_embs
