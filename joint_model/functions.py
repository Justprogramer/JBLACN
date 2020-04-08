# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:23:06
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-06-10 17:49:50
from __future__ import print_function
from __future__ import absolute_import
import sys
import numpy as np

from utils.log import logger


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance(input_file, word_alphabet, label_alphabet, sentence_type_alphabet,
                  number_normalized,
                  max_sent_length):
    in_lines = open(input_file, 'r', encoding='utf-8').readlines()
    instance_texts = []
    instance_Ids = []
    document = []
    document_id = []
    words = []
    labels = []
    word_ids = []
    label_ids = []
    sentence_type = None
    for line in in_lines:
        if len(line.strip()) > 0:
            pairs = line.strip().split()
            if len(pairs) == 2:
                if sys.version_info[0] < 3:
                    word = pairs[0].decode('utf-8')
                else:
                    word = pairs[0]
                if number_normalized:
                    word = normalize_word(word)
                label = pairs[-1]
                words.append(word)
                labels.append(label)
                word_ids.append(word_alphabet.get_index(word))
                label_ids.append(label_alphabet.get_index(label))
            elif line.strip() == "<end>":
                instance_texts.append(document)
                instance_Ids.append(document_id)
                words = []
                labels = []
                word_ids = []
                label_ids = []
                document = []
                document_id = []
            else:
                sentence_type = line.strip()
        else:
            if len(words) > 0:
                document.append([words, labels, sentence_type])
                document_id.append([word_ids, label_ids, sentence_type_alphabet.get_index(sentence_type)])
            words = []
            labels = []
            word_ids = []
            label_ids = []
    return instance_texts, instance_Ids


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index, :] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index, :] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    logger.info("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
        pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / alphabet_size))
    return pretrain_emb, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0 or len(line.split()) == 2:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                if embedd_dim + 1 != len(tokens):
                    continue
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            if sys.version_info[0] < 3:
                first_col = tokens[0].decode('utf-8')
            else:
                first_col = tokens[0]
            embedd_dict[first_col] = embedd
    return embedd_dict, embedd_dim
