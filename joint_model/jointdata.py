# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-14 17:34:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-06-22 00:01:47
from __future__ import absolute_import
from __future__ import print_function

import os
import sys

from joint_model.functions import read_instance, normalize_word, build_pretrain_embedding
from utils.alphabet import Alphabet
from utils.log import logger

try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"


class JointData:
    def __init__(self):
        self.MAX_SENTENCE_LENGTH = 250
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = True
        self.norm_word_emb = False
        self.word_alphabet = Alphabet('word')
        self.label = ["O", "B-A", "I-A", "B-O", "I-O", "B-E", "I-E", "B-T", "I-T", "B-C", "I-C"]
        self.label_alphabet = Alphabet('label', True)
        self.sentence_type_alphabet = Alphabet('sentence', True)
        self.tagScheme = "NoSeg"  ## BMES/BIO

        self.seg = True

        ### I/O
        self.train_dir = None
        self.dev_dir = None
        self.test_dir = None
        self.raw_dir = None

        self.decode_dir = None
        self.dset_dir = None  ## data vocabulary related file
        self.model_dir = None  ## model save  file
        self.load_model_dir = None  ## model load file

        self.word_emb_dir = None
        self.word_emb_file = None

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []

        self.pretrain_word_embedding = None
        self.use_pre_trained_model = None

        self.word_alphabet_size = 0
        self.opinion_label_alphabet_size = 0
        self.evidence_label_alphabet_size = 0
        self.sentence_alphabet_size = 0
        self.word_emb_dim = 50
        self.lstm_input_size = 50

        ###Networks
        self.word_feature_extractor = "LSTM"  ## "LSTM"/"CNN"/"GRU"/
        self.use_crf = True
        self.nbest = None

        ## Training
        self.average_batch_loss = False
        self.optimizer = "SGD"  ## "SGD"/"AdaGrad"/"AdaDelta"/"RMSProp"/"Adam"
        self.status = "train"
        ### Hyperparameters
        self.HP_iteration = 100
        self.HP_batch_size = 10
        self.HP_hidden_dim = 200
        self.HP_attention_query_input_dim = 200
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True

        self.HP_gpu = False
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05
        self.HP_clip = None
        self.HP_momentum = 0
        self.HP_l2 = 1e-8

    def show_data_summary(self):
        logger.info("++" * 50)
        logger.info("DATA SUMMARY START:")
        logger.info(" I/O:")
        logger.info("     Tag          scheme: %s" % (self.tagScheme))
        logger.info("     MAX SENTENCE LENGTH: %s" % (self.MAX_SENTENCE_LENGTH))
        logger.info("     MAX   WORD   LENGTH: %s" % (self.MAX_WORD_LENGTH))
        logger.info("     Number   normalized: %s" % (self.number_normalized))
        logger.info("     Word  alphabet size: %s" % (self.word_alphabet_size))
        logger.info("     Opinion Label alphabet size: %s" % (self.opinion_label_alphabet_size))
        logger.info("     Evidence Label alphabet size: %s" % (self.evidence_label_alphabet_size))
        logger.info("     Word embedding  dir: %s" % (self.word_emb_dir))
        logger.info("     Word embedding size: %s" % (self.word_emb_dim))
        logger.info("     Norm   word     emb: %s" % (self.norm_word_emb))
        logger.info("     Train  file directory: %s" % (self.train_dir))
        logger.info("     Dev    file directory: %s" % (self.dev_dir))
        logger.info("     Test   file directory: %s" % (self.test_dir))
        logger.info("     Raw    file directory: %s" % (self.raw_dir))
        logger.info("     Dset   file directory: %s" % (self.dset_dir))
        logger.info("     Model  file directory: %s" % (self.model_dir))
        logger.info("     Loadmodel   directory: %s" % (self.load_model_dir))
        logger.info("     Decode file directory: %s" % (self.decode_dir))
        logger.info("     Train instance number: %s" % (len(self.train_texts)))
        logger.info("     Dev   instance number: %s" % (len(self.dev_texts)))
        logger.info("     Test  instance number: %s" % (len(self.test_texts)))
        logger.info("     Raw   instance number: %s" % (len(self.raw_texts)))
        logger.info(" " + "++" * 20)
        logger.info(" Model Network:")
        logger.info("     Model        use_crf: %s" % (self.use_crf))
        logger.info("     Model word extractor: %s" % (self.word_feature_extractor))
        logger.info(" " + "++" * 20)
        logger.info(" Training:")
        logger.info("     Optimizer: %s" % (self.optimizer))
        logger.info("     Iteration: %s" % (self.HP_iteration))
        logger.info("     BatchSize: %s" % (self.HP_batch_size))
        logger.info("     Average  batch   loss: %s" % (self.average_batch_loss))

        logger.info(" " + "++" * 20)
        logger.info(" Hyperparameters:")

        logger.info("     Hyper              lr: %s" % (self.HP_lr))
        logger.info("     Hyper        lr_decay: %s" % (self.HP_lr_decay))
        logger.info("     Hyper         HP_clip: %s" % (self.HP_clip))
        logger.info("     Hyper        momentum: %s" % (self.HP_momentum))
        logger.info("     Hyper              l2: %s" % (self.HP_l2))
        logger.info("     Hyper      hidden_dim: %s" % (self.HP_hidden_dim))
        logger.info("     Hyper attention_input: %s" % (self.HP_attention_query_input_dim))
        logger.info("     Hyper         dropout: %s" % (self.HP_dropout))
        logger.info("     Hyper      lstm_layer: %s" % (self.HP_lstm_layer))
        logger.info("     Hyper          bilstm: %s" % (self.HP_bilstm))
        logger.info("     Hyper             GPU: %s" % (self.HP_gpu))
        logger.info("DATA SUMMARY END.")
        logger.info("++" * 50)
        sys.stdout.flush()

    def build_alphabet(self, input_file):
        in_lines = open(input_file, 'r', encoding='utf-8').readlines()
        for line in in_lines:
            pairs = line.strip().split()
            if len(pairs) == 2:
                word = pairs[0]
                if sys.version_info[0] < 3:
                    word = word.decode('utf-8')
                if self.number_normalized:
                    word = normalize_word(word)
                self.word_alphabet.add(word)
            if len(pairs) == 1 and not line.strip() == "<end>":
                sentence_type = line.strip()
                self.sentence_type_alphabet.add(sentence_type)
        for l in self.label:
            self.label_alphabet.add(l)
        self.word_alphabet_size = self.word_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        self.sentence_alphabet_size = self.sentence_type_alphabet.size()
        startS = False
        startB = False
        for label, _ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.label_alphabet.close()
        self.sentence_type_alphabet.close()

    def build_pretrain_emb(self):
        if self.word_emb_dir:
            emb_path = os.path.join(self.word_emb_dir, self.word_emb_file)
            logger.info("Load pretrained word embedding, norm: %s, dir: %s" % (self.norm_word_emb, emb_path))
            self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(emb_path,
                                                                                       self.word_alphabet,
                                                                                       self.word_emb_dim,
                                                                                       self.norm_word_emb)

    def generate_instance(self, name):
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_instance(self.train_dir, self.word_alphabet,
                                                             self.label_alphabet,
                                                             self.sentence_type_alphabet, self.number_normalized,
                                                             self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance(self.dev_dir, self.word_alphabet,
                                                         self.label_alphabet,
                                                         self.sentence_type_alphabet, self.number_normalized,
                                                         self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance(self.test_dir, self.word_alphabet,
                                                           self.label_alphabet,
                                                           self.sentence_type_alphabet, self.number_normalized,
                                                           self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance(self.raw_dir, self.word_alphabet,
                                                         self.label_alphabet,
                                                         self.sentence_type_alphabet, self.number_normalized,
                                                         self.MAX_SENTENCE_LENGTH)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s" % (name))

    def load(self, data_file):
        f = open(data_file, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def save(self, save_file):
        f = open(save_file, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def read_config(self, args):

        ## read data:

        self.train_dir = args.train_dir

        self.dev_dir = args.dev_dir

        self.test_dir = args.test_dir

        self.model_dir = args.model_dir

        self.word_emb_dir = args.word_emb_dir

        self.word_emb_file = args.word_emb_file

        self.norm_word_emb = str2bool(args.norm_word_emb)

        self.use_pre_trained_model = str2bool(str(args.use_pre_trained_model))

        self.number_normalized = str2bool(args.number_normalized)

        self.seg = args.seg

        self.lstm_input_size = int(args.lstm_input_size)

        self.use_crf = str2bool(str(args.use_crf))

        self.word_feature_extractor = args.word_seq_feature

        ## read training setting:

        self.optimizer = args.optimizer

        self.average_batch_loss = args.ave_batch_loss

        self.status = args.status

        self.HP_iteration = int(args.iteration)

        self.HP_batch_size = int(args.batch_size)

        self.HP_hidden_dim = int(args.hidden_dim)

        self.HP_attention_query_input_dim = int(args.attention_query_input_size)

        self.HP_dropout = float(args.dropout)

        self.HP_lstm_layer = int(args.lstm_layer)

        self.HP_bilstm = args.bilstm

        self.HP_gpu = args.gpu

        self.HP_lr = float(args.learning_rate)

        self.HP_lr_decay = float(args.lr_decay)

        self.HP_momentum = float(args.momentum)

        self.HP_l2 = float(args.l2)

        self.clip_grad = float(args.clip_grad)

        self.label_embedding_scale = float(args.label_embedding_scale)

        self.num_attention_head = int(args.num_attention_head)

        self.whether_clip_grad = str2bool(args.whether_clip_grad)

        ##
        # self.MAX_SENTENCE_LENGTH = int(args.MAX_SENTENCE_LENGTH)
        #
        # self.MAX_WORD_LENGTH = int(args.MAX_WORD_LENGTH )


def config_file_to_dict(input_file):
    config = {}
    fins = open(input_file, 'r').readlines()
    for line in fins:
        if len(line) > 0 and line[0] == "#":
            continue
        if "=" in line:
            pair = line.strip().split('#', 1)[0].split('=', 1)
            item = pair[0]
            if item == "feature":
                if item not in config:
                    feat_dict = {}
                    config[item] = feat_dict
                feat_dict = config[item]
                new_pair = pair[-1].split()
                feat_name = new_pair[0]
                one_dict = {}
                one_dict["emb_dir"] = None
                one_dict["emb_size"] = 10
                one_dict["emb_norm"] = False
                if len(new_pair) > 1:
                    for idx in range(1, len(new_pair)):
                        conf_pair = new_pair[idx].split('=')
                        if conf_pair[0] == "emb_dir":
                            one_dict["emb_dir"] = conf_pair[-1]
                        elif conf_pair[0] == "emb_size":
                            one_dict["emb_size"] = int(conf_pair[-1])
                        elif conf_pair[0] == "emb_norm":
                            one_dict["emb_norm"] = str2bool(conf_pair[-1])
                feat_dict[feat_name] = one_dict
                # print "feat",feat_dict
            else:
                if item in config:
                    print("Warning: duplicated config item found: %s, updated." % (pair[0]))
                config[item] = pair[-1]
    return config


def str2bool(string):
    return str(string).lower() == 'true'
