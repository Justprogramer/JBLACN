from __future__ import print_function

import argparse
import gc
import random
import sys
import time

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, classification_report

from model.seqmodel import SeqModel
from utils.data import Data
from utils.metric import get_ner_fmeasure

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def data_initialization(data):
    data.build_alphabet(data.train_dir)
    data.build_alphabet(data.dev_dir)
    data.build_alphabet(data.test_dir)
    data.fix_alphabet()


def predict_check(pred_variable, gold_variable, mask_variable, use_crf=True):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    gold_variable = gold_variable.contiguous().view(-1)
    mask_variable = mask_variable.contiguous().view(-1)
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    if use_crf:
        pred = []
        for p in pred_variable:
            pred.extend(p)
        pred = np.array(pred)
    else:
        pred_variable = pred_variable.view(-1)
        pred = pred_variable.cpu().data.numpy()
        pred = np.array([p for p, m in zip(pred, mask) if m == 1])
    gold = np.array([g for g, m in zip(gold, mask) if m == 1])
    assert len(gold) == len(pred)
    right_token = np.sum(pred == gold)
    total_token = mask.sum()
    assert right_token <= total_token
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, data, word_recover, use_crf):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    gold_variable = gold_variable.contiguous().view(-1)
    gold_tag = gold_variable.cpu().data.numpy()
    mask = mask_variable.contiguous().view(-1)
    mask = mask.cpu().data.numpy()
    gold_tag = [g for g, m in zip(gold_tag, mask) if m == 1]
    if use_crf:
        pred_tag = []
        for p in pred_variable:
            pred_tag.extend(p)
    else:
        pred_variable = pred_variable.contiguous().view(-1)
        pred_variable = pred_variable.cpu().data.numpy()
        pred_tag = [p for p, m in zip(pred_variable, mask) if m == 1]
    assert (len(pred_tag) == len(gold_tag))

    # pred_label = [data.label_alphabet.get_instance(idx) for idx in pred_tag]
    # gold_label = [data.label_alphabet.get_instance(idx) for idx in gold_tag]

    return pred_tag, gold_tag


def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
    """
    # exit(0)
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    nbest = pred_variable.size(2)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = []
        for idz in range(nbest):
            each_pred = [label_alphabet.get_instance(pred_tag[idx][idy][idz]) for idy in range(seq_len) if
                         mask[idx][idy] != 0]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def evaluate(data, model, name):
    instances = None
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
        exit(1)
    start_time = time.time()
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()
    start = 0
    end = start + data.HP_batch_size
    eval_epochs = []
    while end <= len(instances):
        eval_epochs.append((start, end))
        start = end
        end = end + data.HP_batch_size
    if end > len(instances) > start:
        eval_epochs.append((start, len(instances)))
    for idx, (start, end) in enumerate(eval_epochs):
        instance = instances[start:end]
        batch_word, batch_word_len, word_perm_idx, batch_word_recover, batch_label, mask, input_label_seq_tensor = batchify_with_label(
            instance, data.HP_gpu, data)
        with torch.no_grad():
            tag_seq = model.evaluate(batch_word,
                                     batch_word_len,
                                     mask,
                                     input_label_seq_tensor)

        pred_label, gold_label = recover_label(tag_seq, batch_label, mask,
                                               data, batch_word_recover, data.use_crf)

        pred_results.extend(pred_label)
        gold_results.extend(gold_label)

    decode_time = time.time() - start_time
    report = classification_report(gold_results, pred_results,
                                   target_names=data.label_alphabet.instances)
    f_value = f1_score(gold_results, pred_results, average="macro")
    acc = accuracy_score(gold_results, pred_results)
    ner_acc, ner_p, ner_r, ner_f = get_ner_fmeasure(gold_results,
                                                    pred_results,
                                                    data.label_alphabet,
                                                    data.tagScheme,
                                                    name='ner',
                                                    need_save_matrix=name == 'test')
    speed = len(instances) / decode_time
    return speed, acc, report, f_value, ner_acc, ner_p, ner_r, ner_f


def batchify_with_label(input_batch_list, gpu, data, volatile_flag=False, train=True):
    """
        input: list of words and labels, various length. [[words, labels],[words,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
        output:
            zero padding for word, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    # label_instance
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    labels = [sent[1] for sent in input_batch_list]

    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()

    input_label_seq_tensor = autograd.Variable(
        torch.zeros((batch_size, data.label_alphabet_size)),
        volatile=volatile_flag).long()

    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).byte()
    sent_mask = autograd.Variable(torch.zeros(batch_size), volatile=volatile_flag).byte()
    for idx, (seq, label, seq_len) in enumerate(zip(words, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seq_len] = torch.LongTensor(label)
        mask[idx, :seq_len] = torch.ones(seq_len)
        sent_mask[idx] = torch.Tensor([1])
        input_label_seq_tensor[idx, :data.label_alphabet_size] = torch.LongTensor(
            [i for i in range(data.label_alphabet_size)])

    # 按照word length排序
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    # 按照新的顺序对tensor处理
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    # 原顺序索引，用于恢复tensor顺序
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        input_label_seq_tensor = input_label_seq_tensor.cuda()
        mask = mask.cuda()

    return word_seq_tensor, word_seq_lengths, word_perm_idx, word_seq_recover, label_seq_tensor, mask, input_label_seq_tensor


def train(data):
    print("Training model...")
    data.show_data_summary()
    save_data_name = data.model_dir + ".dset"
    # 存储data数据
    data.save(save_data_name)
    model = SeqModel(data)
    # check to load pretrained model
    if data.use_crf:
        pretrain_model_path = os.path.join('model_snapshot', 'lan_crf.model')
    else:
        pretrain_model_path = os.path.join('model_snapshot', 'lan.model')
    if data.use_pre_trained_model and os.path.exists(pretrain_model_path):
        model.load_state_dict(torch.load(pretrain_model_path))
        print("load pretrained model success:%s" % pretrain_model_path)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("--------pytorch total params--------")
    print(pytorch_total_params)
    optimizer = None
    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=data.HP_lr,
                              momentum=data.HP_momentum, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    else:
        print("Optimizer illegal: %s" % (data.optimizer))
        exit(1)
    best_dev = -10
    best_test = -10
    no_imprv_epoch = 0
    ## start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx, data.HP_iteration))  # print (self.train_Ids)
        # every 5 epoch decay learning rate
        if idx % 5 == 0:
            optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        total_loss = 0
        ## set model in train model
        model.train()
        model.zero_grad()
        start = 0
        end = start + data.HP_batch_size
        train_epochs = []
        while end <= len(data.train_Ids):
            train_epochs.append((start, end))
            start = end
            end = end + data.HP_batch_size
        if end > len(data.train_Ids) > start:
            train_epochs.append((start, len(data.train_Ids)))
        for sample_id, (start, end) in enumerate(train_epochs):
            instance = data.train_Ids[start: end]
            sample_loss = 0
            batch_word, batch_word_len, _, batch_word_recover, batch_label, mask, input_label_seq_tensor = batchify_with_label(
                instance, data.HP_gpu, data)
            instance_count += 1
            loss, tag_seq = model.neg_log_likelihood_loss(
                batch_word, batch_word_len, batch_label, mask, input_label_seq_tensor)
            sample_loss += loss.item()
            total_loss += loss.item()
            print("Epoch:%s,no_imprv_epoch:%s,Instance: %s" % (
                idx, no_imprv_epoch, sample_id))
            right, whole = predict_check(tag_seq, batch_label, mask, data.use_crf)
            print("               loss: %.4f, acc: %s/%s=%.4f" % (
                loss.item(), right, whole, (right + 0.) / whole * 100))

            if sample_loss > 1e8 or str(sample_loss) == "nan":
                print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                exit(1)
            sys.stdout.flush()
            loss.backward()
            if data.whether_clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), data.clip_grad)
            optimizer.step()
            model.zero_grad()
            # break
        epoch_finish = time.time()
        if total_loss > 1e8 or str(total_loss) == "nan":
            print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
            exit(1)
        speed, acc, report, f_value, \
        ner_acc, ner_p, ner_r, ner_f = evaluate(data, model, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        if data.seg:
            current_score = f_value
            # current_score = sent_f1
            print("Dev: time: %.2fs, speed: %.2fst/s;\n"
                  "acc: %.4f, f_value: %.4f\n"
                  "ner_acc: %.4f, ner_p: %.4f, ner_r: %.4f, ner_f: %.4f\n"
                  "current f1:%.4f" % (
                      dev_cost, speed, acc, f_value,
                      ner_acc, ner_p, ner_r, ner_f, current_score
                  ))
        else:
            current_score = acc
            print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f" % (
                dev_cost, speed, acc))

        # ## decode test
        speed, acc, report, f_value, \
        ner_acc, ner_p, ner_r, ner_f = evaluate(data, model, "test")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish
        if data.seg:
            print("Test: time: %.2fs, speed: %.2fst/s;\n"
                  "acc: %.4f, f_value: %.4f\n"
                  "ner_acc: %.4f, ner_p: %.4f, ner_r: %.4f, ner_f: %.4f\n"
                  "current f1:%.4f" % (
                      dev_cost, speed, acc, f_value,
                      ner_acc, ner_p, ner_r, ner_f, current_score
                  ))
        else:
            print("Test: time: %.2fs speed: %.2fst/s; acc: %.4f" % (
                dev_cost, speed, acc))

        if current_score > best_dev:
            if data.seg:
                best_test = f_value
                # best_test = sent_f1
                print("Exceed previous best avg f score:", best_dev)
            else:
                best_test = acc
                print("Exceed previous best acc score:", best_dev)
            if data.use_crf:
                result_file = "result_crf.txt"
                model_name = data.model_dir + "_crf.model"
            else:
                result_file = "result.txt"
                model_name = data.model_dir + ".model"
            with open(result_file, 'w', encoding='utf-8') as w:
                w.write(
                    "Save current best model in file:%s, iteration:%s/%s, best_test_f_score:%.5f\n"
                    "ner:\n"
                    "   precision:%.5f, recall:%.5f, f1_score:%.5f\n"
                    "%s\n\n" % (
                        model_name, idx, data.HP_iteration, best_test,
                        ner_p, ner_r, ner_f,
                        report))
            print("Save current best model in file:", model_name)
            torch.save(model.state_dict(), model_name)
            best_dev = current_score
            no_imprv_epoch = 0
        else:
            # early stop
            no_imprv_epoch += 1
            if no_imprv_epoch >= 10:
                print("early stop")
                print("Current best f score in dev", best_dev)
                print("Current best f score in test", best_test)
                break

        if data.seg:
            print("Current best f score in dev", best_dev)
            print("Current best f score in test", best_test)
        else:
            print("Current best acc score in dev", best_dev)
            print("Current best acc score in test", best_test)
        gc.collect()


def load_model_decode(data, name):
    print("Load Model from file: ", data.model_dir)
    model = SeqModel(data)
    ## load model need consider if the model trained in GPU and load in CPU, or vice versa
    # if not gpu:
    #     model.load_state_dict(torch.load(model_dir))
    #     # model.load_state_dict(torch.load(model_dir), map_location=lambda storage, loc: storage)
    #     # model = torch.load(model_dir, map_location=lambda storage, loc: storage)
    # else:
    #     model.load_state_dict(torch.load(model_dir))
    #     # model = torch.load(model_dir)
    model.load_state_dict(torch.load(data.load_model_dir))

    print("Decode %s data, nbest: %s ..." % (name, data.nbest))
    start_time = time.time()
    speed, acc, p, r, f, pred_results, pred_scores = evaluate(data, model, name, data.nbest)
    end_time = time.time()
    time_cost = end_time - start_time
    if data.seg:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
            name, time_cost, speed, acc, p, r, f))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f" % (name, time_cost, speed, acc))
    return pred_results, pred_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with NCRF++')
    # parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--config', help='Configuration File')
    # NER
    parser.add_argument('--train_dir', default='dataset2/train.tsv', help='train_file')
    parser.add_argument('--dev_dir', default='dataset2/dev.tsv', help='dev_file')
    parser.add_argument('--test_dir', default='dataset2/test.tsv', help='test_file')
    parser.add_argument('--model_dir', default='model_snapshot/lan', help='model_file')
    parser.add_argument('--seg', default=True)

    parser.add_argument('--word_emb_dir', default='pretrained', help='word_emb_dir')
    parser.add_argument('--word_emb_file', default='sgns.renmin.bigram-char', help='word_emb_dir')
    parser.add_argument('--norm_word_emb', default=False)
    parser.add_argument('--number_normalized', default=True)
    parser.add_argument('--word_emb_dim', default=100)
    parser.add_argument('--use_pre_trained_model', default=True)

    # NetworkConfiguration
    parser.add_argument('--use_crf', default=False)
    parser.add_argument('--word_seq_feature', default='LSTM')

    # TrainingSetting
    parser.add_argument('--status', default='train')
    parser.add_argument('--optimizer', default='SGD')
    parser.add_argument('--iteration', default=100)
    parser.add_argument('--batch_size', default=10)
    parser.add_argument('--ave_batch_loss', default=False)

    # Hyperparameters
    parser.add_argument('--hidden_dim', default=400)
    parser.add_argument('--dropout', default=0.5)
    parser.add_argument('--lstm_layer', default=0)
    parser.add_argument('--bilstm', default=True)
    parser.add_argument('--learning_rate', default=0.01)
    parser.add_argument('--lr_decay', default=0.05)
    parser.add_argument('--label_embedding_scale', default=0.0025)
    parser.add_argument('--num_attention_head', default=5)
    # 0.05
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--whether_clip_grad', default=True)
    parser.add_argument('--clip_grad', default=5)
    parser.add_argument('--l2', default=1e-8)
    parser.add_argument('--gpu', default=True)
    parser.add_argument('--seed', default=42)

    args = parser.parse_args()
    with open('./model_snapshot/lan.args', 'wb') as f:
        pickle.dump(args, f)
    print("seg:%s" % args.seg)
    print("use_pre_trained_model:%s" % args.use_pre_trained_model)

    seed_num = int(args.seed)
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.backends.cudnn.enabled = False
    np.random.seed(seed_num)

    data = Data()
    # print(data.initial_feature_alphabets())
    data.HP_gpu = torch.cuda.is_available()
    # if data.HP_gpu:
    #     torch.cuda.set_device(int(args.device_id))
    #     print("current gpu:%s" % torch.cuda.current_device())
    data.read_config(args)
    status = data.status.lower()
    print("Seed num:", seed_num)

    if status == 'train':
        print("MODEL: train")
        data_initialization(data)
        data.generate_instance('train')
        data.generate_instance('dev')
        data.generate_instance('test')
        data.build_pretrain_emb()
        train(data)
    elif status == 'decode':
        print("MODEL: decode")
        data.load(data.dset_dir)
        data.read_config(args.config)
        data.show_data_summary()
        data.generate_instance('raw')
        print("nbest: %s" % (data.nbest))
        decode_results, pred_scores = load_model_decode(data, 'raw')
        if data.nbest:
            data.write_nbest_decoded_results(decode_results, pred_scores, 'raw')
        else:
            data.write_decoded_results(decode_results, 'raw')
    else:
        print("Invalid argument! Please use valid arguments! (train/test/decode)")
