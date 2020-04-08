from __future__ import print_function

import argparse
import codecs
import gc
import json
import random
import sys
import time

import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, classification_report

from joint_model.jointdata import JointData
from joint_model.jointmodel import JointModel

from utils.log import logger
from utils.metric import get_ner_fmeasure

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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


def recover_label(word_pred_variable, sent_pred_variable, word_gold_variable, sent_gold_variable,
                  mask_variable, word_recover, use_crf):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    word_gold_tag = word_gold_variable.contiguous().view(-1).cpu().data.numpy()
    mask_variable = mask_variable.contiguous().view(-1).cpu().data.numpy()
    if use_crf:
        word_pred_label = []
        for p in word_pred_variable:
            word_pred_label.extend(p)
    else:
        word_pred_variable = word_pred_variable.contiguous().view(-1).cpu().data.numpy()
        word_pred_label = [p for p, m in zip(word_pred_variable, mask_variable) if m == 1]
    word_gold_label = [g for g, m in zip(word_gold_tag, mask_variable) if m == 1]
    assert (len(word_gold_label) == len(word_pred_label))

    if use_crf:
        sent = []
        for p in sent_pred_variable:
            sent.extend(p)
        sent_pred_variable = torch.tensor(sent)
    sent_pred_variable = sent_pred_variable[word_recover]
    sent_gold_variable = sent_gold_variable[word_recover]
    sent_pred = sent_pred_variable.cpu().data.numpy()
    sent_gold = sent_gold_variable.cpu().data.numpy()
    assert len(sent_pred) == len(sent_gold)
    # sent_pred_label = [data.evidence_label_alphabet.get_instance(idx) for idx in sent_pred_tag]
    # sent_gold_label = [data.opinion_label_alphabet.get_instance(idx) for idx in sent_gold_tag]

    return sent_pred, sent_gold, word_pred_label, word_gold_label


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    logger.info(" Learning rate is set as: %s" % lr)
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
        logger.info("Error: wrong evaluate name,", name)
        exit(1)
    start_time = time.time()
    sent_pred_results = []
    sent_gold_results = []
    word_pred_results = []
    word_gold_results = []
    ## set model in eval model
    model.eval()
    for idx, instance in enumerate(instances):
        if not instance or len(instance) <= 1:
            continue
        batch_word, batch_word_len, word_perm_idx, batch_word_recover, batch_label, batch_sent_type, mask, sent_mask, input_label_seq_tensor, input_sent_type_tensor = batchify_with_label(
            instance, data.HP_gpu, data)
        with torch.no_grad():
            words_tag_seq, sent_tag_seq = model.evaluate(None, batch_word,
                                                         batch_word_len,
                                                         batch_sent_type,
                                                         mask,
                                                         sent_mask,
                                                         input_label_seq_tensor,
                                                         input_sent_type_tensor,
                                                         batch_word_recover,
                                                         word_perm_idx, need_cat=False)
        # with codecs.open("attention_input.txt", "a", "utf-8") as w:
        #     obj = ["".join([data.word_alphabet.get_instance(w_idx - 1) if w_idx != 0 else "" for w_idx in sent]) for
        #            sent in batch_word.data.cpu().numpy().tolist()]
        #     json.dump(obj, w)
        #     w.write("\n")
        sent_pred, sent_gold, word_pred_label, word_gold_label = recover_label(words_tag_seq, sent_tag_seq, batch_label,
                                                                               batch_sent_type, mask,
                                                                               batch_word_recover, data.use_crf)
        sent_pred_results.extend(sent_pred)
        sent_gold_results.extend(sent_gold)
        word_pred_results.extend(word_pred_label)
        word_gold_results.extend(word_gold_label)
    decode_time = time.time() - start_time
    sent_f1 = f1_score(sent_gold_results, sent_pred_results, average="macro")
    sent_report = classification_report(sent_gold_results, sent_pred_results,
                                        target_names=data.sentence_type_alphabet.instances, digits=4)
    speed = len(instances) / decode_time
    word_acc = accuracy_score(word_gold_results, word_pred_results)
    word_f1 = f1_score(word_gold_results, word_pred_results, average='macro')
    word_report = classification_report(word_gold_results, word_pred_results,
                                        target_names=data.label_alphabet.instances, digits=4)
    word_ner_acc, word_ner_p, word_ner_r, word_ner_f = get_ner_fmeasure(word_gold_results,
                                                                        word_pred_results,
                                                                        data.label_alphabet,
                                                                        data.tagScheme,
                                                                        need_save_matrix=name == 'test')

    return speed, word_acc, word_report, word_f1, \
           word_ner_acc, word_ner_p, word_ner_r, word_ner_f, sent_f1, sent_report


def batchify_with_label(input_batch_list, gpu, data, volatile_flag=False):
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
    sentence_types = [sent[2] for sent in input_batch_list]

    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    sent_type_tensor = autograd.Variable(torch.zeros(batch_size), volatile=volatile_flag).long()
    #
    input_sent_type_tensor = autograd.Variable(torch.zeros((batch_size, data.sentence_alphabet_size)),
                                               volatile=volatile_flag).long()
    input_label_seq_tensor = autograd.Variable(torch.zeros((batch_size, data.label_alphabet_size)),
                                               volatile=volatile_flag).long()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).byte()
    sent_mask = autograd.Variable(torch.zeros(batch_size), volatile=volatile_flag).byte()
    for idx, (seq, label, sent_type, seq_len) in enumerate(zip(words, labels, sentence_types, word_seq_lengths)):
        word_seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seq_len] = torch.LongTensor(label)
        sent_type_tensor[idx] = torch.LongTensor([sent_type])
        mask[idx, :seq_len] = torch.ones(seq_len)
        sent_mask[idx] = torch.Tensor([1])
        input_sent_type_tensor[idx, :data.sentence_alphabet_size] = torch.LongTensor(
            [i for i in range(data.sentence_alphabet_size)])
        input_label_seq_tensor[idx, :data.label_alphabet_size] = torch.LongTensor(
            [i for i in range(data.label_alphabet_size)])

    # 按照word length排序
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    # 按照新的顺序对tensor处理
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    sent_type_tensor = sent_type_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    input_label_seq_tensor = input_label_seq_tensor[word_perm_idx]
    # 原顺序索引，用于恢复tensor顺序
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        sent_type_tensor = sent_type_tensor.cuda()
        input_label_seq_tensor = input_label_seq_tensor.cuda()
        input_sent_type_tensor = input_sent_type_tensor.cuda()
        mask = mask.cuda()
        sent_mask = sent_mask.cuda()

    return word_seq_tensor, word_seq_lengths, word_perm_idx, word_seq_recover, label_seq_tensor, sent_type_tensor, mask, sent_mask, input_label_seq_tensor, input_sent_type_tensor


def train(data, weight):
    logger.info("Training model, weight:%s..." % weight)
    data.show_data_summary()
    save_data_name = data.model_dir + ".dset"
    # 存储data数据
    data.save(save_data_name)
    model = JointModel(data)
    print(model)
    # check to load pretrained model
    if data.use_crf:
        pretrain_model_path = os.path.join('model_snapshot', 'joint_crf.model')
    else:
        pretrain_model_path = os.path.join('model_snapshot', 'joint.model')
    if data.use_pre_trained_model and os.path.exists(pretrain_model_path):
        model.load_state_dict(torch.load(pretrain_model_path))
        logger.info("load pretrained model success:%s" % pretrain_model_path)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info("--------pytorch total params--------")
    logger.info(pytorch_total_params)
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
        logger.info("Optimizer illegal: %s" % (data.optimizer))
        exit(1)
    best_dev = -10
    best_test = -10
    no_imprv_epoch = 0
    ## start training
    for idx in range(data.HP_iteration):
        logger.info("Epoch: %s/%s" % (idx, data.HP_iteration))  # print (self.train_Ids)
        # every 5 epoch decay learning rate
        if idx % 5 == 0:
            optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        total_loss = 0
        ## set model in train model
        model.train()
        model.zero_grad()
        for sample_id, instance in enumerate(data.train_Ids):
            if (len(instance) == 1):
                continue
            logger.info(
                "Weight:%s, Epoch:%s,no_imprv_epoch:%s,Instance: %s" % (weight, idx, no_imprv_epoch, sample_id))
            sample_loss = 0
            batch_word, batch_word_len, word_perm_idx, batch_word_recover, batch_label, batch_sent_type, mask, sent_mask, input_label_seq_tensor, input_sent_type_tensor = batchify_with_label(
                instance, data.HP_gpu, data)
            loss = torch.zeros([1])
            if data.HP_gpu:
                loss = loss.cuda()
            instance_count += 1
            words_loss, words_tag_seq, sent_total_loss, sent_tag_seq = model.neg_log_likelihood_loss(
                None,
                batch_word, batch_word_len,
                batch_label,
                batch_sent_type,
                mask, sent_mask,
                input_label_seq_tensor,
                input_sent_type_tensor,
                batch_word_recover,
                word_perm_idx,
                need_cat=False)
            words_loss = weight[0] * words_loss
            sent_total_loss = weight[1] * sent_total_loss
            loss += sent_total_loss
            loss += words_loss
            sample_loss += sent_total_loss.item()
            sample_loss += words_loss.item()
            total_loss += sample_loss
            sent_right, sent_whole = predict_check(sent_tag_seq, batch_sent_type[batch_word_recover].view(-1),
                                                   sent_mask.view(-1), data.use_crf)
            logger.info(
                "               sent_loss: %.4f, sent_acc: %s/%s=%.4f" % (
                    sent_total_loss.item(), sent_right, sent_whole,
                    (sent_right + 0.) / sent_whole * 100))
            word_right, word_whole = predict_check(words_tag_seq, batch_label, mask, data.use_crf)
            logger.info(
                "               word_loss: %.4f, word_acc: %s/%s=%.4f" % (
                    words_loss.item(), word_right, word_whole,
                    (word_right + 0.) / word_whole * 100))
            logger.info("               sample_loss:%.4f" % sample_loss)
            if sample_loss > 1e8 or str(sample_loss) == "nan":
                logger.info("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                exit(1)
            sys.stdout.flush()
            loss.backward()
            if data.whether_clip_grad:
                import torch.nn as nn
                nn.utils.clip_grad_norm_(model.parameters(), data.clip_grad)
            optimizer.step()
            model.zero_grad()
            # break
        epoch_finish = time.time()
        if total_loss > 1e8 or str(total_loss) == "nan":
            logger.info("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
            exit(1)
        speed, word_acc, word_report, word_f1, word_ner_acc, word_ner_p, word_ner_r, word_ner_f, sent_f1, sent_report = evaluate(
            data, model, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish
        if word_ner_f <= 0:
            logger.info("f1 is negative, need to train again, epoch:%s" % idx)
            continue
        if data.seg:
            current_score = word_ner_f
            # current_score = sent_f1
            logger.info("Dev: time: %.2fs, speed: %.2fst/s;\n"
                        "word_acc: %.4f, word_f1: %.4f\n"
                        "word_ner_acc: %.4f, word_ner_p: %.4f, word_ner_r: %.4f, word_ner_f: %.4f\n"
                        "sentence_f:%.4f\n"
                        "current f1:%.4f" % (
                            dev_cost, speed, word_acc, word_f1,
                            word_ner_acc, word_ner_p, word_ner_r, word_ner_f, sent_f1, current_score
                        ))
        else:
            current_score = word_ner_acc
            logger.info("Dev: time: %.2fs speed: %.2fst/s; word_acc: %.4f" % (
                dev_cost, speed, word_acc))

        # ## decode test
        speed, word_acc, word_report, word_f1, word_ner_acc, word_ner_p, word_ner_r, word_ner_f, sent_f1, sent_report = evaluate(
            data, model, "test")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish
        if data.seg:
            logger.info("Test: time: %.2fs, speed: %.2fst/s;\n"
                        "word_acc: %.4f, word_f1: %.4f\n"
                        "word_ner_acc: %.4f, word_ner_p: %.4f, word_ner_r: %.4f, word_ner_f: %.4f\n"
                        "sentence_f:%.4f\n"
                        % (
                            dev_cost, speed, word_acc, word_f1,
                            word_ner_acc, word_ner_p, word_ner_r, word_ner_f, sent_f1
                        ))
        else:
            logger.info("Test: time: %.2fs speed: %.2fst/s; word_acc: %.4f" % (
                dev_cost, speed, word_acc))

        if current_score > best_dev:
            if data.seg:
                best_test = word_ner_f
                logger.info("Exceed previous best avg f score: %s" % best_dev)
            else:
                best_test = word_ner_acc
                logger.info("Exceed previous best acc score: %s" % best_dev)
            # 保存不同weight的结果
            if data.use_crf:
                # result_file = "joint_result_crf.txt"
                result_file = "result_%s_%s.txt" % (weight[0], weight[1])
                model_name = data.model_dir + "%s_%s.model" % (weight[0], weight[1])
            else:
                # result_file = "joint_result.txt"
                result_file = "result_%s_%s_wo_crf.txt" % (weight[0], weight[1])
                model_name = data.model_dir + "%s-%s_wo_crf.model" % (weight[0], weight[1])
            with open(result_file, 'w', encoding='utf-8') as w:
                w.write(
                    "Save current best model in file:%s, iteration:%s/%s, best_test_f_score:%.5f, sent_f_score:%.5f\n"
                    "ner:\n"
                    "   precision:%.5f, recall:%.5f, f1_score:%.5f\n"
                    "%s\n\n"
                    "%s\n\n" % (
                        model_name, idx, data.HP_iteration, best_test, sent_f1,
                        word_ner_p, word_ner_r, word_ner_f,
                        word_report,
                        sent_report))
            logger.info("Save current best model in file: %s" % model_name)
            torch.save(model.state_dict(), model_name)
            best_dev = current_score
            no_imprv_epoch = 0
        else:
            # early stop
            no_imprv_epoch += 1
            if no_imprv_epoch >= 10:
                logger.info("early stop")
                logger.info("Current best f score in dev: %s" % best_dev)
                logger.info("Current best f score in test: %s" % best_test)
                break

        if data.seg:
            logger.info("Current best f score in dev: %s" % best_dev)
            logger.info("Current best f score in test: %s" % best_test)
        else:
            logger.info("Current best acc score in dev: %s" % best_dev)
            logger.info("Current best acc score in test: %s" % best_test)
        gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with NCRF++')
    # parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--config', help='Configuration File')
    # NER
    parser.add_argument('--train_dir', default='dataset/train.tsv', help='train_file')
    parser.add_argument('--dev_dir', default='dataset/dev.tsv', help='dev_file')
    parser.add_argument('--test_dir', default='dataset/test.tsv', help='test_file')
    parser.add_argument('--model_dir', default='model_snapshot/joint', help='model_file')
    parser.add_argument('--seg', default=True)

    parser.add_argument('--word_emb_dir', default='pretrained', help='word_emb_dir')
    parser.add_argument('--word_emb_file', default='sgns.renmin.bigram-char', help='word_emb_dir')
    parser.add_argument('--norm_word_emb', default=False)
    parser.add_argument('--number_normalized', default=True)
    parser.add_argument('--lstm_input_size', default=300)
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
    parser.add_argument('--attention_query_input_size', default=403)
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

    with open('./model_snapshot/joint.args', 'wb') as f:
        pickle.dump(args, f)
    logger.info("seg:%s" % args.seg)
    logger.info("use_pre_trained_model:%s" % args.use_pre_trained_model)

    seed_num = int(args.seed)
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.backends.cudnn.enabled = False
    np.random.seed(seed_num)

    data = JointData()
    # logger.info(data.initial_feature_alphabets())
    data.HP_gpu = torch.cuda.is_available()
    # if data.HP_gpu:
    #     torch.cuda.set_device(int(args.device_id))
    #     logger.info("current gpu:%s" % torch.cuda.current_device())
    data.read_config(args)
    status = data.status.lower()
    logger.info("Seed num:%s" % seed_num)

    if status == 'train':
        logger.info("MODEL: train")
        data_initialization(data)
        data.generate_instance('train')
        data.generate_instance('dev')
        data.generate_instance('test')
        data.build_pretrain_emb()
        # weight = words:sent
        # for weight_idx, weight in enumerate([(100, 100), (150, 100), (100, 150), (100, 25), (25, 100)]):
        for weight_idx, weight in enumerate([(100, 25)]):
            train(data, weight)
    else:
        logger.info("Invalid argument! Please use valid arguments! (train/test/decode)")
