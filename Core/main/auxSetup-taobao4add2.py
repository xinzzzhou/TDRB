#!/usr/bin/env python
# coding:utf-8
from __future__ import absolute_import
from __future__ import division
import argparse

from Core.tool.Draw import Draw
from Core.tool.Evaluator import auxEvaluator
from Core.tool.Dataset import DatasetTaobao
from Core.method.auxConvNCF1paper import auxConvNCF
from Core.tool.Log import Log
import tensorflow as tf
from time import time
import os
import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count

# use conv instead of pooling

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1' #use GPU with ID=0
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9 # maximun alloc gpu50% of MEM
# config.gpu_options.allow_growth = True #allocate dynamically
# np.seterr(divide='ignore',invalid='ignore')

################INPUT1
pwd = os.getcwd()
# project_path=os.path.abspath(os.path.dirname(pwd)+os.path.sep)+"/ConvNCF-master7/"
project_path=os.path.abspath(os.path.dirname(pwd)+os.path.sep+"..")+"/"
# project_path = pwd+"/"
################INPUT2
def parse_args():
    parser = argparse.ArgumentParser(description="Run MF-BPR.")
    parser.add_argument('--path', nargs='?', default=project_path+'Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='taobao',
                        help='Choose a dataset.')
    parser.add_argument('--model', nargs='?', default='auxConvNCF4',
                        help='Choose model: ConvNCF')
    parser.add_argument('--verbose', type=int, default=50,
                        help='Interval of evaluation.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=600,
                        help='Number of epochs.')
    parser.add_argument('--embed_size', type=int, default=32,
                        help='Embedding size.')
    # parser.add_argument('--hidden_size', type=int, default=64,
    #                     help='Number of hidden neurons.')
    parser.add_argument('--dns', type=int, default=1,
                        help='number of negative sample for each positive in dns.')
    parser.add_argument('--regs', nargs='?', default='[0.01,10,1]',
                        help='Regularization for user and item embeddings, fully-connected weights, CNN filter weights.')
    parser.add_argument('--task', nargs='?', default='',
                        help='Add the task name for launching experiments')
    parser.add_argument('--num_neg', type=int, default=1,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr_embed', type=float, default=0.05,
                        help='Learning rate for embeddings.')
    parser.add_argument('--lr_net', type=float, default=0.0001,
                        help='Learning rate for CNN.')
    parser.add_argument('--net_channel', nargs='?', default='[32,32,32,32,32,32]',
                        help='net_channel, should be 6 layers here')
    parser.add_argument('--pretrain', type=int, default=1,
                        help='Use the pretraining weights or not')
    parser.add_argument('--ckpt', type=int, default=0,
                        help='Save the pretraining weights or not')
    parser.add_argument('--train_auc', type=int, default=0,
                        help='Calculate train_auc or not')
    parser.add_argument('--keep', type=float, default=1.0,
                        help='keep probability in training')
    parser.add_argument('--len_sequence', type=int, default=2,
                        help='length of the action sequence')
    parser.add_argument('--seq_model', nargs='?', default='gru',
                        help='type of sequencal model')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='layers of sequencal model')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='hidden_dim of sequencal model')
    parser.add_argument('--num_action', type=int, default=4,
                        help='total number of actions')
    parser.add_argument('--gate', type=float, default=1,
                        help='gate for auxiliary sequencal')
    parser.add_argument('--loss_threshold', type=float, default=0.000001)
    parser.add_argument('--choose', type=int, default = 2)
    return parser.parse_args()

#---------- data preparation -------

# data sampling and shuffling

# input: dataset(Mat, List, Rating, Negatives), batch_choice, num_negatives
# output: [_user_input_list, _item_input_pos_list]
def sampling(dataset):
    _user_input, _item_input_pos = [], []
    # for (u, i) in dataset.trainMatrix.keys():
    #     # positive instance
    #     key = str(u) + "-" + str(i)
    #     actions = dataset.trainAuxiliaryPosActionMap.get(key)
    #     for lens in range(len(actions)):
    #         _user_input.append(u)
    #         _item_input_pos.append(i)
    for (u, i) in dataset.trainMatrix.keys():
        _user_input.append(u)
        _item_input_pos.append(i)
    return _user_input, _item_input_pos

def all_zero(ar, type):
    if type == "str":
        for a in ar:
            if a != '0':
                return False
    elif type == "int":
        for a in ar:
            if a != 0:
                return False
    return True

def sample_not_zero_duan(arr):
    idxs = []
    idx = 0
    for i in arr:
        if not all_zero(i, "str"):
            idxs.append(idx)
        idx += 1
    if idxs == None:
        return ['0', '0']
    return arr[np.random.randint(len(idxs))]

def splitAttention_train(arr):
    aaa = []
    for a in arr:
        a = np.array(a.split("\t"), dtype=np.int32)
        aaa.append(a.astype(np.int32))
    return aaa

def get_action_input(user,item, type):
    gates = []
    actionss = []
    # actions = [0, 0]  ####fixed!!!!
    key = str(user) + "-" + str(item)
    if type == "pos":
        if _dataset.trainAuxiliaryPosActionMap.get(key) != None:
            actions = _dataset.trainAuxiliaryPosActionMap.get(key)
            actions = splitAttention_train(actions)
            # if len(actions) > 1:
            #     print("a")
            for a in actions:
                gate = np.zeros([1, 1])
                if not all_zero(a, "int"):
                    gate = np.full([1, 1], args.gate)
                gates.append(gate)
                actionss.append(a)
        else:
            gates.append(np.zeros([1, 1]))
            actionss.append([0,0])
    elif type == "neg":
        if _dataset.trainAuxiliaryNegActionMap.get(key) != None:
            actions = _dataset.trainAuxiliaryNegActionMap.get(key)
            actions = splitAttention_train(actions)
            for a in actions:
                gate = np.zeros([1, 1])
                if not all_zero(a, "int"):
                    gate = np.full([1, 1], args.gate)
                gates.append(gate)
                actionss.append(a)
        else:
            gates.append(np.zeros([1, 1]))
            actionss.append([0, 0])
    return actionss, gates

def shuffle(samples, batch_size, dataset, model):
    global _user_input
    global _item_input_pos
    global _batch_size
    global _model
    global _dataset
    global _index
    _user_input, _item_input_pos = samples
    _batch_size = batch_size
    _model = model
    _dataset = dataset
    _index = list(range(len(_user_input)))
    num_batch = len(_user_input) // _batch_size

    # res = []
    # for i in range(num_batch):
        # if i == 12:
        #     print("a")
        # res.append(_get_train_batch(i))

    pool = Pool(cpu_count())
    res = pool.map(_get_train_batch, range(num_batch))
    pool.close()
    pool.join()
    user_list = [r[0] for r in res]
    item_pos_list = [r[1] for r in res]
    user_dns_list = [r[2] for r in res]
    item_dns_list = [r[3] for r in res]
    action_input_pos = [r[4] for r in res]
    action_input_neg = [r[5] for r in res]
    return user_list, item_pos_list, user_dns_list, item_dns_list, action_input_pos, action_input_neg

def lcm(x, y):
   if x > y:
       greater = x
   else:
       greater = y
   while(True):
       if((greater % x == 0) and (greater % y == 0)):
           lcm = greater
           break
       greater += 1
   return lcm

def _get_train_batch(i):
    user_batch, item_batch, action_batch, action_gate = [], [], [], []
    user_neg_batch, item_neg_batch, action_neg_batch, action_neg_gate = [], [], [], []
    begin = i * _batch_size
    for idx in range(begin, begin + _batch_size):
        tmp_u = _user_input[_index[idx]]
        tmp_i = _item_input_pos[_index[idx]]
        actions, gates = get_action_input(tmp_u, tmp_i, "pos")

        gtItem = _dataset.testRatings[tmp_u][1]
        j = np.random.randint(_dataset.num_items)
        while j in _dataset.trainList[tmp_u] or j == gtItem:
            j = np.random.randint(_dataset.num_items)
        tmp_negaction, tmp_neggate = get_action_input(tmp_u, j, "neg")

        m = 0
        for aa in actions:
            n = 0
            for bb in tmp_negaction:
                user_batch.append(tmp_u)
                user_neg_batch.append(tmp_u)
                item_batch.append(tmp_i)
                item_neg_batch.append(j)
                action_batch.append(aa)
                action_neg_batch.append(bb)
                action_gate.append(gates[m])
                action_neg_gate.append(tmp_neggate[n])
                n += 1
            m += 1
    # bbb=np.array(action_batch)
    # c = np.array(action_batch).reshape([-1, 4])
    return np.array(user_batch)[:,None], np.array(item_batch)[:,None], \
           np.array(user_neg_batch)[:,None], np.array(item_neg_batch)[:,None],\
           np.array(action_batch), np.array(action_neg_batch),\
           np.array(action_gate).reshape([-1, 1]), np.array(action_neg_gate).reshape([-1, 1])

# def _get_train_batch(i):
#     user_batch, item_batch, action_batch, action_gate = [], [], [], []
#     user_neg_batch, item_neg_batch, action_neg_batch, action_neg_gate = [], [], [], []
#     begin = i * _batch_size
#     for idx in range(begin, begin + _batch_size):
#         user_batch.append(_user_input[_index[idx]])
#         item_batch.append(_item_input_pos[_index[idx]])
#         action, gate = get_action_input(_user_input[_index[idx]], _item_input_pos[_index[idx]], "pos")
#         action_batch.append(action)
#         action_gate.append(gate)
#         for dns in range(_model.dns):
#             user = _user_input[_index[idx]]
#             user_neg_batch.append(user)
#             # negtive k
#             gtItem = _dataset.testRatings[user][1]
#             j = np.random.randint(_dataset.num_items)
#             while j in _dataset.trainList[_user_input[_index[idx]]] or j == gtItem:
#                 j = np.random.randint(_dataset.num_items)
#             item_neg_batch.append(j)
#             action_neg, neg_gate = get_action_input(user, j, "neg")
#             action_neg_batch.append(action_neg)
#             action_neg_gate.append(neg_gate)
#     return np.array(user_batch)[:,None], np.array(item_batch)[:,None], \
#            np.array(user_neg_batch)[:,None], np.array(item_neg_batch)[:,None],\
#            np.array(action_batch), np.array(action_neg_batch),\
#            np.array(action_gate).reshape([-1, 1]), np.array(action_neg_gate).reshape([-1, 1])

#---------- training -------

# training
def training(model, dataset, args, logging, saver = None): # saver is an object to save pq
    with tf.Session() as sess:
        # initialized the save op
        # ckpt_save_path = args.logpath+"Pretrain/dropout/%s_embed_%s/" %  (args.dataset, "_".join(map(str,model.nc)))
        # ckpt_save_file = str(TRAIN_KEEP_PROB) + '_'.join(map(str, eval(args.regs)))
        # if not os.path.exists(ckpt_save_path):
        #     os.makedirs(ckpt_save_path)
        # initialize saver
        # saver_ckpt = tf.train.Saver(tf.trainable_variables())
        # pretrain or not
        sess.run(tf.global_variables_initializer())

        evalua = auxEvaluator(model, sess, dataset, args, logging)

        # restore the weights when pretrained
        if args.pretrain:
            #saver_ckpt.restore(sess, "Pretrain/MF_BPR/embed_32_32_32_32_32_32/1e-06_0_10-1440")
            model.load_parameter_MF(sess, project_path + "result/" + args.dataset + "/mfbpr/best_%s_MF.npy" % args.dataset)
            evalua.dxyeval()

        # sample the data
        samples = sampling(dataset)

        #initialize the max_ndcg to memorize the best result
        max_ndcg = 0
        max_res = " "

        last_train_loss = 0.0
        delta_train_loss = 0.0
        last_delta_train_loss = 0.0
        last_last_delta_train_loss = 0.0
        # train by epoch

        test_log = []
        validate_log = []
        # train by epoch
        for epoch_count in range(args.epochs):
            # print("start epoch", epoch_count)
            # initialize for training batches
            batch_begin = time()
            batches = shuffle(samples, args.batch_size, dataset, model)
            batch_time = time() - batch_begin

            # compute the accuracy before training
            _, prev_acc = 0,0 #training_loss_acc(model, sess, prev_batch)

            # training the model
            train_begin = time()
            train_batches = training_batch(model, sess, batches)
            train_time = time() - train_begin

            this_train_loss, this_total_loss = evalua.get_loss(train_batches)
            # print("batch cost:", batch_time, 'train cost:', train_time)

            if epoch_count % args.verbose == 0 or args.epochs - epoch_count < 10:
                # _, ndcg, cur_res , this_loss, hr, auc, validate_result, test_result = evalua.output_evaluate(train_batches, epoch_count, batch_time, train_time, prev_acc, delta_loss)
                _, ndcg, cur_res, hr, auc, validate_result, test_result = evalua.output_evaluate(train_batches, epoch_count, batch_time, train_time, prev_acc, delta_train_loss)

                # print and log the best result
                if max_ndcg < ndcg:
                    max_ndcg = ndcg
                    max_res = cur_res
                    # saver_ckpt.save(sess, ckpt_save_path+ckpt_save_file, global_step=epoch_count)
                    # print("saved best", epoch_count)


                validate_log.append(validate_result)
                test_log.append(test_result)

            delta_train_loss = last_train_loss - this_train_loss
            logging.info("[Epoch %d] train_loss:%.4f, delta_train_loss:%.4f, loss:%4f" % (
                epoch_count, this_train_loss, delta_train_loss, this_total_loss))

            # logging.info("delta_loss:" + str(delta_loss))
            if epoch_count > 2:
                if abs(delta_train_loss) < args.loss_threshold and abs(
                        last_delta_train_loss) < args.loss_threshold and abs(
                    last_last_delta_train_loss) < args.loss_threshold:
                    break

            last_last_delta_train_loss = last_delta_train_loss
            last_delta_train_loss = delta_train_loss
            last_train_loss = this_train_loss

        idx = 0
        for i in validate_log:
            logging.info("Epoch " + str(idx) + " [validate_result]:" + str(i))
            idx += 1
        idx = 0
        for i in test_log:
            logging.info("Epoch " + str(idx) + " [test_result]:" + str(i))
            idx += 1

        print("best:" + max_res)
        logging.info("best:" + max_res)

# input: batch_index (shuffled), model, sess, batches
# do: train the model optimizer
def training_batch(model, sess, batches):
    user_input, item_input_pos, user_dns_list, item_dns_list, action_input_pos, action_input_neg  = batches
    # dns for every mini-batch
    # dns = 1, i.e., BPR
    if model.dns == 1:
        item_input_neg = item_dns_list
        # for BPR training
        for i in range(len(user_input)):
            feed_dict = {model.user_input: user_input[i],
                         model.item_input_pos: item_input_pos[i],
                         model.item_input_neg: item_input_neg[i],
                         model.action_input_pos: action_input_pos[i],
                         model.action_input_neg: action_input_neg[i],
                         # model.one: one[i],
                         model.keep_prob: TRAIN_KEEP_PROB}
            # if i == 13:
            #     print("a")
            sess.run(model.get_optimizer(), feed_dict)
        model.prepared = True
    # dns > 1, i.e., BPR-dns
    elif model.dns > 1:
        item_input_neg = []
        for i in range(len(user_input)):
            # get the output of negtive sample
            feed_dict = {model.user_input: user_dns_list[i],
                         model.item_input_neg: item_dns_list[i],
                         model.keep_prob: TRAIN_KEEP_PROB}
            output_neg = sess.run(model.output_neg, feed_dict)
            # select the best negtive sample as for item_input_neg
            item_neg_batch = []
            for j in range(0, len(output_neg), model.dns):
                item_index = np.argmax(output_neg[j : j + model.dns])
                item_neg_batch.append(item_dns_list[i][j : j + model.dns][item_index][0])
            item_neg_batch = np.array(item_neg_batch)[:,None]
            # for mini-batch BPR training
            feed_dict = {model.user_input: user_input[i],
                         model.item_input_pos: item_input_pos[i],
                         model.item_input_neg: item_neg_batch,
                         model.keep_prob: TRAIN_KEEP_PROB}
            sess.run(model.get_optimizer(), feed_dict)
            item_input_neg.append(item_neg_batch)
    return user_input, item_input_pos, item_input_neg, action_input_pos, action_input_neg

if __name__ == '__main__':

    # initialize logging
    args = parse_args()
    args.net_channel = eval(args.net_channel)
    args.logpath = project_path + "/result/" + args.dataset + "/" + args.model + "/"
    TRAIN_KEEP_PROB = args.keep
    # initialize dataset
    dataset = DatasetTaobao(args.path + args.dataset +"/"+ str(args.len_sequence) +"/"+args.dataset+"-" + str(args.len_sequence))
    # dataset = DatasetTaobao(args.path + args.dataset +"/"+ str(args.len_sequence) +"/"+args.dataset)

    choose = args.choose
    logging = Log(args).get_logging()
    # initialize models
    model = auxConvNCF(dataset.num_users, dataset.num_items, args)
    model.build_graph()
    # start trainging
    training(model, dataset, args, logging)
    # if choose == 1:
    #     args.lr_net = 0.0001
    #     for j in range(6):
    #         logging = Log(args).get_logging()
    #         # initialize models
    #         model = auxConvNCF(dataset.num_users, dataset.num_items, args)
    #         model.build_graph()
    #         # start trainging
    #         training(model, dataset, args, logging)
    #         args.lr_net = args.lr_net * 10
    #
    # else:
    #     args.lr_net = 0.0001
    #     for j in range(3):
    #         logging = Log(args).get_logging()
    #         # initialize models
    #         model = auxConvNCF(dataset.num_users, dataset.num_items, args)
    #         model.build_graph()
    #         # start trainging
    #         training(model, dataset, args, logging)
    #         args.lr_net = args.lr_net * 10
    #     #
    #     regs = [0.001, 0.01, 0.1, 10, 100]
    #     for i in range(5):
    #         args.regs = "[0.01,10," + str(regs[i]) + "]"
    #         args.lr_net = 0.0001
    #         for j in range(3):
    #             logging = Log(args).get_logging()
    #             # initialize models
    #             model = auxConvNCF(dataset.num_users, dataset.num_items, args)
    #             model.build_graph()
    #             # start trainging
    #             training(model, dataset, args, logging)
    #             args.lr_net = args.lr_net * 10
    #     regs = [0.001, 0.01, 0.1, 1, 100]
    #     for i in range(5):
    #         args.regs = "[0.01," + str(regs[i]) + ",1]"
    #         args.lr_net = 0.0001
    #         for j in range(3):
    #             logging = Log(args).get_logging()
    #             # initialize models
    #             model = auxConvNCF(dataset.num_users, dataset.num_items, args)
    #             model.build_graph()
    #             # start trainging
    #             training(model, dataset, args, logging)
    #             args.lr_net = args.lr_net * 10
    # if choose == 1:
    #     logging = Log(args).get_logging()
    #
    #     #initialize models
    #     model = auxConvNCF(dataset.num_users, dataset.num_items, args)
    #     model.build_graph()
    #     #start trainging
    #     training(model, dataset, args,logging)
    # elif choose == 2:
    #     args.lr_embed = 0.0001
    #     for i in range(3):
    #         args.lr_net = 0.0001
    #         for j in range(3):
    #             logging = Log(args).get_logging()
    #             # initialize models
    #             model = auxConvNCF(dataset.num_users, dataset.num_items, args)
    #             model.build_graph()
    #             # start trainging
    #             training(model, dataset, args, logging)
    #             args.lr_net = args.lr_net * 10
    #         args.lr_embed = args.lr_embed * 10
    # elif choose == 3:
    #     reg1 = 0.0001
    #     for i in range(5):
    #         reg2 = 0.0001
    #         for j in range(5):
    #             reg3 = 0.0001
    #             for k in range(5):
    #                 regs = "[" + str(reg1) + "," + str(reg2) + "," + str(reg3) + "]"
    #                 # initialize logging
    #                 logging = Log(args).get_logging()
    #                 # initialize models
    #                 model = auxConvNCF(dataset.num_users, dataset.num_items, args)
    #                 model.build_graph()
    #                 # start trainging
    #                 training(model, dataset, args)
    #                 reg3 = reg3 * 10
    #             reg2 = reg2 * 10
    #         reg1 = reg1 * 10
    # else:
    #     lr_net = [0.01, 0.01]
    #     lr_embed = [0.001, 0.01]
    #     for i in range(2):
    #         args.lr_net = lr_net[i]
    #         args.lr_embed = lr_embed[i]
    #         logging = Log(args).get_logging()
    #         model = auxConvNCF(dataset.num_users, dataset.num_items, args)
    #         model.build_graph()
    #         # start trainging
    #         training(model, dataset, args, logging)
