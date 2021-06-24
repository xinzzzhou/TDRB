#!/usr/bin/env python
# coding:utf-8
from __future__ import absolute_import
from __future__ import division
import numpy as np
import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

class ConvNCF:
    def __init__(self, num_users, num_items, args):
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = args.embed_size
        self.lr_embed = args.lr_embed
        self.lr_net = args.lr_net
        self.hidden_size = args.hidden_size
        self.nc = eval(args.net_channel)
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.lambda_weight = regs[2]
        self.dns = args.dns
        self.train_auc = args.train_auc
        self.prepared = False

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape = [None, 1], name = "user_input")
            self.item_input_pos = tf.placeholder(tf.int32, shape = [None, 1], name = "item_input_pos")
            self.item_input_neg = tf.placeholder(tf.int32, shape = [None, 1], name = "item_input_neg")
            self.keep_prob = tf.placeholder(tf.float32, name = "keep_prob")

    def _conv_weight(self, isz, osz):
        return (weight_variable([2,2,isz,osz]), bias_variable([osz]))

    def _conv_layer(self, input, P):
        conv = tf.nn.conv2d(input, P[0], strides=[1, 2, 2, 1], padding='SAME')
        return tf.nn.relu(conv + P[1])

    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.embedding_P = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='embedding_P', dtype=tf.float32)  #(users, embedding_size)
            self.embedding_Q = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='embedding_Q', dtype=tf.float32)  #(items, embedding_size)

            # here should have 6 iszs due to the size of outer products is 64x64
            iszs = [1] + self.nc[:-1]
            oszs = self.nc
            self.P = []
            for isz, osz in zip(iszs, oszs):
                self.P.append(self._conv_weight(isz, osz))

            self.W = weight_variable([self.nc[-1], 1])
            self.b = weight_variable([1])

    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            # embedding look up
            self.embedding_p = tf.nn.embedding_lookup(self.embedding_P, self.user_input)
            self.embedding_q = tf.nn.embedding_lookup(self.embedding_Q, item_input)

            # outer product of P_u and Q_i
            self.relation = tf.matmul(tf.transpose(self.embedding_p, perm=[0, 2, 1]), self.embedding_q)
            self.net_input = tf.expand_dims(self.relation, -1)

            # CNN
            self.layer = []
            input = self.net_input
            for p in self.P:
                self.layer.append(self._conv_layer(input, p))
                input = self.layer[-1]

            # prediction
            self.dropout = tf.nn.dropout(self.layer[-1], self.keep_prob)
            self.output_layer = tf.matmul(tf.reshape(self.dropout,[-1,self.nc[-1]]), self.W) + self.b

            return self.embedding_p, self.embedding_q, self.output_layer


    def _regular(self, params):
        res = 0
        for param in params:
            res += tf.reduce_sum(tf.square(param[0])) + tf.reduce_sum(tf.square(param[1]))
        return res

    def _create_loss(self):
        with tf.name_scope("loss"):
            # BPR loss for L(Theta)
            self.p1, self.q1, self.output = self._create_inference(self.item_input_pos)
            self.p2, self.q2, self.output_neg = self._create_inference(self.item_input_neg)
            self.result = self.output - self.output_neg
            self.loss = tf.reduce_sum(tf.log(1 + tf.exp(-self.result)))

            self.opt_loss = self.loss + self.lambda_bilinear * ( tf.reduce_sum(tf.square(self.p1)) \
                                    + tf.reduce_sum(tf.square(self.q2)) + tf.reduce_sum(tf.square(self.q1)))\
                                    + self.gamma_bilinear * self._regular([(self.W, self.b)]) \
                                    + self.lambda_weight * (self._regular(self.P) + self._regular([(self.W, self.b)]))

    # used at the first time when emgeddings are pretrained yet network are randomly initialized
    # if not, the parameters may be NaN.
    def _create_pre_optimizer(self):
        self.pre_opt = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(self.loss)

    def _create_optimizer(self):
        # seperated optimizer
        var_list1 = [self.embedding_P, self.embedding_Q]
        #[self.W1,self.W2,self.W3,self.W4,self.b1,self.b2,self.b3,self.b4,self.P1,self.P2,self.P3]
        var_list2 = list(set(tf.trainable_variables()) - set(var_list1))
        opt1 = tf.train.AdagradOptimizer(self.lr_embed)
        opt2 = tf.train.AdagradOptimizer(self.lr_net)
        grads = tf.gradients(self.opt_loss, var_list1 + var_list2)
        grads1 = grads[:len(var_list1)]
        grads2 = grads[len(var_list1):]
        train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
        train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
        self.optimizer = tf.group(train_op1, train_op2)


    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_pre_optimizer()
        self._create_optimizer()

    def load_parameter_MF(self, sess, path):
        ps = np.load(path,allow_pickle=True)
        ap = tf.assign(self.embedding_P, ps[0])
        aq = tf.assign(self.embedding_Q, ps[1])
        #ah = tf.assign(self.h, np.diag(ps[2][:,0]).reshape(4096,1))
        sess.run([ap,aq])
        print("parameter loaded")

    def load_parameter_logloss(self, sess, path):
        ps = np.load(path).tolist()
        ap = tf.assign(self.embedding_P, ps['P'])
        aq = tf.assign(self.embedding_Q, ps['Q'])
        sess.run([ap,aq])
        print("logloss parameter loaded")

    def save_net_parameters(self, sess, path):
        pass

    def get_optimizer(self):
        if self.prepared:  # common optimize
            return self.optimizer
        else:
            # do a previous optimizing with none regularizations if it is the first time to optimize the neural network.
            # if not, the parameters may be NaN.
            return self.pre_opt
