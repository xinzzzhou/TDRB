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

class auxConvNCF(object):
    def __init__(self, num_users, num_items, args):
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = args.embed_size
        self.lr_embed = args.lr_embed
        self.lr_net = args.lr_net
        # self.nc = eval(args.net_channel)
        self.nc = args.net_channel
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.lambda_weight = regs[2]
        self.dns = args.dns
        self.train_auc = args.train_auc
        self.prepared = False
        self.len_sequence = args.len_sequence
        self.num_actions = args.num_action
        self.seq_model = args.seq_model
        self.num_layers = args.num_layers
        self.hidden_dim = args.hidden_dim
        self.gate = args.gate

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape = [None, 1], name = "user_input")
            self.item_input_pos = tf.placeholder(tf.int32, shape = [None, 1], name = "item_input_pos")
            self.item_input_neg = tf.placeholder(tf.int32, shape = [None, 1], name = "item_input_neg")
            self.action_input_pos = tf.placeholder(tf.int32, shape=[None, self.len_sequence], name="action_input_pos")
            self.action_input_neg = tf.placeholder(tf.int32, shape=[None, self.len_sequence], name="action_input_neg")
            self.keep_prob = tf.placeholder(tf.float32, name = "keep_prob")

            self.embedding_p = tf.nn.embedding_lookup(self.embedding_P, self.user_input)
            self.embedding_q_pos = tf.nn.embedding_lookup(self.embedding_Q, self.item_input_pos)
            self.embedding_q_neg = tf.nn.embedding_lookup(self.embedding_Q, self.item_input_neg)
            self.embedding_o_pos = tf.nn.embedding_lookup(self.embedding_O, self.action_input_pos)
            self.embedding_o_neg = tf.nn.embedding_lookup(self.embedding_O, self.action_input_neg)  #shape=[bs,sequen_len,emb_size]

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
            self.embedding_O = tf.Variable(tf.truncated_normal(shape=[self.num_actions-1, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='embedding_O', dtype=tf.float32)  # (actions, embedding_size)
            tmp = tf.zeros([1, self.embedding_size])
            self.embedding_O = tf.concat([tmp, self.embedding_O], 0)
            # here should have 6 iszs due to the size of outer products is 64x64
            iszs = [1] + self.nc[:-1]
            oszs = self.nc
            self.P = []
            for isz, osz in zip(iszs, oszs):
                self.P.append(self._conv_weight(isz, osz))

            self.W = weight_variable([self.nc[-1], 1])
            self.b = weight_variable([1])

    def _regular(self, params):
        res = 0
        for param in params:
            res += tf.reduce_sum(tf.square(param[0])) + tf.reduce_sum(tf.square(param[1]))
        return res

    def _create_loss(self):
        with tf.name_scope("loss"):
            ## pos
            self.target_pos = self._cnn(self.embedding_p, self.embedding_q_pos)
            self.auxiliary_pos = self._rnn(self.embedding_o_pos, tf.reshape(self.embedding_p,[-1, self.nc[-1]]))
            a = tf.reshape(self._cnn(self.target_pos, self.auxiliary_pos), [-1, self.nc[-1]])
            self.output = tf.matmul(a, self.W) + self.b

            ## neg
            ## pos
            self.target_neg = self._cnn(self.embedding_p, self.embedding_q_neg)
            self.auxiliary_neg = self._rnn(self.embedding_o_neg, tf.reshape(self.embedding_p, [-1, self.nc[-1]]))
            a = tf.reshape(self._cnn(self.target_neg, self.auxiliary_neg), [-1, self.nc[-1]])
            self.output_neg = tf.matmul(a, self.W) + self.b

            ## result
            self.result = self.output - self.output_neg
            ##loss
            self.loss = tf.reduce_sum(tf.log(1 + tf.exp(-self.result)))
            self.opt_loss = self.loss + self.lambda_bilinear * (tf.reduce_sum(tf.square(self.embedding_p)) \
                                    + tf.reduce_sum(tf.square(self.embedding_q_pos)) + tf.reduce_sum(tf.square(self.embedding_q_neg))\
                                    + tf.reduce_sum(tf.square(self.embedding_o_pos)) + tf.reduce_sum(tf.square(self.embedding_o_neg))) \
                                    + self.gamma_bilinear * self._regular([(self.W, self.b)]) \
                                    + self.lambda_weight * (self._regular(self.P) + self._regular([(self.W, self.b)]))

    # used at the first time when emgeddings are pretrained yet network are randomly initialized
    # if not, the parameters may be NaN.
    def _create_pre_optimizer(self):
        with tf.variable_scope("preoptimizer", reuse=tf.AUTO_REUSE) as scope:
            self.pre_opt = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(self.loss)

    def _create_optimizer(self):
        # seperated optimizer
        var_list1 = [self.embedding_P, self.embedding_Q]
        var_list3 = [self.embedding_P, self.embedding_Q, self.embedding_O[0:,:]]
        var_list2 = list(set(tf.trainable_variables()) - set(var_list3))
        with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE) as scope:
            opt1 = tf.train.AdagradOptimizer(self.lr_embed)
            opt2 = tf.train.AdagradOptimizer(self.lr_net)
            grads = tf.gradients(self.opt_loss, var_list1 + var_list2)
            grads1 = grads[:len(var_list1)]
            grads2 = grads[len(var_list1):]
            train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
            train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
            self.optimizer = tf.group(train_op1, train_op2)


    def build_graph(self):
        self._create_variables()
        self._create_placeholders()
        self._create_loss()
        self._create_pre_optimizer()
        self._create_optimizer()
    def _cnn(self, embedding_p, embedding_q):
        with tf.name_scope("cnn_inference"):
            # outer product of P_u and Q_i
            relation = tf.matmul(tf.transpose(embedding_p, perm=[0, 2, 1]), embedding_q)
            net_input = tf.expand_dims(relation, -1)
            # CNN
            self.layer = []
            input = net_input
            for p in self.P:
                self.layer.append(self._conv_layer(input, p))
                input = self.layer[-1]
            # prediction
            dropout = tf.reshape(tf.nn.dropout(self.layer[-1], self.keep_prob), [-1, 1, self.nc[-1]])
            return dropout
            #return tf.matmul(dropout, self.W) + self.b, dropout

    def _rnn(self, action_input, init_state, reuse=False):
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTM(self.hidden_dim,state_is_tuple=True)
        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.hidden_dim)
        def dropout_cell():
            if(self.seq_model=='lstm'):
                cell=lstm_cell()
            else:
                cell=gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=self.keep_prob)

        with tf.variable_scope("sequenceloop", reuse=tf.AUTO_REUSE) as scope:
            self.cells = [dropout_cell() for _ in range(self.num_layers)]
            self.cell = tf.contrib.rnn.MultiRNNCell(self.cells, state_is_tuple=True)

            init_state = tuple([init_state for _ in range(self.num_layers)])
            self._outputs, _ = tf.nn.dynamic_rnn(self.cell, inputs=action_input, dtype=tf.float32, initial_state=init_state)
            self.last = self._outputs[:, -1, :]
            self.logits = tf.layers.dense(inputs=self.last, units=self.embedding_size)  # y_rnn

        return tf.reshape(self.logits,[-1,1,self.embedding_size])

    def load_parameter_MF(self, sess, path):
        ps = np.load(path,allow_pickle=True)
        ap = tf.assign(self.embedding_P, ps[0])
        aq = tf.assign(self.embedding_Q, ps[1])
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