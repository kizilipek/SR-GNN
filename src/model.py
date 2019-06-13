#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import math


class Model(object):
    def __init__(self, hidden_size=100, out_size=100, batch_size=100, n_node=None, l2=None, step=None, lr=None  ):

        self.hidden_size = hidden_size
        self.out_size = out_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.L2 = l2
        self.step = step
        self.lr = lr

        self.mask = tf.placeholder(dtype=tf.float32)
        self.alias = tf.placeholder(dtype=tf.int32)
        self.item = tf.placeholder(dtype=tf.int32)
        self.tar = tf.placeholder(dtype=tf.int32)


        self.stdv = 1.0 / math.sqrt(self.hidden_size)


        self.nasr_w1 = tf.get_variable('nasr_w1', [self.out_size, self.out_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_w2 = tf.get_variable('nasr_w2', [self.out_size, self.out_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_v = tf.get_variable('nasrv', [1, self.out_size], dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_b = tf.get_variable('nasr_b', [self.out_size], dtype=tf.float32, initializer=tf.zeros_initializer())

        self.embedding = tf.get_variable(shape=[n_node, hidden_size], name='embedding', dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.adj_in = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.adj_out = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])

        self.W_in = tf.get_variable('W_in', shape=[self.out_size, self.out_size], dtype=tf.float32,initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_in = tf.get_variable('b_in', [self.out_size], dtype=tf.float32,initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_out = tf.get_variable('W_out', [self.out_size, self.out_size], dtype=tf.float32, initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_out = tf.get_variable('b_out', [self.out_size], dtype=tf.float32,initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))

        with tf.variable_scope('graph_nn_model', reuse=None):
            self.loss_train, _ = self.forward()
        with tf.variable_scope('graph_nn_model', reuse=True):
            self.loss_test, self.score_test = self.forward(train=False)

        self.global_step = tf.Variable(0)
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss_train, global_step=self.global_step)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def forward(self, train=True):

        fin_state = tf.nn.embedding_lookup(self.embedding, self.item)
        cell = tf.nn.rnn_cell.GRUCell(self.out_size)
        with tf.variable_scope('gru'):
            for i in range(self.step):
                fin_state = tf.reshape(fin_state, [self.batch_size, -1, self.out_size])
                fin_state_in = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                    self.W_in) + self.b_in, [self.batch_size, -1, self.out_size])
                fin_state_out = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                     self.W_out) + self.b_out, [self.batch_size, -1, self.out_size])
                av = tf.concat([tf.matmul(self.adj_in, fin_state_in),
                                tf.matmul(self.adj_out, fin_state_out)], axis=-1)
                state_output, fin_state = \
                    tf.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(av, [-1, 2*self.out_size]), axis=1),
                                      initial_state=tf.reshape(fin_state, [-1, self.out_size]))
        re_embedding= tf.reshape(fin_state, [self.batch_size, -1, self.out_size])

        rm = tf.reduce_sum(self.mask, 1)
        last_id = tf.gather_nd(self.alias, tf.stack([tf.range(self.batch_size), tf.to_int32(rm)-1], axis=1))
        last_h = tf.gather_nd(re_embedding, tf.stack([tf.range(self.batch_size), last_id], axis=1))
        seq_h = tf.stack([tf.nn.embedding_lookup(re_embedding[i], self.alias[i]) for i in range(self.batch_size)],
                         axis=0)                                                           #batch_size*T*d
        last = tf.matmul(last_h, self.nasr_w1)
        seq = tf.matmul(tf.reshape(seq_h, [-1, self.out_size]), self.nasr_w2)
        last = tf.reshape(last, [self.batch_size, 1, -1])
        m = tf.nn.sigmoid(last + tf.reshape(seq, [self.batch_size, -1, self.out_size]) + self.nasr_b)
        coef = tf.matmul(tf.reshape(m, [-1, self.out_size]), self.nasr_v, transpose_b=True) * tf.reshape(self.mask, [-1, 1])

        b = self.embedding[1:]
        ma = tf.concat([tf.reduce_sum(tf.reshape(coef, [self.batch_size, -1, 1]) * seq_h, 1), tf.reshape(last, [-1, self.out_size])], -1)
        self.B = tf.get_variable('B', [2 * self.out_size, self.out_size],
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        y1 = tf.matmul(ma, self.B)
        logits = tf.matmul(y1, b, transpose_b=True)

        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tar - 1, logits=logits))
        self.vars = tf.trainable_variables()
        if train:
            for v in self.vars:
                print(v)
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.vars if v.name not in ['bias', 'gamma', 'b', 'g', 'beta']]) * self.L2
            loss = loss + lossL2
        return loss, logits


