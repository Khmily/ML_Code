#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : tf_tb_graph.py
# @Author: Huangqinjian
# @Date  : 2018/1/28
# @Desc  :

import tensorflow as tf


def add_layer(inputs, input_size, output_size, activation_function=None):
    """
    :param inputs: 输入的数据
    :param input_size: 输入数据的维数
    :param output_size: 输入数据的维数
    :param activation_function: 激活函数，默认为None
    :return: 返回输出值
    """
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([input_size, output_size]))
        with tf.name_scope('biaes'):
            biaes = tf.Variable(tf.zeros([1, output_size]) + 0.1)
        with tf.name_scope('output_hat'):
            output_hat = tf.matmul(inputs, Weights) + biaes
        if activation_function is None:
            output = output_hat
        else:
            output = activation_function(output_hat)
        return output


with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# 经过隐藏层后的输出值
layer1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 经过输出层后的输出值，即为最后的预测值
layer2 = add_layer(layer1, 10, 1, activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean((tf.reduce_sum(tf.square(ys - layer2), reduction_indices=[1])))
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)
