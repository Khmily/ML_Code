#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : tf_tb_histogram.py
# @Author: Huangqinjian
# @Date  : 2018/1/28
# @Desc  :

import tensorflow as tf
import numpy as np


def add_layer(inputs, input_size, output_size, n_layer, activation_function=None):
    """
    :param inputs: 输入的数据
    :param input_size: 输入数据的维数
    :param output_size: 输入数据的维数
    :param activation_function: 激活函数，默认为None
    :return: 返回输出值
    """
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([input_size, output_size]))
            tf.summary.histogram(layer_name + "/weights", Weights)
    with tf.name_scope('biaes'):
        biaes = tf.Variable(tf.zeros([1, output_size]) + 0.1)
        tf.summary.histogram(layer_name + "/biaes", biaes)
    with tf.name_scope('output_hat'):
        output_hat = tf.matmul(inputs, Weights) + biaes
    if activation_function is None:
        output = output_hat
    else:
        output = activation_function(output_hat)
    tf.summary.histogram(layer_name + "/output", output)
    return output


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# 模拟数据噪音
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# 经过隐藏层后的输出值
layer1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# 经过输出层后的输出值，即为最后的预测值
layer2 = add_layer(layer1, 10, 1, n_layer=2, activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean((tf.reduce_sum(tf.square(ys - layer2), reduction_indices=[1])))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()

merged = tf.summary.merge_all()

writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
    writer.add_summary(result, i)
