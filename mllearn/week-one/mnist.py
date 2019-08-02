#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 8/1/2019 10:35 AM 
# @Author : gfyulx 
# @File : mnist.py 
# @description: mnist分类

import tensorflow as tf
import numpy as np
from PIL import Image

import tensorflow.examples.tutorials.mnist.input_data as input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 为shape填充正太分布标准差在0.1*2内的随机值
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # 使用卷积核W
    # strides中[1,stride,stride,1]头尾固定为1,中间两个代表步长.
    # SAME表示边界不足时使用0填充，VALID时不填充直接丢弃不足的数据不计算。
    # x格式为[batch,in_height,in_weight,in_channel]，多通道输入，单通道输出时，通道值相加。
    # w格式为[filter_height,filter_weight,in_channel,out_channel] 其中in_channel与x一致


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 池化层
    # 最大池化层直接计算每个区域的最大值做为输出值


def MNISTV2():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder("float", [None, 784])
    y_ = tf.placeholder("float", [None, 10])  # 实际值

    gpu_options = tf.GPUOptions(allow_growth=True)  # 设置gpu显示根据需要自动增长，默认为申请所有显存
    # gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4) #设置使用的显示比例
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    W_conv1 = weight_variable([5, 5, 1, 32])  # 卷积核定义，输出为32通道。 --定义了32个特征数组
    #tf.summary.histogram('w_conv1', W_conv1)   #tfboard中查看变量的变化
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 输出为28*28*32
    h_pool1 = max_pool_2x2(h_conv1)  # 输出为14*14*32 #第一层，卷积+最大池化

    # 第二层
    W_conv2 = weight_variable([5, 5, 32, 64])  # 输入为32通道，输出为64输出通道，即64个特征。
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 输出为14*14*64
    h_pool2 = max_pool_2x2(h_conv2)  # 输出为7*7*64

    # 全连接层1
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # reshape里-1代表自动计算维度
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # dropout防止过拟合
    keep_prob = tf.placeholder("float")
    h_fcl_drop = tf.nn.dropout(h_fc1, rate=1 - keep_prob)
    # 输出层
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fcl_drop, W_fc2) + b_fc2)

    # 测试评估
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    tf.summary.scalar('loss', cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # 使用Adam梯度下降
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # 取aixs=N维中的最大值的索引
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # cast转为float后求平均值
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()
    writer=tf.summary.FileWriter("logs/",sess.graph)
    sess.run(init)
    for i in range(100):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0},session=sess)
            print("step %d, training accuracy %g" % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        rs = sess.run(merged,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        writer.add_summary(rs, i)
    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images[:1000], y_: mnist.test.labels[:1000], keep_prob: 1.0},session=sess))
    saver=tf.train.Saver()
    #save_path=saver.save(sess,"model/model.ckpt")
#测试从模型中恢复
def recover():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder("float", [None, 784])
    y_ = tf.placeholder("float", [None, 10])  # 实际值

    gpu_options = tf.GPUOptions(allow_growth=True)  # 设置gpu显示根据需要自动增长，默认为申请所有显存
    # gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4) #设置使用的显示比例
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    W_conv1 = weight_variable([5, 5, 1, 32])  # 卷积核定义，输出为32通道。 --定义了32个特征数组
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 输出为28*28*32
    h_pool1 = max_pool_2x2(h_conv1)  # 输出为14*14*32 #第一层，卷积+最大池化

    # 第二层
    W_conv2 = weight_variable([5, 5, 32, 64])  # 输入为32通道，输出为64输出通道，即64个特征。
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 输出为14*14*64
    h_pool2 = max_pool_2x2(h_conv2)  # 输出为7*7*64

    # 全连接层1
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # reshape里-1代表自动计算维度
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # dropout防止过拟合
    keep_prob = tf.placeholder("float")
    h_fcl_drop = tf.nn.dropout(h_fc1, rate=1 - keep_prob)
    # 输出层
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fcl_drop, W_fc2) + b_fc2)

    # 测试评估
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # 使用Adam梯度下降
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # 取aixs=N维中的最大值的索引
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # cast转为float后求平均值
    init = tf.global_variables_initializer()
    sess.run(init)
    saver=tf.train.Saver()   #恢复模型，并测试
    saver.restore(sess,"model/model.ckpt")
    # print("test accuracy %g" % accuracy.eval(feed_dict={
    #     x: mnist.test.images[:1000], y_: mnist.test.labels[:1000], keep_prob: 1.0}, session=sess))
    # #给一幅图片，测试输出
    result=tf.argmax(y_conv,1)
    fileName="test_3.jpg"   #label=1  默认为3通道，转为灰度
    image_raw = tf.io.gfile.GFile(fileName, 'rb').read()
    image_raw = tf.image.decode_png(image_raw)
    image_raw=tf.image.rgb_to_grayscale(image_raw)
    input=tf.image.resize(image_raw,[28,28],method=0)
    input=np.asarray(input.eval(session=sess),dtype='float')
    input=np.reshape(input,[-1,784])
    print("result", sess.run(result, feed_dict={x: input, keep_prob: 1.0}))

    #print("result",sess.run(result,feed_dict={x: mnist.test.images[:10],keep_prob:1.0}))
    #print("label",mnist.test.labels[:10])
# v1t版本的mnist 约91%
def MNISTV1():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder("float", [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)  # 预测值  --激活函数softmax
    y_ = tf.placeholder("float", [None, 10])  # 实际值
    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(allow_growth=True)  # 设置gpu显示根据需要自动增长，默认为申请所有显存
    # gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4) #设置使用的显示比例
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  # tf.reduce_sum求和 交叉熵
    train_setp = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    sess.run(init)
    for i in range(10000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train_setp, feed_dict={x: batch_x, y_: batch_y})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # 取aixs=N维中的最大值的索引
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # cast转为float后求平均值
    print("accuracy:%f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
    MNISTV2()
    #recover()
