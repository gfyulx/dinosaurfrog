#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 8/6/2019 2:45 PM 
# @Author : gfyulx 
# @File : vgg.py 
# @description: 使用vgg-16模型做迁移学习

import os
import tensorflow as tf
from tensorflow.contrib import slim
import tensorflow.contrib.slim.nets as nets
from six.moves import xrange
import math
import numpy as np

# http://c.biancheng.net/view/1934.html
#'https://github.com/DengT1ng/vgg16_flowers/blob/master/demo.py'
# 只加载特定参数时，使用variables_to_restore

FLAGS = tf.app.flags.FLAGS  # 定义tensorflow的变量定义 dict
tf.app.flags.DEFINE_string('pre_model_path', "../data/vgg_16.ckpt", '预加载vgg16模型位置')
tf.app.flags.DEFINE_string('model_dir', "./model/vgg", '模型保存目录')
tf.app.flags.DEFINE_string('data_dir', "../data/flower_photos", '预加载vgg16模型位置')
tf.app.flags.DEFINE_integer("max_steps", 1000, '最大训练次数')
tf.app.flags.DEFINE_integer('batch_size', 64, '批次大小')
#处理训练数据集
def data_load(batch_size):
    data_dir=FLAGS.data_dir
    contents=os.listdir(data_dir)
    classes=[x for x in contents if os.path.isdir(data_dir+x)]  #文件夹为所有分类,5类
    labels=[]
    images=[]
    for i in classes:
        files=os.listdir(data_dir+i)
        for ii,file in enumerate(files,1):
            img_raw=tf.gfile.FastGFile(file,'rb').read()
            img=tf.image.decode_jpeg(img_raw)
            images.append(img.reshape((1,224,224,3)))
            labels.append(i)
        #这里如果文件太多，可能一次性造成内存负荷太大，改成queue读取为好
    return images,labels

def vgg_16(inputs, scope='vgg_16'):
    #inputs是224*224*3 的图像
    # 使用slim快速定义一个vgg16模型，以用于特征提取,.对应到vgg16模型的fc8层之前的所有层
    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.fully_connected(net, 4096, scope='fc6')
            net = slim.fully_connected(net, 4096, scope='fc7')
    return net

def main():

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    exclude_layer = ['vgg_16/fc8']  # 不加载最后的fc8层  --全连接到1000个分类 ## 。
    variables_to_restore=[]
    for var in tf.model_variables():
        print(var)
        for exclude in exclude_layer:
            if var.op.name.startwith(exclude):
                continue
            variables_to_restore.append(var)
    #variables_to_restore = slim.get_variables_to_restore(exclude=exclude_layer)
    #init_fn = slim.assign_from_checkpoint_fn(FLAGS.pre_model_path, variables_to_restore, ignore_missing_vars=True)
    print(variables_to_restore)
    saver = tf.train.Saver(variables_to_restore)
    init = tf.global_variables_initializer
    sess.run(init)   #重新加载特征
    saver.restore(sess,FLAGS.pre_model_path)
    input_=tf.placeholder(tf.float32,[None,224,224,3])
    images,labels=data_load(FLAGS.batch_size)
    num_iter = int(math.ceil(len(images) / FLAGS.batch_size))
    feature_op=vgg_16(input_,"vgg_16")
    for step in xrange(num_iter):
        batch_data=images[step*FLAGS.batch_size:(step+1)*FLAGS.batch_size]
        batch_data=np.concatenate(batch_data)
        feature=sess.run(feature_op,feed_dict={input_:batch_data})
        print(feature)
    #获取到特征
    #重新构建分类网络

if __name__=='__main__':
    main()