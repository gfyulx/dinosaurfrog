#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 8/9/2019 3:09 PM 
# @Author : gfyulx 
# @File : test.py 
# @description:

import os
import tensorflow as tf
from tensorflow.contrib import slim
import tensorflow.contrib.slim.nets as nets
from six.moves import xrange
import math
import numpy as np
from sklearn.preprocessing import LabelEncoder  # 用于Label编码
from sklearn.preprocessing import OneHotEncoder  # 用于one-hot编码
from tensorflow.python.platform import gfile
import cv2
import time
from datetime import datetime



def oneHotEncoder(labels):
    batch_size=tf.size(labels)
    labels=tf.expand_dims(labels,1)
    indices=tf.expand_dims(tf.range(0,batch_size,1),1)
    concated=tf.concat([indices,labels],1)
    onehot_labels=tf.sparse_to_dense(concated,tf.stack([batch_size,5]),1.0,0.0)

    return onehot_labels

images=[[[1,0]],[[2,1]],[[3,1]],[[4,3]],[[5,3]],[[6,4]],[[7,1]],[[8,0]]]
label=[1,2,3,0,1,2,3,4,0,1]
label_vec=oneHotEncoder(label)
images=np.random.randint(0,10,[8,1,1,5])
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
images2=tf.reshape(images,[8,5])
lab=tf.argmax(label_vec,1)
x1=tf.argmax(images,1)
sess.run(label_vec)
print(sess.run(lab))
print(lab)
print(images)
print(sess.run(x1))
print(sess.run(images2))
print(sess.run(tf.argmax(images2,1)))