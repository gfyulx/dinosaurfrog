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
from sklearn.preprocessing import LabelEncoder  # 用于Label编码
from sklearn.preprocessing import OneHotEncoder  # 用于one-hot编码
from tensorflow.python.platform import gfile
import cv2
import time
from datetime import datetime
from tensorflow.python import pywrap_tensorflow


FLAGS = tf.app.flags.FLAGS  # 定义tensorflow的变量定义 dict
tf.app.flags.DEFINE_string('pre_model_path', "../data/vgg_16.ckpt", '预加载vgg16模型位置')
tf.app.flags.DEFINE_string('model_dir', "./model/vgg", '模型保存目录')
tf.app.flags.DEFINE_string('train_dir', "./logs/vgg", '训练日志目录')
tf.app.flags.DEFINE_string('tfrecord_dir', "../data/tfrecord/flower/", '生成用于队列的tfrecord文件目录')
tf.app.flags.DEFINE_string('data_dir', "../data/flower_photos/", '预加载vgg16模型位置')
tf.app.flags.DEFINE_integer("max_steps", 10, '最大训练次数')
tf.app.flags.DEFINE_integer('batch_size', 1, '批次大小')
tf.app.flags.DEFINE_integer('num_example_for_train', 1000, '每次训练队列中的最小样本数量')

HEIGHT = 224
WIDTH = 224

# 清理目录
if gfile.Exists(FLAGS.train_dir):
    gfile.DeleteRecursively(FLAGS.train_dir)
gfile.MakeDirs(FLAGS.train_dir)

# 处理训练数据集
def get_file(data_dir):
    data_dir = FLAGS.data_dir
    contents = os.listdir(data_dir)
    classes = [x for x in contents if os.path.isdir(data_dir + x)]  # 文件夹为所有分类,5类
    labels = []
    images = []
    print(classes)
    for i in classes:
        files = os.listdir(data_dir + i)
        for ii, file in enumerate(files, 1):
            # img_raw=tf.gfile.FastGFile(data_dir+i+"/"+file,'rb').read()
            # img=tf.image.decode_jpeg(img_raw)
            # img.set_shape([224, 224, 3])
            images.append(data_dir + i + "/" + file)
            labels.append(i)  # 转为数字标签
    lf = LabelEncoder().fit(labels)
    data_label = lf.transform(labels).tolist()
    return images, data_label


def writeTFRecord():
    if gfile.Exists(FLAGS.tfrecord_dir):
        gfile.DeleteRecursively(FLAGS.tfrecord_dir)
    gfile.MakeDirs(FLAGS.tfrecord_dir)
    images, labels = get_file(FLAGS.data_dir)
    # print(len(images), len(labels))
    len_per_shard = 1000  # 每个tfrecord文件的记录数
    num_shards = int(np.ceil(len(images) / len_per_shard))
    for index in xrange(num_shards):
        # 文件编号使用tf-00000n ,代表编号
        filename = os.path.join(FLAGS.tfrecord_dir, 'tfrecord-%.5d' % (index))
        writer = tf.python_io.TFRecordWriter(filename)
        for file, label in zip(images[index * len_per_shard:((index + 1) * len_per_shard)],
                               labels[index * len_per_shard:((index + 1) * len_per_shard)]):
            img_raw = tf.gfile.FastGFile(file, 'rb').read()
            im = cv2.imread(file)
            im1 = cv2.resize(im, (224, 224))  # 此处一定要裁剪成模型输入的大小
            im = im1.tobytes()
            sample = tf.train.Example(features=tf.train.Features(
                feature={'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[im])),
                         "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                         'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[file.encode()]))}))
            serialized = sample.SerializeToString()
            writer.write(serialized)
        writer.close()


def parse_map(reader):
    return tf.io.parse_single_example(reader, features={'image': tf.io.FixedLenFeature([], tf.string),
                                                        'label': tf.io.FixedLenFeature([], tf.int64)})


# 读取tfrecord文件并写入队列。
def read_and_decode(fileName):
    filename_queue = tf.train.string_input_producer(fileName)
    reader = tf.TFRecordReader()
    _, serialized_sample = reader.read(filename_queue)
    # # reader=tf.data.TFRecordDataset(fileName)
    #
    # features =reader.map(parse_map)
    features = tf.io.parse_single_example(serialized_sample, features={'image': tf.io.FixedLenFeature([], tf.string),
                                                                       'label': tf.io.FixedLenFeature([], tf.int64),

                                                                       'filename': tf.io.FixedLenFeature([], tf.string)})

    # print(features)
    img = tf.decode_raw(features['image'], tf.uint8)
    img = tf.reshape(img, [HEIGHT, WIDTH, 3])
    img = tf.cast(img, tf.float32)
    label = features['label']
    label = tf.cast(label, tf.int32)
    return img, label,features['filename']


def input_data(fileName):
    img, label,filename = read_and_decode(fileName)

    num_threads = 16
    capacity = FLAGS.num_example_for_train
    #print(img, label)
    images_batch, labels_batch,filename_batch = tf.train.shuffle_batch([img, label,filename], batch_size=FLAGS.batch_size,
                                                        capacity=capacity + 3 * FLAGS.batch_size,
                                                        min_after_dequeue=capacity, num_threads=num_threads)

    return images_batch, labels_batch ,filename_batch #tf.reshape(labels_batch, [FLAGS.batch_size])


def oneHotEncoder(labels):
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat([indices, labels], 1)
    onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, 5]), 1.0, 0.0)

    return onehot_labels


def vgg_16(inputs, scope='vgg_16'):
    # inputs是224*224*3 的图像
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
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
            net = slim.dropout(net, 0.5, is_training=True, scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            net = slim.dropout(net, 0.5, is_training=True, scope='dropout7')
    net = slim.flatten(net)
    net = slim.fully_connected(net, 256, scope='new/fc8')
    net = slim.dropout(net, 0.5, is_training=True, scope='dropout8')
    net = slim.fully_connected(net, 5,  scope='new/fc9')
    # # 得到最后的预测分布
    predicted = tf.nn.softmax(net, name='classify')
    return predicted
    #return net

def losses(logits,labels):
    # 交叉熵，使用one-hot
    #predict=tf.reshape(logits, [FLAGS.batch_size, 5])
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=predict)  #使用one-hot
    # 计算损失函数
    cost = tf.reduce_mean(cross_entropy)
    return cost

def train(loss):
    var_list = []
    for var in tf.model_variables():
        if var.op.name.startswith('new'):
            var_list.append(var)
    #print(var_list)
    # 采用用得最广泛的AdamOptimizer优化器
    #optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss, var_list=var_list)
    optimizer=tf.train.GradientDescentOptimizer(0.01).minimize(loss,var_list=var_list)
    return optimizer


def main():
    data_dir = FLAGS.tfrecord_dir
    contents = os.listdir(data_dir)
    fileNames = [os.path.join(data_dir, x) for x in contents]
    images, labels,filename = input_data(fileNames)
    labels_vecs = oneHotEncoder(labels)
    feature_op = vgg_16(images, "vgg_16")
    cost_op=losses(feature_op,labels_vecs)
    tf.summary.scalar("loss", cost_op)
    train_op=train(cost_op)
    # 计算准确度
    #predicted=tf.nn.softmax(feature_op,name="classify")
    correct_pred = tf.equal(tf.argmax(feature_op, 1), tf.argmax(labels_vecs, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  #交叉熵，使用one-hot
    tf.summary.scalar("accuracy", accuracy)
    gpu_options = tf.GPUOptions(allow_growth=True)
    coord = tf.train.Coordinator()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(init)
    exclude_layer = ['vgg_16/fc8', 'new']  # 不加载最后的fc8层  --全连接到1000个分类 ## 。
    variables_to_restore = []
    for var in tf.model_variables():
        flags = 0
        for exclude in exclude_layer:
            if var.op.name.startswith(exclude):
                flags = 1
        if flags == 0:
            variables_to_restore.append(var)
    print(variables_to_restore)
    saver = tf.train.Saver(variables_to_restore)

    queue_runner = tf.train.start_queue_runners(sess=sess, coord=coord)
    saver.restore(sess, FLAGS.pre_model_path)  # 恢复vg16预训练模型中的参数值

    saver=tf.train.Saver(tf.model_variables())
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=sess.graph)
    for step in xrange(FLAGS.max_steps):
        start_time = time.time()
        loss,_,acc= sess.run([cost_op,train_op,accuracy])
        duration = time.time() - start_time
        if step % 100 == 0:
            num_example_per_step = FLAGS.batch_size
            examples_per_sec = num_example_per_step / duration
            sec_per_batch = float(duration)
            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch) accuracy=%.2f')
            print(format_str % (datetime.now(), step, loss,
                                examples_per_sec, sec_per_batch,acc))
        if step % 100 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_path = os.path.join(FLAGS.model_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
    summary_writer.close()
    coord.request_stop()
    coord.join(queue_runner)
    print("Training end!")

def test():
    gpu_options = tf.GPUOptions(allow_growth=True)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #saver = tf.train.Saver()
    sess.run(init)
    #saver.restore(sess, FLAGS.pre_model_path)
    # for var in tf.model_variables():
    #     print(var.op.name)
    #     print(sess.run(var.op.name))
    reader=pywrap_tensorflow.NewCheckpointReader('model/vgg/model.ckpt-13000')
    for var in reader.get_variable_to_shape_map():
        print(var)
        print(reader.get_tensor(var))
    # for var in tf.trainable_variables():
    #     print(var.op.name,sess.run(var))



def classify():
    file = "../data/flower_photos/test/sunflower_01.jpg"
    img_raw = tf.gfile.FastGFile(file, 'rb').read()
    im = cv2.imread(file)
    im1 = cv2.resize(im, (224, 224))  # 此处一定要裁剪成模型输入的大小
    img = im1.tobytes()
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [HEIGHT, WIDTH, 3])
    # img=np.expand_dims(img,0)
    img = tf.cast(img, tf.float32)
    # define op
    feature_op = vgg_16([img], "vgg_16")
    #class_op = losses(feature_op)
    predicted = tf.nn.softmax(feature_op)

    gpu_options = tf.GPUOptions(allow_growth=True)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # saver=tf.train.import_meta_graph('model/vgg/model.ckpt-1000.meta',clear_devices=True)

    saver.restore(sess, 'model/vgg/model.ckpt-10000')
    sess.run(init)


    result=sess.run(predicted)
    result=tf.reshape(result, [1, 5])
    print(tf.argmax(result,1))


if __name__ == '__main__':

    #writeTFRecord()
    #main()
    #classify()
    test()
    print("ok")
