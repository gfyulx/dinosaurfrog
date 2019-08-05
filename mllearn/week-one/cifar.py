#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 8/2/2019 10:26 AM 
# @Author : gfyulx 
# @File : cifar.py 
# @description: CIFAR图像分类 --tensorflow

import tensorflow as tf

from six.moves import xrange
import numpy as np
import os.path
import time
from datetime import datetime
import re
import os
import urllib
from urllib import request
import tarfile
import tensorflow.python.platform
from tensorflow.python.platform import gfile

FLAGS = tf.app.flags.FLAGS  # 定义tensorflow的变量定义 dict

tf.app.flags.DEFINE_string('train_dir', "./logs/cifar", '训练日志目录')
tf.app.flags.DEFINE_string('model_dir', "./model/cifar", '模型保存目录')
tf.app.flags.DEFINE_integer("max_steps", 10000, '最大训练次数')
tf.app.flags.DEFINE_boolean('log_device_placement', False, '')

tf.app.flags.DEFINE_integer('batch_size', 64, '批次大小')
tf.app.flags.DEFINE_string('data_dir', '../../data/cifar', '训练数据目录')
tf.app.flags.DEFINE_integer('log_frequency', 10, '输出日志间隔')

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
# "https://www.jianshu.com/p/4ed7f7b15736""
# 清理目录
if gfile.Exists(FLAGS.train_dir):
    gfile.DeleteRecursively(FLAGS.train_dir)
gfile.MakeDirs(FLAGS.train_dir)

# 记录全局训练步数变量
global_step = tf.train.get_or_create_global_step()
# 图片大小  24*24
IMAGE_SIZE = 24
# 10个分类
NUM_CLASSES = 10
# 每次训练使用的图片数量
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
# 每次评估使用的图片数量
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

TOWER_NAME = 'tower'


MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.

def dataCheck():
    destDir = FLAGS.data_dir()
    if not os.path.exists(destDir):
        os.mkdir(destDir)
    if not os.path.exists(FLAGS.model_dir):
        os.mkdir(FLAGS.model_dir)
    fileName = DATA_URL.split('/')[-1]
    filePath = os.path.join(destDir, fileName)
    if not os.path.exists(filePath):
        # 文件不存在，尝试下载
        print("file not exists,downloading from url %s" % DATA_URL)
        filePath, _ = urllib.request.urlretrieve(DATA_URL, filePath)

        print("Successfully downloaded!")
        # 解压
        tarfile.open(filePath, 'r:gz').extractall(destDir)


# 按批次加载数据
def load_data(batch_size):
    if not FLAGS.data_dir:
        raise ValueError('data_dir must be set!')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    fileNames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
    # 检查文件是否缺失
    for f in fileNames:
        if not gfile.Exists(f):
            raise ValueError("Failed to find file:", f)
    # 合并文件成sequence
    fileNameQueue = tf.train.string_input_producer(fileNames)
    read_input = read_cifar10(fileNameQueue)

    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # 随机裁剪图像，增加训练模型的鲁棒性
    distorted_images = tf.random_crop(reshaped_image, [height, width, 3])  # 28*28*3的图片
    distorted_images = tf.image.random_flip_left_right(distorted_images)  # 随机水平翻转图片
    distorted_images = tf.image.random_brightness(distorted_images, max_delta=63)  # 随机调整亮度
    distorted_images = tf.image.random_contrast(distorted_images, lower=0.2, upper=1.8)  # 随机调度对比度
    float_image = tf.image.per_image_standardization(distorted_images)  # 白化操作

    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

    # 返回一个样本队列image,labels
    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size)


def read_cifar10(filenameQueue):
    # 解析cifar10数据集文件
    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()
    # 类别 0-9 10个分类
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3

    # 图片大小
    image_bytes = result.height * result.width * result.depth
    # 返回记录的格式label+image
    record_bytes = label_bytes + image_bytes
    # CIFAR10文件固定 按label+image排列成2进制
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    # 从文件队列中读取一条数据
    result.key, value = reader.read(filenameQueue)
    # 转换record_bytes为张量
    record_bytes = tf.decode_raw(value, tf.uint8)
    # 切出记录里的标签值
    result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
    # 标签后的数据为图像数据，取出后转换shape .因为cifar10图片格式为depth,height,width
    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.width])
    # 转换图片格式为height,width,depth
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result


# 构建图像和标签的队列
def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size):
    num_preprocess_threads = 16
    images, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                                 num_threads=num_preprocess_threads,
                                                 capacity=min_queue_examples + 3 * batch_size,
                                                 min_after_dequeue=min_queue_examples)
    tf.summary.image('images', images)
    return images, tf.reshape(label_batch, [batch_size])


# 计算图构建
def interface(images):
    # 第一层conv1卷积层
    with tf.variable_scope('conv1') as scope:  # 为每层定义一个层名
        # 5*5卷积核 3通道，数量64个
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64], stddev=1e-4, wd=0.0)
        # 按kernel卷积核 ，1步长做卷积
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        # 为卷积操作加上偏置
        bias = tf.nn.bias_add(conv, biases)
        # relu
        conv1 = tf.nn.relu(bias, name=scope.name)
        # 创建激活函数conv1r tfboard event summary
        _activation_summary(conv1)

    # 第二层max_pool池化层
    # 3*3池化，步长为2
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # normal-局部响应归一化LRN层
    # 对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        # 卷积核 5*5,64个
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    # max_pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3-全连接层  384节点
    with tf.variable_scope('local3') as scope:
        dim = 1
        for d in pool2.get_shape()[1:].as_list():
            dim *= d
        # 将单个样本的特征拼成列向量
        reshape = tf.reshape(pool2, [FLAGS.batch_size, dim])
        weights = _variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)

    # local4-全连接层,192个节点
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)

    # 最后输出层使用softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', shape=[192, NUM_CLASSES], stddev=1 / 192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        # 使用sotmax,可使用tf.nn.softmax()
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)
    return softmax_linear


def _variable_with_weight_decay(name, shape, stddev, wd):
    '''
    帮助创建一个权重衰减的初始化变量
    请注意，变量是用截断的正态分布初始化的
    只有在指定了权重衰减时才会添加权重衰减

    Args:
    name: 变量的名称
    shape: 整数列表
    stddev: 截断高斯的标准差
    wd: 加L2Loss权重衰减乘以这个浮点数.如果没有，此变量不会添加权重衰减.

    Returns:
    变量张量
    '''
    var = _variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _variable_on_cpu(name, shape, initializer):
    '''
    帮助创建存储在CPU内存上的变量
    ARGS：
     name：变量的名称
     shape：整数列表
     initializer：变量的初始化操作
    返回：
     变量张量
    '''
    with tf.device('/gpu:0'):  # 用 with tf.device 创建一个设备环境, 这个环境下的 operation 都统一运行在环境指定的设备上.
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _activation_summary(x):
    '''
    为激活创建summary
    添加一个激活直方图的summary
    添加一个测量激活稀疏度的summary

    ARGS：
     x：张量
    返回：
     没有
    '''
    # 如果这是多GPU训练，请从名称中删除'tower_ [0-9] /'.这有助于张量板上显示的清晰度.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


# 损失函数定义

def loss(logits, labels):
    '''
    将L2_loss添加到训练变量上
    返回：float类型 的损失张量
    :param logits:
    :param labels:
    :return:
    '''
    labels = tf.cast(labels, tf.int64)
    # 计算交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                   name='total_loss')
    cross_entropy_mean = tf.reduce_mean(cross_entropy+1e-10, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # 总损失定义为交叉熵加上所有权重的权重衰减项（L2)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def train(total_loss, global_step):
    '''
    训练 CIFAR-10模型

    创建一个optimizer并应用于所有可训练变量. 为所有可训练变量添加移动平均值.
    ARGS：
     total_loss：loss()的全部损失
     global_step：记录训练步数的整数变量
    返回：
     train_op：训练的op
    '''

    # 学习率变量
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # 根据步骤以指数方式衰减学习率
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    tf.summary.scalar('learning_rate', lr)

    loss_averages_op = _add_loss_summaries(total_loss)

    # 计算梯度
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # 添加训练直方图
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # 添加梯度直方图
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    #跟踪变量移动平均值
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    with tf.control_dependencies([apply_gradient_op]):
        variable_averages_op=variable_averages.apply(tf.trainable_variables())

    return variable_averages_op

def _add_loss_summaries(total_loss):
    '''
    往CIFAR-10模型中添加损失summary
    为所有损失和相关summary生成移动平均值，以便可视化网络的性能

    ARGS：
     total_loss：loss()的全部损失
    返回：
     loss_averages_op：用于生成移动平均的损失
    '''
    loss_averages=tf.train.ExponentialMovingAverage(0.9,name='avg')
    losses=tf.get_collection('losses')
    loss_averages_op=loss_averages.apply(losses+[total_loss])

    #将损失添加到summary中预测
    for l in losses+[total_loss]:
        tf.summary.scalar(l.op.name+' (raw)',l)
        tf.summary.scalar(l.op.name,loss_averages.average(l))

    return loss_averages_op


####main#####
images, labels = load_data(batch_size=FLAGS.batch_size)
logits = interface(images)
loss = loss(logits, labels)
train_op=train(loss,global_step)
#开始训练

saver=tf.train.Saver(tf.global_variables())
summary_op=tf.summary.merge_all()

init=tf.global_variables_initializer()
gpu_options = tf.GPUOptions(allow_growth=True)
sess=tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement,gpu_options=gpu_options))
sess.run(init)

# 调用run或者eval去执行read之前，必须调用tf.train.start_queue_runners来将文件名填充到队列.否则read操作会被阻塞到文件名队列中有值为止
tf.train.start_queue_runners(sess=sess)
summary_writer=tf.summary.FileWriter(FLAGS.train_dir,graph=sess.graph)

for step in xrange(FLAGS.max_steps):
    start_time=time.time()
    _,loss_value=sess.run([train_op,loss])
    duration=time.time()-start_time
    assert not np.isnan(loss_value),'Model diverged with loss=NaN'

    if step % 100 ==0:
        num_example_per_step=FLAGS.batch_size
        examples_per_sec=num_example_per_step/duration
        sec_per_batch=float(duration)
        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

    if step % 1000 == 0:
        # 添加summary日志
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

    # 定期保存模型检查点
    if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.model_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

print("Success end train!")

