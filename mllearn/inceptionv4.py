#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 10/15/2019 10:27 AM 
# @Author : gfyulx 
# @File : inceptionv4.py 
# @description: distribution train inceptionv4 implement by tf.keras
# saved model pb
# use tfrecords and tf.data.dataset(tf官方推荐)
import os
import json
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.callbacks import TensorBoard, Callback
from tensorflow.keras.layers import Input, Dropout, Dense, Flatten, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import optimizers
from six.moves import xrange
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.platform import gfile

"""
model.compile(target_tensors 的方式来编译目标张量，来引用定义好的张量及队列)
"""


class InceptionV4():

    def initTF(self):
        flags = tf.app.flags
        flags.DEFINE_integer("task_index", None,
                             "Worker task index, should be >= 0. task_index=0 is "
                             "the master worker task the performs the variable "
                             "initialization ")
        flags.DEFINE_integer("replicas_to_aggregate", None,
                             "Number of replicas to aggregate before parameter update"
                             "is applied (For sync_replicas mode only; default: "
                             "num_workers)")
        flags.DEFINE_string("ps_hosts", "localhost:2222",
                            "Comma-separated list of hostname:port pairs")
        flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                            "Comma-separated list of hostname:port pairs")
        flags.DEFINE_string("chief_hosts", "localhost:2223,localhost:2224",
                            "Comma-separated list of hostname:port pairs")
        flags.DEFINE_integer("num_gpus", 1, "Total number of gpus for each machine."
                                            "If you don't use GPU, please set it to '0'")
        flags.DEFINE_boolean(
            "existing_servers", False, "Whether servers already exists.")

        flags.DEFINE_string("job_name", None, "job name: worker or ps")
        flags.DEFINE_string("nodeId", None, "job id")
        flags.DEFINE_string("runMethod", "rpc", "run method")
        flags.DEFINE_string("comName", None, "padding")
        flags.DEFINE_string("runType", None, "padding")
        flags.DEFINE_string("runSpace", None, "padding")
        flags.DEFINE_string("preTrain", None, "padding")
        flags.DEFINE_string("author", None, "padding")
        flags.DEFINE_string("args", None, "padding")
        flags.DEFINE_string("notifyUrl", None, "padding")
        flags.DEFINE_string("statusPath", None, "padding")
        flags.DEFINE_string("lockerPath", None, "padding")
        flags.DEFINE_integer("workers", 2, "Number of workers")
        flags.DEFINE_integer("ps", 1, "Number of ps")
        flags.DEFINE_string('tfrecord_dir', "/Computer/cifar_data/", '生成用于队列的tfrecord文件目录')
        flags.DEFINE_string('data_dir', "../data/flower_photos/", '训练数据位置')
        flags.DEFINE_integer("HEIGHT", 299, "input height")
        flags.DEFINE_integer("WIDTH", 299, "input width")
        flags.DEFINE_string('logdir', "logs/inception/", '日志目录')
        flags.DEFINE_float('learning_rate', 0.001, '训练速率')
        flags.DEFINE_integer('batch_size', 2, '批次数量')
        flags.DEFINE_integer('epoch', 5, '迭代次数')
        self.FLAGS = flags.FLAGS

    def train(self):
        # img, label = self.generate_data(shuffle=True)
        tensorboardogs = TensorBoard(log_dir=self.FLAGS.logdir, write_graph=True, write_images=True)

        tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
        task_config = tf_config.get('task', {})
        task_index = task_config.get('index')
        is_chief = (self.FLAGS.task_index == 0)

        task_type = task_config.get('type')

        self.FLAGS.job_name = task_type
        self.FLAGS.task_index = task_index

        print("job name = %s" % self.FLAGS.job_name)
        print("task index = %d" % self.FLAGS.task_index)

        cluster_config = tf_config.get('cluster', {})
        ps_hosts = cluster_config.get('ps')
        worker_hosts = cluster_config.get('worker')
        # chief_hosts = cluster_config.get('chief')
        ps_hosts_str = ','.join(ps_hosts)
        worker_hosts_str = ','.join(worker_hosts)
        # chief_hosts_str = ','.join(chief_hosts)

        self.FLAGS.ps_hosts = ps_hosts_str
        self.FLAGS.worker_hosts = worker_hosts_str
        # self.FLAGS.chief_hosts = chief_hosts_str
        ps_spec = self.FLAGS.ps_hosts.split(",")
        worker_spec = self.FLAGS.worker_hosts.split(",")
        chief_spec = self.FLAGS.chief_hosts.split(",")
        num_workers = len(worker_spec)
        cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec,"chief":{}})

        if not self.FLAGS.existing_servers:
            server = tf.train.Server(
                cluster, job_name=self.FLAGS.job_name, task_index=self.FLAGS.task_index)
        if self.FLAGS.num_gpus > 0:
            gpu = (self.FLAGS.task_index % self.FLAGS.num_gpus)
            if  self.FLAGS.job_name == "worker":
                worker_device = "/job:worker/task:%d/gpu:%d" % (self.FLAGS.task_index, gpu)
            elif  self.FLAGS.job_name == "chief":
                worker_device = "/job:chief/task:%d/gpu:%d" % (self.FLAGS.task_index, gpu)
        elif self.FLAGS.num_gpus == 0:
            cpu = 0
            worker_device = "/job:worker/task:%d/cpu:%d" % (self.FLAGS.task_index, cpu)

        if self.FLAGS.job_name == "ps":
            server.join()
        ###train#####
        elif self.FLAGS.job_name == "worker" or self.FLAGS.job_name == "chief":
            # with tf.Session(server.target):
            with tf.device(tf.train.replica_device_setter(
                    worker_device=worker_device,
                    ps_device="/job:ps/cpu:0",
                    cluster=cluster)):
                # global_step = tf.Variable(0, name="global_step", trainable=False)
                sess_config = tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False,
                    device_filters=["/job:ps",
                                    "/job:worker/task:%d" % self.FLAGS.task_index],
                    gpu_options=tf.GPUOptions(allow_growth=True))

                # img, label = self.generate_data()
                model = self.inception_v4(num_classes=5, dropout_keep_prob=0.2,
                                          learningRate=self.FLAGS.learning_rate, weights=None, include_top=None)
                optimizer = tf.train.AdamOptimizer(learning_rate=self.FLAGS.learning_rate)
                model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                # config = tf.estimator.RunConfig(
                #     experimental_distribute=tf.contrib.distribute.DistributeConfig(
                #         train_distribute=tf.contrib.distribute.CollectiveAllReduceStrategy(
                #             num_gpus_per_worker=self.FLAGS.num_gpus),
                #         eval_distribute=tf.contrib.distribute.MirroredStrategy(
                #             num_gpus_per_worker=self.FLAGS.num_gpus)))
                #ps_strategy = tf.distribute.experimental.ParameterServerStrategy()
                config=tf.estimator.RunConfig(train_distribute=tf.distribute.experimental.MultiWorkerMirroredStrategy())
                # config = tf.estimator.RunConfig(
                #     model_dir="/Computer/",
                #     save_summary_steps=100,
                #     log_step_count_steps=100,
                #     save_checkpoints_steps=500,
                #     session_config=sess_config
                # )
                esModel = tf.keras.estimator.model_to_estimator(model,config=config)

                # model.fit(img, label, batch_size=self.FLAGS.batch_size, epochs=self.FLAGS.epoch, verbose=1,
                #           validation_split=0.3,
                #           callbacks=[tensorboardogs])
                # esModel.train(img,label,epochs=self.FLAGS.epoch,batch_size=self.FLAGS.batch_size,hooks=[])
                # esModel.export_saved_model(self.FLAGS.model_dir,inputs)    模型保存
                # Train and evaluate the model. Evaluation will be skipped if there is not an
                # "evaluator" job in the cluster.
                #esModel.train(input_fn=lambda:self.generate_data(),steps=1000)
                tf.estimator.train_and_evaluate(
                    esModel,
                    train_spec=tf.estimator.TrainSpec(input_fn=self.generate_data),
                    eval_spec=tf.estimator.EvalSpec(input_fn=self.generate_data))

    def get_file(self, data_dir):
        data_dir = self.FLAGS.data_dir
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

    def writeTFrecord(self):
        if gfile.Exists(self.FLAGS.tfrecord_dir):
            gfile.DeleteRecursively(self.FLAGS.tfrecord_dir)
        gfile.MakeDirs(self.FLAGS.tfrecord_dir)
        images, labels = self.get_file(self.FLAGS.data_dir)
        # print(len(images), len(labels))
        len_per_shard = 1000  # 每个tfrecord文件的记录数
        num_shards = int(np.ceil(len(images) / len_per_shard))
        for index in xrange(num_shards):
            # 文件编号使用tf-00000n ,代表编号
            filename = os.path.join(self.FLAGS.tfrecord_dir, 'tfrecord-%.5d' % (index))
            writer = tf.python_io.TFRecordWriter(filename)
            for file, label in zip(images[index * len_per_shard:((index + 1) * len_per_shard)],
                                   labels[index * len_per_shard:((index + 1) * len_per_shard)]):
                img_raw = tf.gfile.FastGFile(file, 'rb').read()
                im = cv2.imread(file)
                im1 = cv2.resize(im, (self.FLAGS.HEIGHT, self.FLAGS.WIDTH))  # 此处一定要裁剪成模型输入的大小
                im = im1.tobytes()
                sample = tf.train.Example(features=tf.train.Features(
                    feature={'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[im])),
                             "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                             'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[file.encode()]))}))
                serialized = sample.SerializeToString()
                writer.write(serialized)
            writer.close()

    def parse_map(self, reader):
        features = tf.parse_single_example(reader,
                                           features={'image': tf.io.FixedLenFeature([], tf.string),
                                                     'label': tf.io.FixedLenFeature([], tf.int64),
                                                     'filename': tf.io.FixedLenFeature([], tf.string)})
        # print(features)
        img = tf.decode_raw(features['image'], tf.uint8)
        img = tf.reshape(img, [self.FLAGS.HEIGHT, self.FLAGS.WIDTH, 3])
        img = tf.cast(img, tf.float32)
        label = tf.cast(features['label'], tf.int32)
        size = tf.constant([img.shape[0], img.shape[1], img.shape[2]], dtype=tf.int64)
        return img, label

    def generate_data(self, shuffle=True):
        fileName = [os.path.join(self.FLAGS.tfrecord_dir, x) for x in os.listdir(self.FLAGS.tfrecord_dir)]
        dataset = tf.data.TFRecordDataset(fileName)
        if shuffle:
            dataset = dataset.map(self.parse_map).repeat().batch(self.FLAGS.batch_size).shuffle(buffer_size=1000)
        else:
            dataset = dataset.map(self.parse_map).repeat().batch(self.FLAGS.batch_size)
        # iterator = dataset.make_one_shot_iterator()
        # img_input, label = iterator.get_next()
        # return img_input, label
        return dataset

    def preprocess_input(self, x):
        x = np.divide(x, 255.0)
        x = np.subtract(x, 0.5)
        x = np.multiply(x, 2.0)
        return x

    def conv2d_bn(self, x, nb_filter, num_row, num_col,
                  padding='same', strides=(1, 1), use_bias=False):

        channel_axis = -1  # 固定图像的channel放在最后一个维度
        x = Convolution2D(nb_filter, (num_row, num_col),
                          strides=strides,
                          padding=padding,
                          use_bias=use_bias,
                          kernel_regularizer=regularizers.l2(0.00004),
                          kernel_initializer=initializers.VarianceScaling(scale=2.0, mode='fan_in',
                                                                          distribution='normal', seed=None))(x)
        x = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(x)
        x = Activation('relu')(x)
        return x

    def block_inception_a(self, input):

        channel_axis = -1  # 固定图像的channel放在最后一个维度
        branch_0 = self.conv2d_bn(input, 96, 1, 1)

        branch_1 = self.conv2d_bn(input, 64, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 96, 3, 3)

        branch_2 = self.conv2d_bn(input, 64, 1, 1)
        branch_2 = self.conv2d_bn(branch_2, 96, 3, 3)
        branch_2 = self.conv2d_bn(branch_2, 96, 3, 3)

        branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
        branch_3 = self.conv2d_bn(branch_3, 96, 1, 1)

        x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
        return x

    def block_reduction_a(self, input):
        channel_axis = -1  # 固定图像的channel放在最后一个维度
        branch_0 = self.conv2d_bn(input, 384, 3, 3, strides=(2, 2), padding='valid')

        branch_1 = self.conv2d_bn(input, 192, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 224, 3, 3)
        branch_1 = self.conv2d_bn(branch_1, 256, 3, 3, strides=(2, 2), padding='valid')

        branch_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

        x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
        return x

    def block_inception_b(self, input):
        channel_axis = -1  # 固定图像的channel放在最后一个维度

        branch_0 = self.conv2d_bn(input, 384, 1, 1)

        branch_1 = self.conv2d_bn(input, 192, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 224, 1, 7)
        branch_1 = self.conv2d_bn(branch_1, 256, 7, 1)

        branch_2 = self.conv2d_bn(input, 192, 1, 1)
        branch_2 = self.conv2d_bn(branch_2, 192, 7, 1)
        branch_2 = self.conv2d_bn(branch_2, 224, 1, 7)
        branch_2 = self.conv2d_bn(branch_2, 224, 7, 1)
        branch_2 = self.conv2d_bn(branch_2, 256, 1, 7)

        branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
        branch_3 = self.conv2d_bn(branch_3, 128, 1, 1)

        x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
        return x

    def block_reduction_b(self, input):
        channel_axis = -1  # 固定图像的channel放在最后一个维度

        branch_0 = self.conv2d_bn(input, 192, 1, 1)
        branch_0 = self.conv2d_bn(branch_0, 192, 3, 3, strides=(2, 2), padding='valid')

        branch_1 = self.conv2d_bn(input, 256, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 256, 1, 7)
        branch_1 = self.conv2d_bn(branch_1, 320, 7, 1)
        branch_1 = self.conv2d_bn(branch_1, 320, 3, 3, strides=(2, 2), padding='valid')

        branch_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

        x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
        return x

    def block_inception_c(self, input):
        channel_axis = -1  # 固定图像的channel放在最后一个维度

        branch_0 = self.conv2d_bn(input, 256, 1, 1)

        branch_1 = self.conv2d_bn(input, 384, 1, 1)
        branch_10 = self.conv2d_bn(branch_1, 256, 1, 3)
        branch_11 = self.conv2d_bn(branch_1, 256, 3, 1)
        branch_1 = concatenate([branch_10, branch_11], axis=channel_axis)

        branch_2 = self.conv2d_bn(input, 384, 1, 1)
        branch_2 = self.conv2d_bn(branch_2, 448, 3, 1)
        branch_2 = self.conv2d_bn(branch_2, 512, 1, 3)
        branch_20 = self.conv2d_bn(branch_2, 256, 1, 3)
        branch_21 = self.conv2d_bn(branch_2, 256, 3, 1)
        branch_2 = concatenate([branch_20, branch_21], axis=channel_axis)

        branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
        branch_3 = self.conv2d_bn(branch_3, 256, 1, 1)

        x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
        return x

    def inception_v4_base(self, input):
        channel_axis = -1  # 固定图像的channel放在最后一个维度

        # 默认的Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
        # 自定义的input shape 应为 xxx,xxx,xxx  其中x,y值>70  如   150x150x3
        net = self.conv2d_bn(input, 32, 3, 3, strides=(2, 2), padding='valid')
        net = self.conv2d_bn(net, 32, 3, 3, padding='valid')
        net = self.conv2d_bn(net, 64, 3, 3)

        branch_0 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(net)

        branch_1 = self.conv2d_bn(net, 96, 3, 3, strides=(2, 2), padding='valid')

        net = concatenate([branch_0, branch_1], axis=channel_axis)

        branch_0 = self.conv2d_bn(net, 64, 1, 1)
        branch_0 = self.conv2d_bn(branch_0, 96, 3, 3, padding='valid')

        branch_1 = self.conv2d_bn(net, 64, 1, 1)
        branch_1 = self.conv2d_bn(branch_1, 64, 1, 7)
        branch_1 = self.conv2d_bn(branch_1, 64, 7, 1)
        branch_1 = self.conv2d_bn(branch_1, 96, 3, 3, padding='valid')

        net = concatenate([branch_0, branch_1], axis=channel_axis)

        branch_0 = self.conv2d_bn(net, 192, 3, 3, strides=(2, 2), padding='valid')
        branch_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(net)

        net = concatenate([branch_0, branch_1], axis=channel_axis)

        # 35 x 35 x 384
        # 4 x Inception-A blocks
        for idx in range(4):
            net = self.block_inception_a(net)

        # 35 x 35 x 384
        # Reduction-A block
        net = self.block_reduction_a(net)

        # 17 x 17 x 1024
        # 7 x Inception-B blocks
        for idx in range(7):
            net = self.block_inception_b(net)

        # 17 x 17 x 1024
        # Reduction-B block
        net = self.block_reduction_b(net)

        # 8 x 8 x 1536
        # 3 x Inception-C blocks
        for idx in range(3):
            net = self.block_inception_c(net)

        return net

    def inception_v4(self, num_classes, dropout_keep_prob, learningRate, weights, include_top):

        '''
        Creates the inception v4 network
        Args:
            num_classes: number of classes
            dropout_keep_prob: float, the fraction to keep before final layer.
        Returns:
            logits: the logits outputs of the model.
        '''

        # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
        inputs = Input((299, 299, 3))
        # inputs=tf.reshape(inputs,[-1,299,299,3])
        self.inputes = inputs

        # Make inception base
        x = self.inception_v4_base(inputs)

        # Final pooling and prediction
        if include_top:
            # 1 x 1 x 1536
            x = AveragePooling2D((8, 8), padding='valid')(x)
            x = Dropout(dropout_keep_prob)(x)
            x = Flatten()(x)
            # 1536
            x = Dense(units=num_classes, activation='softmax')(x)

        model = Model(inputs, x, name='inception_v4')

        return model


if __name__ == '__main__':
    # 默认输入图像大小为299*299*3
    model = InceptionV4()
    model.initTF()
    # model.writeTFrecord()  #生成tfreocrd
    model.train()
