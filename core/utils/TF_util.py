#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 7/25/2019 9:44 AM
# @Author : gfyulx
# @File : TFutil.py
# @description:

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (signature_constants, signature_def_utils, tag_constants, utils)



class TFReader:

    def __init__(self, x_dimension=0, labelNums=0, y_dimension=0, z_dimension=0):
        self.__x_dimension = x_dimension
        self.__y_dimension = y_dimension
        self.__z_dimension = z_dimension
        self.labelNums = labelNums

    def readAndDecode(self, filename, batchSize=5, isTrain=False,num_threads=2,type=TF_PARSE_METHOD.simple,
                      image_shape=None, flip_lr=False, flip_ud=False, brightness=False,
                       bright_delta=0.2, contrast=False, contrast_lower=0.2, contrast_up=1.8, hue=False,
                       hue_delta=0.2, saturation=False, saturation_low=0.2, saturation_up=1.8, standard=False):
        if os.path.isfile(filename):
            filename_queues = tf.train.string_input_producer([filename])
        else:
            tmps = []
            for cur_file in os.listdir(filename):
                tmps.append(os.path.join(filename, cur_file))
            filename_queues = tf.train.string_input_producer(tmps)
        reader = tf.TFRecordReader()
        _, datas = reader.read(filename_queues)
        if type==TF_PARSE_METHOD.simple:
            raw_data, label = self._parse_simple(datas)
        else:
            raw_data, label = self._pares_img(datas, image_shape, flip_lr, flip_ud, brightness,
                       bright_delta, contrast, contrast_lower, contrast_up, hue,
                       hue_delta, saturation, saturation_low, saturation_up, standard)
        if isTrain:
            batchSize = 1
        data_batch, label_batch = tf.train.shuffle_batch([raw_data, label],
                                                         batch_size=batchSize, capacity=batchSize * 4,
                                                         num_threads=num_threads, min_after_dequeue=batchSize)
        return data_batch, label_batch

    def _parse_simple(self, dt):
        dics = {
            'label': tf.io.FixedLenFeature([1], tf.int64),
            'datas': tf.io.FixedLenFeature([self.__x_dimension], tf.float32)}
        rs = tf.io.parse_single_example(serialized=dt, features=dics)
        data = rs['datas']
        label = tf.one_hot(rs['label'][0], depth=self.labelNums, on_value=1)
        return data, label

    def _pares_img(self, dt, image_shape, flip_lr, flip_ud, brightness,bright_delta, contrast,
                   contrast_lower, contrast_up, hue, hue_delta, saturation, saturation_low,
                   saturation_up, standard):
        img_feature = {
            "image": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "height": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "width": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "depth": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "label": tf.io.FixedLenFeature(shape=[], dtype=tf.int64)
        }

        rs = tf.io.parse_single_example(serialized=dt, features=img_feature)

        # get each of feature
        raw_img = tf.decode_raw(rs['image'], tf.uint8)
        height = tf.cast(rs['height'], tf.int32)
        width = tf.cast(rs['width'], tf.int32)
        depth = tf.cast(rs['depth'], tf.int32)
        # convert feature
        img_data = tf.reshape(raw_img, [height, width, depth])
        # second step augmentation image
        img_data = augmentationImage(input_image=img_data, image_shape=image_shape, flip_lr=flip_lr,
                                     flip_ud=flip_ud, brightness=brightness, bright_delta=bright_delta,
                                     contrast=contrast, contrast_lower=contrast_lower, contrast_up=contrast_up,
                                     hue=hue, hue_delta=hue_delta, saturation=saturation,
                                     saturation_low=saturation_low, saturation_up=saturation_up, standard=standard)
        # cast image format to float32
        img_data = tf.cast(img_data, dtype=tf.float32)
        # image = tf.cast(image, dtype=tf.uint8)
        label = tf.cast(rs['label'], tf.int64)
        # onehot label
        label = tf.one_hot(label, depth=self.labelNums, on_value=1)
        return img_data, label
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
            dataset = dataset.map(self.parse_map).repeat(self.FLAGS.epoch).batch(self.FLAGS.batch_size).shuffle(buffer_size=1000)
        else:
            dataset = dataset.map(self.parse_map).repeat(self.FLAGS.epoch).batch(self.FLAGS.batch_size)
        # iterator = dataset.make_one_shot_iterator()
        # img_input, label = iterator.get_next()
        # return img_input, label
        return dataset.prefetch(buffer_size=int(1000/self.FLAGS.batch_size))




def augmentationImage(input_image, image_shape, flip_lr=False, flip_ud=False, brightness=False,
                       bright_delta=0.2, contrast=False, contrast_lower=0.2, contrast_up=1.8, hue=False,
                       hue_delta=0.2, saturation=False, saturation_low=0.2, saturation_up=1.8, standard=False):
    # enlarge image to same size
    resize_img = tf.image.resize_images(images=input_image, size=(int(1.2*image_shape[0]), int(1.2*image_shape[1])))

    try:
        # crop image
        distort_img = tf.image.random_crop(value=resize_img, size=image_shape, seed=0)
        # flip image in left and right
        if flip_lr:
            distort_img = tf.image.random_flip_left_right(image=distort_img, seed=0)
        # flip image in left and right
        if flip_ud:
            distort_img = tf.image.random_flip_up_down(image=distort_img, seed=0)
        # adjust image brightness
        if brightness:
            distort_img = tf.image.random_brightness(image=distort_img, max_delta=bright_delta)
        # # adjust image contrast
        if contrast:
            distort_img = tf.image.random_contrast(image=distort_img, lower=contrast_lower, upper=contrast_up)
        # adjust image hue
        if hue:
            distort_img = tf.image.random_hue(image=distort_img, max_delta=hue_delta)
        #  adjust image saturation
        if saturation:
            distort_img = tf.image.random_saturation(image=distort_img, lower=saturation_low, upper=saturation_up)
        # reduce pixel value to (0., 1.)
        normalize_img = distort_img / 255.
        # image standard process
        if standard:
            normalize_img = tf.image.per_image_standardization(image=normalize_img)
            # resize image
        output_img = tf.image.resize_images(images=normalize_img, size=image_shape[:-1])
        return output_img
    except Exception as err:
        print(err)


def initSess(runSpace, sess):
    tf.compat.v1.summary.FileWriter(runSpace.getLogPath(True), sess.graph)
    saver = tf.compat.v1.train.Saver(max_to_keep=5)
    if os.path.exists(runSpace.getModelPath() + "/checkpoint"):
        ckpt_state = tf.train.get_checkpoint_state(runSpace.getModelPath())
        saver.restore(sess, ckpt_state.model_checkpoint_path)
    else:
        init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
        sess.run(init_op)
    return saver


def exportModel(sess, check_path, export_path, **args):
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(check_path))
    model_signature = signature_def_utils.build_signature_def(
        inputs={"input": utils.build_tensor_info(args['input'])},
        outputs={"output": utils.build_tensor_info(args['output'])},
        method_name=signature_constants.PREDICT_METHOD_NAME)
    legacy_init_op = tf.group(
        tf.compat.v1.tables_initializer(), name='legacy_init_op')
    if os.path.exists(export_path):
        Common.removeURI(export_path)
    try:
        builder = saved_model_builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            sess, [tag_constants.SERVING],
            clear_devices=True,
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    model_signature,
            },
            legacy_init_op=legacy_init_op)
        builder.save()
    except Exception as e:
        raise ComputerException(Common.formatException(e))


def writeTFRecord(savePath, sourcePath, format, label_length=None, img_height=None, img_width=None, img_depth=None, label_forward=True):
    """
    transform data format to tfrecord files.
    :param savePath:
    :param sourcePath:
    :param format:
    :return:
    """
    import cv2

    images, labels, oriClass, bin_filenames = getDataLabel(sourcePath, format, label_length, img_height, img_width, img_depth, label_forward)
    # len_per_shard = 1000  # 每个tfrecord文件的记录数
    # num_shards = int(np.ceil(len(images) / len_per_shard))
    # for index in xrange(num_shards):
    # 文件编号使用tf-00000n ,代表编号
    index = 0
    filename = os.path.join(savePath, 'tfrecord-%.5d' % (index))
    writer = tf.python_io.TFRecordWriter(filename)
    for file, label, bin_file in zip(images, labels, bin_filenames):
        image = None
        height = None
        width = None
        channel = None
        if format in ['labelfile', 'folder']:
            image = cv2.imread(file)
            height = image.shape[0]
            width = image.shape[1]
            depth = image.shape[2]
        elif format in ['binary']:
            image = file
            file = bin_file
            height = img_height
            width = img_width
            depth= img_depth
        # # normalize
        # image *= 1 / 255.
        # 此处一定要裁剪成模型输入的大小
        # im1 = cv2.resize(im, (224, 224))
        byte_img = image.tobytes()
        sample = tf.train.Example(features=tf.train.Features(
            feature={'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[byte_img])),
                     'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                     'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                     'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[depth])),
                     'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                     'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[file.encode()]))}))
        serialized = sample.SerializeToString()
        writer.write(serialized)
    writer.close()

    # sample image
    sample_img = None
    class_names = {}
    if format in ['labelfile', 'folder']:
        sample_img = cv2.imread(images[0])
    elif format in ['binary']:
        sample_img = images[0]

    for index, class_name in enumerate(oriClass):
        class_names[index] = class_name

    tdf = TFDataModel()
    tdf.setStructureXdimension(sample_img.shape[0])
    tdf.setStructureYdimension(sample_img.shape[1])
    tdf.setStructureZdimension(sample_img.shape[2])
    tdf.setStructureMaxValue(255)
    tdf.setStructureDataNums(len(images))
    tdf.setStructureLabelNums(len(oriClass))
    tdf.setStructureLabelMap(class_names)


    return filename, tdf.getStructure()


def getDataLabel(folder, format, label_length, img_height, img_width, img_channel, label_forward):
    """
    get dataset and label under floder
    :param floder:
    :param format:
    :return:
    """
    import os
    from sklearn.preprocessing import LabelEncoder

    # 获取文件夹下的所有文件【文件/文件夹】
    contents = os.listdir(folder)
    # image name | image tensor
    images = []
    # image label
    labels = []
    # image class
    classes = []
    # bin filename
    filename = []

    if format == 'binary':
        assert 'meta.txt' in contents

        image_array, label_array, filename= unpicpkleBin(folder, label_length, img_height, img_width, img_channel, label_forward)
        for i in range(image_array.shape[0]):
            raw_img = image_array[i]
            reshape_img = np.reshape(raw_img, (img_channel, img_height, img_width))
            trans_img = reshape_img.transpose((1, 2, 0))
            images.append(trans_img)

        labels = label_array[:, 0].tolist()

        # read classes
        with open(os.path.join(folder, 'meta.txt'), 'r') as fr:
            all_lines = fr.readlines()
            for i, class_name in enumerate(all_lines):
                classes.append(class_name.strip('\n'))
            fr.close()

    if format == 'labelfile':
        assert 'label.txt' in contents

        with open(os.path.join(folder, 'label.txt')) as file:
            all_lines = file.readlines()
            for index, line in enumerate(all_lines):
                # 空格：‘’或者逗号：‘,’分割
                # [image_name, class_label]
                line = line.strip('\n').split(',')
                images.append(os.path.join(folder, line[0]))
                labels.append(line[-1])
            # classes = list(set(labels))
        # 字符串类别转化为数字类别
        lf = LabelEncoder().fit(labels)
        labels = lf.transform(labels).tolist()
        classes = lf.classes_
        filename = images

    elif format == 'folder':
        # 类别 如：哈巴狗/熊狮
        classes = [x for x in contents if os.path.isdir(os.path.join(folder, x))]
        # 对于所有的类别
        for i in classes:
            # 读取类别文件夹下的所有文件
            files = os.listdir(os.path.join(folder, i))
            # 遍历
            for ii, file in enumerate(files, 1):
                # 读取图片
                images.append(os.path.join(folder, i, file))
                # 转为数字标签
                labels.append(i)
        # 字符串类别转化为数字类别
        lf = LabelEncoder().fit(labels)
        labels = lf.transform(labels).tolist()
        filename = images

    return images, labels, classes, filename


def unpicpkleBin(bin_path, label_length, img_height, img_width, img_channel, label_forward=True):
    """
    unpickle binary file
    :param bin_path:
    :param label_length: label length bytes
    :param img_length: image length
    :param img_wigth: image width
    :param img_channel: image channel num
    :param label_head: label is head
    :return:
    """
    # get bin file list
    file_list = os.listdir(bin_path)
    bin_list = [file for file in file_list if os.path.splitext(file)[1] == '.bin']
    image_vec_bytes = label_length + img_height * img_width * img_channel
    image_length = img_height * img_width * img_channel
    filename_list = []

    labels = np.zeros((0, label_length), dtype=np.uint8)
    images = np.zeros((0, image_length), dtype=np.uint8)
    for bin_file in bin_list:
        with open(os.path.join(bin_path, bin_file), 'rb') as f:
            bin_data = f.read()
        data = np.frombuffer(bin_data, dtype=np.uint8)
        data = data.reshape(-1, image_vec_bytes)
        filename_list.extend([os.path.join(bin_path, bin_file) for i in range(data.shape[0])])
        # save label and image data
        if label_forward:
            label_image = np.hsplit(data, [label_length])
            label = label_image[0]
            image = label_image[1]
        else:
            image_label = np.hsplit(data, [image_length])
            label = image_label[1]
            image = image_label[0]
        # stack array
        labels = np.vstack((labels, label))
        images = np.vstack((images, image))


    return images, labels, filename_list

