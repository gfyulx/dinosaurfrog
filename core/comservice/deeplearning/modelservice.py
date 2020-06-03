# -*- encoding: utf-8 -*-
"""
@File    :   modelservice.py    
@Author  :   gfyulx@163.com
@Version :    1.0
@Description:network model interface
@Modify TIme:  2020/6/1 15:15
"""

from abc import ABCMeta, abstractmethod
import tensorflow as tf
import os

from core.utils.diException import DiException


class ModelService(metaclass=ABCMeta):

    _session=None

    def __init__(self,**args):
        """
        desc:
        init model env 、config
        """
        _session=self.init_session(args)
        self.env_config(**args)


    def env_config(self,**args):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    def init_session(self,train_type=0,**args):
        """
        :param train_type:
        0 :Mirrored Strategy
        1:MultiWorker Mirrored Strategy
        2:Parameter Server Strategy
        other:not use strategy
        :return:
        """
        strategy=None
        if train_type==0:
            strategy=tf.distribute.MirroredStrategy()
        elif train_type==1:
            strategy=tf.distribute.experimental.MultiWorkerMirroredStrategy()
        elif train_type==2:
            strategy=tf.distribute.experimental.ParameterServerStrategy()
        return strategy


    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def train(self):
        with self._session.scope():
            pass

    def accuracy(self):
        pass

    def saveModel(self):
        pass


