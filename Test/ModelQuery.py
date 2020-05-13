#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 11/12/2019 2:52 PM 
# @Author : gfyulx 
# @File : ModelQuery.py 
# @description:

import numpy as np

def checkModel(modelname):
    data_path =modelname # 文件保存路径
    # 注意这个文件要到网上自行下载
    data_dict = np.load(data_path, encoding='latin1').item()
    keys = sorted(data_dict.keys())
    for key in keys:
        weights = data_dict[key][0]
        biases = data_dict[key][1]
        print('\n')
        print(key)
        print('weights shape: ', weights.shape)
        print('biases shape: ', biases.shape)
        print('bais:',data_dict[key][1])

if __name__=='__main__':
    checkModel("../storage/premodel/vgg/vgg16.npy")