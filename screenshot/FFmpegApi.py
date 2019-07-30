#!/usr/bin/env python3
# coding=utf-8

import ctypes
import os
from ctypes import *
import sys
import ctypes
import ffmpeg as ff



if __name__ == '__main__':
    # 因为dll有依赖关系，要把整个lib目录引入到PATH环境变量中
    os.environ['path'] += ";"+os.getcwd()+"\\libs\\ffmpeg"
    avdevice = os.getcwd()+"\\libs\\ffmpeg\\avdevice-58.dll"
    avformat = os.getcwd()+"\\libs\\ffmpeg\\avformat-58.dll"
    # ctypes.windll.LoadLibrary(file)
    avformatHandler = ctypes.cdll.LoadLibrary(avformat)
    avdeviceHandler = ctypes.cdll.LoadLibrary(avdevice)

    # ret=dllHandler.avcodec_configuration()
    avdeviceHandler.avdevice_register_all()
    audioDevices = avdeviceHandler.avdevice_configuration()
    # 设置AVFormatContext格式的变量
    avformatContext = avformatHandler.avformat_alloc_context()
    # 设置设备返回值列表
    ifformat = avformatHandler.av_find_input_format("dshow")
    # avDevices=ctypes.Structure(avformatHandler.AVDictionary())

    print(avformatContext)
    print(audioDevices)
    ab =AVDeviceInfoList()
    print(ab)
    # ret=avdeviceHandler.avdevice_list_input_sources(avformatContext,ifformat)
    #调用函数前需要指定函数的形参类型和返回值类型
    avdeviceHandler.avdevice_list_devices.argtypes(None,POINTER(AVDeviceInfoList))
    ret1 = avdeviceHandler.avdevice_list_devices(avformatContext,ab)
    # print(ret)
    print(ret1)
    print(ab)

