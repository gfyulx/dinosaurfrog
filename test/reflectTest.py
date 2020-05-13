#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 10/17/2019 11:21 AM 
# @Author : gfyulx 
# @File : reflectTest.py.py 
# @description:
import inspect

import WordSimilarCompute

if __name__=='__main__':
    object=getattr(WordSimilarCompute,"WordSimilarCompute")
    print(object)
    classA=getattr(object,"loadFile")
    print(classA)
    params=inspect.signature(classA).parameters;
    print(params)
    if params:
        listP=[]
        for x in params:

