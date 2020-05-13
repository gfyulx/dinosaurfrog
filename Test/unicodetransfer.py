#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 8/6/2019 2:45 PM
# @Author : gfyulx
# @File :
# @description: 全半角转换
# -*- coding: cp936 -*-
from six import unichr
import pandas as pd

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += unichr(inside_code)
    return rstring


def strB2Q(ustring):
    """半角转全角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 32:  # 半角空格直接转化
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:  # 半角字符（除空格）根据关系转化
            inside_code += 65248

        rstring += unichr(inside_code)
    return rstring


if __name__=='__main__':
    # with open("test.txt","r",encoding="utf-8") as fp:
    #     buffer=fp.read()
    #     fullBuffer=strQ2B(buffer)
    #     with open("new.txt","w+",encoding="utf-8") as fw:
    #         fw.write(fullBuffer)
    a=pd.read_csv("test.txt",sep="\t",encoding="utf-8-sig")
    print(a.all)
