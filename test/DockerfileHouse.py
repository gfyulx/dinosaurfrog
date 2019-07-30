#!/usr/bin/env python3
# coding=utf-8


# 从镜像名自动生成提交github上的阿里Dockerfile文件
#

import re
import os


def deal(imageFile):
    with open(imageFile, "r") as f:
        for buff in f.readlines():
            buff = buff.strip()
            print(buff)
            image = re.match(".*/(.*):.*$", buff).group(1)
            ver = re.match(".*:(.*)", buff).group(1)
            if not os.path.exists(image):
                os.mkdir(image)
            saveFile=image+"/Dockerfile-"+ver
            with open(saveFile,"w+") as fw:
                fw.write("FROM "+buff)

if __name__ == '__main__':
    deal("images.txt")
