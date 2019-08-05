#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 7/31/2019 5:45 PM 
# @Author : gfyulx 
# @File : MongoDBTest.py 
# @description:


import os
import base64
import pandas as pd
from pymongo import MongoClient


def base2img(baseStr, fileName):
    with open(fileName, "wb+") as f:
        # f.write(base64.b64decode(baseStr))
        f.write(baseStr)


if __name__ == "__main__":
    # mongoDB 用户密码远程登陆格式
    # mongodb://[username:password@]hostname[:port][/database]
    mongo_client = MongoClient("mongodb://license:linewell_license123@{0}:27017/admin".format('192.168.81.24'))
    # mongo_client = MongoClient("mongodb://192.168.81.24:27017/admin")
    # mongo_client.adb.authenticate("license", "linewell_license123", mechanism='MONGODB-CR')
    LicenseFilesDBNew = mongo_client['LicenseFilesDB']
    # LicenseFilesDBNew = mongo_client['LicenseFilesDBNew']
    LicenseFilesDBNew_new_collection = LicenseFilesDBNew['fs.chunks']
    filter = {"_id": "ObjectId(\"5c662f24a0b2801604c5d67f\")"}
    document = LicenseFilesDBNew_new_collection.find()
    fileName = "test"
    i=0
    res = []
    # [res.append(x) for x in document]
    for x in document:
        base2img(x['data'], fileName+str(i)+".png")
        i=i+1

# basecode = document[0]['data']
# print(dir(basecode),type(basecode))

# print(dir(document['data']),dir(document))


