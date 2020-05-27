#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 7/25/2019 9:44 AM 
# @Author : gfyulx 
# @File : Common.py 
# @description:

import os
import datetime
from core.utils import *
from core.utils.global_variable import *
import json
import numpy as np

def mkdir(*paths):
    for path in paths:
        path = path.strip()
        path = path.rstrip("\\")
        isExists = os.path.exists(path)
        if not isExists:
            try:
                os.makedirs(path)
            except Exception as e:
                sysLog(formatException(e),LOG_LEVEL.ERROR)


def getComputerLogger():
    """
    get singleton logger to write logs
    :return:logger
    """
    import logging
    __ = logging.getLogger(CONFIG.get("ENV","project"))
    if len(__.handlers)==0:
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s]  %(message)s')
        if systemIsDebug():
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            __.setLevel(logging.DEBUG)
            __.addHandler(console_handler)
        else:
            __.setLevel(logging.INFO)
            fname = datetime.datetime.now().strftime('%Y%m%d')
            logHandler = logging.FileHandler(CONFIG.get("ENV", "logPath") + "/" + fname + ".log")
            logHandler.setFormatter(formatter)
            __.addHandler(logHandler)
    return __

def sysLog(msg, level: LOG_LEVEL = LOG_LEVEL.INFO):
    __logger = getComputerLogger()
    if level == LOG_LEVEL.INFO:
        __logger.info(msg)
    elif level == LOG_LEVEL.WARN:
        __logger.warning(msg)
    else:
        __logger.error(msg)

def formatException(e):
    res = "FILE: " + e.__traceback__.tb_frame.f_globals['__file__']
    res += ",LINE: " + str(e.__traceback__.tb_lineno)
    res += ",Err: " + str(e)
    return res

def systemIsDebug():
    return (CONFIG.get("ENV", "debug").lower() == "true")



class extJsonEncoder(json.JSONEncoder):
    """
    support json encode array and bytes
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding=DATA_ENCODING.utf.value)
        elif isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)
