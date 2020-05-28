#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 7/25/2019 9:44 AM
# @Author : gfyulx
# @File : GlobalVariable.py
# @description:

from enum import Enum
import time
import configparser
from . import __version__


SYS_START_TIME = time.strftime("%Y-%m-%d %H:%M:%S")
CONFIG = configparser.ConfigParser()
PROJECT_NAME = "DI"

PROJECT_VERSION = str(__version__)
PROJECT_ENV = {}

COMS={}

class LOG_LEVEL(Enum):
    """
    log level
    """
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"

class DATA_ENCODING(Enum):
    """
    define data encode type
    """
    utf = "utf-8"
    gbk = "gbk"
