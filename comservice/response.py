# -*- coding: utf-8 -*-
"""
Response

Author: gfyulx
date:   3/28/2019 1:55 PM
Description:define rest service response data
"""

import json
import time

from utils.common import extJsonEncoder
from utils.global_variable import PROJECT_NAME, CONFIG, SYS_START_TIME, PROJECT_VERSION
from .response_status import responseStatus


class response:

    __runSpace = ""
    def __init__(self):
        self.__data = {}
        self.set_status(responseStatus.OK.getCode())
        self.set_info()
        self.set_env(CONFIG.get("ENV", "indicators"))
        self.__data['request'] = {}
        self.__data['data'] = {}
        self.__data['dataType'] = "train"
        self.start_time = time.time()

    def set_runspace(self,run_space):
        self.__run_space = run_space

    def set_env(self, env):
        self.__data['env'] = env

    def set_status(self, status):
        self.__data['status'] = status

    def get_status(self):
        return int(self.__data['status'])

    def set_info(self, info="success"):
        self.__data['info'] = info

    def set_datatype(self,dtype):
        self.__data['data_type'] = dtype

    def set_KV(self, k, v,autoResume=False):
        self.__data['data'][k] = self.resume(v) if autoResume else v

    def set_request(self, rq):
        self.__data['request'] = rq

    def set_data(self, data={}):
        self.__data['data'].update(data)

    def add_data(self,data={}):
        if len(self.__data['data'])==0:
            self.set_data(data)
        else:
            self.get_model_data().update(data['datas'])

    def get_data(self):
        return self.__data['data']

    def get_model_data(self):
        return self.__data['data']['datas']

    def set_framework(self,framework):
        self.__data['framework'] = framework

    def to_string(self):
        self.__data['useTime'] = str(round((time.time() - self.start_time) * 1000, 3)) + " ms"
        self.__data['createTime'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        self.__data['__SOURCE'] = PROJECT_NAME
        self.__data['__DEBUG'] = CONFIG.get("ENV", "debug").lower()
        self.__data['__SYS_START_TIME'] = SYS_START_TIME
        self.__data['__VERSION'] = PROJECT_VERSION
        return json.dumps(self.__data, cls=extJsonEncoder,indent=4)
