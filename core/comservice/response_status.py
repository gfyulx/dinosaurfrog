# -*- coding: utf-8 -*-
"""
ResponseStatus

Author: gfyulx
date:   3/28/2019 1:55 PM
Description: 状态码枚举类
"""
from enum import Enum, unique

@unique
class responseStatus(Enum):
    OK = {"0": "success"}
    UNKNOWN_EXCEPTION = {"1": "Unknown exception!"}
    PARAM_NOT_SET = {"201": "Parameter not set!"}
    PARAM_NOT_MATCH = {"202": "Parameter not match!"}
    PARAM_DATA_EXCEPTION = {"203": "Parameter data exception!"}
    DATA_NOT_PREPARE = {"204":"data not prepare!"}
    SPACE_NOT_PREPARE = {"205": "Run space not prepare,Component is in running!"}
    COM_NOT_RUN = {"206": "Component not in running!"}
    COM_IN_RUN = {"207": "Component is in running!"}
    COM_IN_READY = {"208": "Component is not started and in ready state !"}
    SPACE_INIT_FAILED = {"210": "Run space init failed!"}
    CODE_EXCEPTON = {"400": "Code Running Exception!"}
    RUN_INTERRUPTED = {"401": "Component Running Interrupted!"}
    COM_NOT_FOUND = {"402": "component not found!"}
    METHOD_NOT_FOUND = {"404": "Method not found!"}
    EXTERNAL_EXCEPTION = {"405": "External run exception!"}
    FRAME_EXCEPTION = {"500": "Frame run exception!"}
    EXTERNAL_INTERRUPTED = {"505": "Code execution was externally interrupted!"}
    def get_code(self):
        """
        :return: Enum code
        """
        return list(self.value.keys())[0]

    def get_msg(self,ext=""):
        """
        :return: Enum message
        """
        return list(self.value.values())[0]+" "+ext
