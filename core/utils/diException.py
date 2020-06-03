# -*- encoding: utf-8 -*-
"""
@File    :   diException.py    
@Author  :   gfyulx@163.com
@Version :    1.0
@Description:
@Modify TIme:  2020/5/28 16:24
"""

class DiException(Exception):

    def __init__(self, message):
        super().__init__(self)
        self.message = message

    def __str__(self):
        return self.message