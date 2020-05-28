# -*- encoding: utf-8 -*-
"""
@File    :   diException.py    
@Author  :   gfyulx@163.com
@Version :    1.0
@Description:
@Modify TIme:  2020/5/28 16:24
Copyright:  Fujian Linewell Software Co., Ltd. All rights reserved.
注意：本内容仅限于福建南威软件股份有限公司内部传阅，禁止外泄以及用于其他的商业目的
"""

class DiException(Exception):

    def __init__(self, message):
        super().__init__(self)
        self.message = message

    def __str__(self):
        return self.message