# -*- encoding: utf-8 -*-
"""
@File    :   servicevgg.py    
@Author  :   gfyulx@163.com
@Version :    1.0
@Description:
@Modify TIme:  2020/6/1 15:44
Copyright:  Fujian Linewell Software Co., Ltd. All rights reserved.
注意：本内容仅限于福建南威软件股份有限公司内部传阅，禁止外泄以及用于其他的商业目的
"""

from core.comservice.deeplearning.modelservice import ModelService
from .NN import VGG16



class ServiceVGG(ModelService):


    def __init__(self,**args):
        super(ServiceVGG,self).__init__(args)



