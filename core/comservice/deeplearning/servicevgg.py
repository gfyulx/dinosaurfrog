# -*- encoding: utf-8 -*-
"""
@File    :   servicevgg.py    
@Author  :   gfyulx@163.com
@Version :    1.0
@Description:
@Modify TIme:  2020/6/1 15:44
"""

from tensorflow.keras.callbacks import TensorBoard
from core.comservice.deeplearning.modelservice import ModelService
from .NN.VGG16 import VGG16
import numpy as np
from core.utils.diException import DiException
from core.utils.common import format_exception, sys_log, LOG_LEVEL


class ServiceVGG(ModelService):
    batch_size=8
    epoch=100
    log_path=None

    def __init__(self,**args):
        super(ServiceVGG,self).__init__(**args)
        self._model=VGG16.model(args.get("num_classes"))
        self._strategy=self.init_session(args)
        self.log_path=args.get("log_path","./")
        self.logs = TensorBoard(log_dir=self.log_path, write_graph=True, write_images=True)

    def train(self,**args):
        try:
            if self._strategy is not None:
                with self._strategy.scope():
                    data_path = args.get("data_path")
                    label_path = args.get("label_path")
                    save_path = args.get("model_path")
                    x = np.load(data_path)
                    y = np.load(label_path)
                    # 乱序列输出
                    np.random.seed(200)
                    np.random.shuffle(x)
                    np.random.seed(200)
                    np.random.shuffle(y)
                    self._model.fit(x, y, batch_size=self.batch_size, epochs=self.epoch, verbose=1, validation_split=0.3,
                              callbacks=[self.logs])
            self._model.save(save_path)
        except Exception as e:
            sys_log(format_exception(e),level=LOG_LEVEL.ERROR)
            raise DiException(format_exception(e))

    def predict(self):
        pass


