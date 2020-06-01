# -*- encoding: utf-8 -*-
"""
@File    :   VGG16.py    
@Author  :   gfyulx@163.com
@Version :    1.0
@Description:
@Modify TIme:  2020/6/1 15:42
Copyright:  Fujian Linewell Software Co., Ltd. All rights reserved.
注意：本内容仅限于福建南威软件股份有限公司内部传阅，禁止外泄以及用于其他的商业目的
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Dropout


class VGG16():

    def __init__(self,shape=(224,224,3)):
        self.shape=shape

    def model(self):
        input_1 = Input(shape=self.shape)  # 输入224*224*3
        # 第一部分
        # 卷积 64深度，大小是3*3 步长为1 使用零填充 激活函数relu
        # 2次卷积 一次池化 池化尺寸2*2 步长2*2
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(input_1)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)  # 64 224*224
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)  # 64 112*112
        # 第二部分 2次卷积 一次池化
        # 卷积 128深度 大小是3*3 步长1 零填充
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)  # 128 112*112
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)  # 128 56*56
        # 第三部分 3次卷积 一次池化 卷积256 3*3
        x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)
        x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)
        x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)  # 256 56*56
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)  # 256 28*28
        # 第四部分 3次卷积 一次池化 卷积 512 3*3
        x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)
        x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)
        x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)  # 512 28*28
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)  # 512 14*14
        # 第五部分 3次卷积 一次池化 卷积 512 3*3
        x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)
        x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)
        x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)  # 512 14*14
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)  # 512 7*7

        x = Flatten()(x)  # 扁平化，用在全连接过渡
        # 第六部分 三个全连接
        # 第一个和第二个全连接相同 输出4096 激活relu 使用dropout，随机丢弃一半
        x = Dense(4096, activation="relu")(x)
        Dropout(0.5)(x)
        x = Dense(4096, activation="relu")(x)
        Dropout(0.5)(x)  # 输出 4096 1*1
        # 第三个全连接层 输出 softmax分类
        out_ = Dense(self.num_classes, activation="softmax")(x)
        model = Model(inputs=input_1, outputs=out_)
        # print(model.summary())
        sgd = optimizers.sgd(lr=0.001, momentum=0.9, nesterov=True)
        model.compile(sgd, loss="categorical_crossentropy", metrics=["accuracy"])
        # plot_model(model,"model.png")
        return model