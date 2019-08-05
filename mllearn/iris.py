#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 8/1/2019 9:29 AM 
# @Author : gfyulx 
# @File : iris.py.py 
# @description: tensorflow实现的鸢尾花分类

import pandas as pd
import tensorflow as tf


CSV_COLUMN_NAMES = ['SepalLength','SepalWidth','PetalLength', 'PetalWidth', 'Species']
data_train = pd.read_csv("../../data/iris_training.csv",names= CSV_COLUMN_NAMES,header=0)
data_test = pd.read_csv("../../data/iris_test.csv",names=CSV_COLUMN_NAMES, header=0)

train_x,train_y=data_train,data_train.pop("Species")
test_x,test_y=data_test,data_test.pop("Species")

train_x.head()
feature_columns=[]
for key in train_x.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))
print(feature_columns)
classifier=tf.estimator.DNNClassifier(feature_columns=feature_columns,hidden_units=[5,5,5],n_classes=3)


def generator_data(x,y):
    dataset=tf.data.Dataset.from_tensor_slices((dict(x),y))
    dataset=dataset.shuffle(1000).repeat().batch(100)
    return dataset

classifier.train(input_fn=lambda :generator_data(train_x,train_y),steps=1000)

def eval(features,labels,batch_size):
    features=dict(features)
    if labels is None:
        inputs=features
    else:
        inputs=(features,labels)
    dataset=tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None,"batch_size must not be None"
    dataset=dataset.batch(batch_size)
    return dataset

predict_arr=[]
predictions=classifier.predict(input_fn=lambda :eval(test_x,labels=test_y,batch_size=100))
for predict in predictions:
    predict_arr.append(predict['probabilities'].argmax())
result=predict_arr==test_y
result1=[w for w in result if w==True]
print("准确率为%s" %  str((len(result1)/len(result))))


