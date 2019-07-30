#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 7/26/2019 10:12 AM 
# @Author : gfyulx 
# @File : WordCut.py 
# @description:


import jieba
from jieba import posseg as pg
import pandas as pd
from WordSimilarCompute import WordSimilarCompute

stopwords = []
with open("stopword.txt", "r") as f:
    [stopwords.append(x.strip()) for x in f.readlines()]
print(stopwords)


def cutTest():
    df = pd.read_csv("word.csv", header=0)
    oriList = df.values.tolist()
    [x.append(x[0]) for x in oriList]
    for item in oriList:
        words = pg.cut(item[0])
        description = ''
        description = description.join([x.word for x in words if x.word not in stopwords and x.flag in ['v', 'n', 'nr', 'ns', 'nt', 'nz', 'vn']])
        if description != '':
            item[0]=description

    print(oriList)
    sim = WordSimilarCompute()
    wordInput = ''
    inword = '涉案物品价值(人民币万元)'
    words = pg.cut(inword)
    wordInput = wordInput.join(
        [x.word for x in words if x.word not in stopwords and x.flag in ['v', 'n', 'nr', 'ns', 'nt', 'nz', 'vn']])
    if wordInput == '':
        wordInput = inword
    print(wordInput)
    res = sim.editDistance(wordInput, oriList[:])
    print(res)
    resMap = {}
    for x in res[:5]:
        #result.append([x[0], df["中文名称"][i], df["对象类词"][i], df["特性词"][i], df['表示词'][i]])
        if resMap.get(x[1], 0):
            resMap[x[1]] += 1
        else:
            resMap[x[1]] = 1
        print(x)
    a = sorted(resMap.items(), key=lambda x: x[1], reverse=True)
    print(a[0][0])
    print(getcharacter(inword))
    c = getexpress(inword)
    if c == inword:
        expressWord = [x.word for x in pg.cut(inword) if x.flag in ['v', 'n', 'nr', 'ns', 'nt', 'nz',
                                                                                   'vn']][-1]
        print("express",expressWord)
        for item in oriList:
            words = pg.cut(item[4])
            tmpList=[x.word for x in words if
                                            x.word not in stopwords and x.flag in ['v', 'n', 'nr', 'ns', 'nt', 'nz',
                                                                                   'vn']]
            if tmpList==[]:
                item[0]=item[4]
            else:
                item[0] = ''.join(tmpList[-1])
        print(oriList)
        res = sim.editDistance(expressWord, oriList[:])
        resexpressMap = {}
        for x in res[:5]:
            print(x)
            # result.append([x[0], df["中文名称"][i], df["对象类词"][i], df["特性词"][i], df['表示词'][i]])
            if resexpressMap.get(x[3], 0):
                resexpressMap[x[3]] += 1
            else:
                resexpressMap[x[3]] = 1
            if x[-1] == 0.0:
                break
        b = sorted(resexpressMap.items(), key=lambda x: x[1], reverse=True)

        print(b[0][0])
    else:
        print(c)

    # 返回一个语句的特征词
    """
    规则：1、分词后有多个n或者n开头的词，取最后一个n
         1处理后为n 加b 的，去除n，只保留b
         每个v之后的n保留，此规则不限制规则1
    """


def getcharacter(inputWords):
    res = []
    words = pg.cut(inputWords)
    nList = []
    flagList = []
    for x in words:
        print(x.word, x.flag)
        if x.word in stopwords:
            continue
        if x.flag in ['r','uj','un', 'l', 'n', 'nr', 'ns', 'nt', 'nz', 'vn']:
            nList.append(x.word)
        res.append(x.word)
        flagList.append(x.flag)

    for x in range(len(flagList)):
        if flagList[x] == 'c':  # 连接词前面保留
            if x - 1 >= 0 and flagList[x - 1] in ['un', 'n', 'nr', 'ns', 'nt', 'nz', 'vn']:
                if res[x - 1] in nList:
                    nList.remove(res[x - 1])
            if x + 1 < len(flagList) and flagList[x + 1] in ['un', 'n', 'nr', 'ns', 'nt', 'nz', 'vn']:
                if res[x + 1] in nList:
                    nList.remove(res[x + 1])
        if flagList[x] == 'p' or flagList[x] == 'x':  # 介词后连名词
            if x - 1 >= 0 and flagList[x - 1] in ['un', 'n', 'nr', 'ns', 'nt', 'nz', 'vn']:
                if res[x - 1] in nList:
                    nList.remove(res[x - 1])
            if x + 1 < len(flagList) and flagList[x + 1] in ['un', 'n', 'nr', 'ns', 'nt', 'nz', 'vn']:
                if res[x + 1] in nList:
                    nList.remove(res[x + 1])
        # 动词后保留有多个名词的，移除这个动词
        if flagList[x] == 'v' or flagList[x] == 'vn':
            a = 0
            a = a + len([1 for x in flagList[x + 1:] if x in ['un', 'n', 'nr', 'ns', 'nt', 'nz', 'vn']])
            if a > 1:
                res[x]=''
    if nList != []:
        [res.remove(x) for x in nList[:-1] if x in res]
    return ''.join(res)


def getexpress(inputWords):
    words = pg.cut(inputWords)
    for x in words:
        if x.word in stopwords:
            return x.word
    #计算切词后的相似度

    return inputWords


if __name__ == '__main__':
    cutTest()
