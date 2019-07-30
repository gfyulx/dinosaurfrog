#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @Time : 7/25/2019 9:47 AM 
# @Author : gfyulx 
# @File : WordSimilarCompute.py 
# @description:


import pandas as pd
import hashlib
import pypinyin as py
import numpy as np


class WordSimilarCompute:
    scoreList = []
    chinesefourcorner = {}

    def __init__(self):
        with open("chinesefourcorner.txt", "r") as f:
            for line in f.readlines():
                key, value = line.split(",")
                value = value.strip()
                self.chinesefourcorner[key] = value

    def loadFile(self, path):
        df = pd.read_csv(path, header=0)
        return df

    # 计算输入字符与输入的字符数组的编辑距离
    def editDistance(self, inputword, inputList):
        mixList = []
        for word in inputList:
            mixList.append(self.preProcess(word[0]))
            #mixList.append([word, word])
        #print(mixList)
        wordMix = self.preProcess(inputword)
        distanceList = self.distanceCompute(wordMix, mixList)
        [inputList[x].append(distanceList[x]) for x in range(len(inputList))]

        inputList.sort(key=lambda x: x[-1])
        # 前5中，最优距离值》9,说明匹配度不高
        # 对计算后的距离加上长度匹配值，其值为如果长度一致，减去值为长度*N,不一致则加上长度差值*N，N设为3
        print(inputList[0][-1],len(inputword))
        if inputList[0][-1] >= len(inputword)-1:
            print(inputList[0][-1])
            wordLen = len(inputword)
            for x in inputList:
                if wordLen==len(x[0]):
                    x[-1]-=wordLen*2.0
                else:
                    x[-1]+=abs(wordLen-len(x[0]))*2.0
            ##加上最长子串匹配值
                subseq=self.find_lcseque(inputword,x[0])
                x[-1]-=len(subseq)*15
        inputList.sort(key=lambda x: x[-1])
        return inputList

    def penaltyfactor(self,inputword,resList):
        pass
    # 转拼音后+四角编码+五笔编码后转为hash
    # 拼音+声调（数字）+4位四角编码+4位五笔编码
    def preProcess(self, inputList):
        mixStr = ''
        for sigle in inputList:
            fourcorner = self.chinesefourcorner.get(sigle, sigle)
            pinyin = ''.join(py.pinyin(sigle, style=py.Style.TONE3)[0])
            #mixStr+=pinyin
            mixStr += fourcorner + pinyin  # 四角编码为数字，防止与拼音的声调重合，放在前面，两者的权重一样
        return mixStr

    # y计算word2List对word1的距离
    def distanceCompute(self, word1, word2List):
        distanceList = []
        for item in word2List:
            word2 = item
            m = len(word1)
            n = len(word2)
            dp = np.zeros((m + 1, n + 1))
            for i in range(m):
                dp[i][0] = i
            for i in range(n):
                dp[0][i] = i
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    delta = 0 if word1[i - 1] == word2[j - 1] else 1
                    dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
            distance = dp[m][n]
            distanceList.append(distance)
        return distanceList
    #计算n-gram相似度
    def ngram(self):
        pass

    def find_lcseque(self,s1, s2):
        # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
        m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
        # d用来记录转移方向
        d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]

        for p1 in range(len(s1)):
            for p2 in range(len(s2)):
                if s1[p1] == s2[p2]:  # 字符匹配成功，则该位置的值为左上方的值加1
                    m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                    d[p1 + 1][p2 + 1] = 'ok'
                elif m[p1 + 1][p2] > m[p1][p2 + 1]:  # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
                    m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                    d[p1 + 1][p2 + 1] = 'left'
                else:  # 上值大于左值，则该位置的值为上值，并标记方向up
                    m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                    d[p1 + 1][p2 + 1] = 'up'
        (p1, p2) = (len(s1), len(s2))

        s = []
        while m[p1][p2]:  # 不为None时
            c = d[p1][p2]
            if c == 'ok':  # 匹配成功，插入该字符，并向左上角找下一个
                s.append(s1[p1 - 1])
                p1 -= 1
                p2 -= 1
            if c == 'left':  # 根据标记，向左找下一个
                p2 -= 1
            if c == 'up':  # 根据标记，向上找下一个
                p1 -= 1
        s.reverse()
        return ''.join(s)

if __name__ == '__main__':
    sim = WordSimilarCompute()
    f = sim.loadFile("word.csv")
    res = sim.editDistance("信用评定机构", f["中文名称"].tolist())
    #[print(x) for x in res[:5]]
    result=[]
    for x in res[:5]:
        i=-1
        for j in f["中文名称"].tolist():
            i+=1
            if x[0]==j:
                break
        result.append([x[0],f["对象类词"][i],f["特性词"][i],f['表示词'][i]])
    [ print(x) for x in result]



