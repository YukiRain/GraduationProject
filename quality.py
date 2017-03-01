# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from math import log

# 为降低程序耦合，把计算熵之前的统计过程封装起来
def getLabels(img,addLabel=None):
    labels={} if addLabel==None else addLabel
    row,col=img.shape
    for i in xrange(row):
        for j in xrange(col):
            # 限定len(labels)最大只能是26，否则算起来太零碎了
            greyLabel=int(img[i,j]/10)
            if greyLabel not in labels.keys():
                labels[greyLabel]=0
            labels[greyLabel]+=1
    return labels

# 计算熵
def entropy(img):
    labels=getLabels(img)
    row,col=img.shape
    ent=0.00
    entrySize=row*col
    for key in labels:
        prob=float(labels[key])/entrySize
        ent-=prob*log(prob,2)
    return ent

# 计算两张图熵的差值
def entropyDiff(apple,orange):
    appleEnt=entropy(apple)
    orangeEnt=entropy(orange)
    return appleEnt-orangeEnt

# 计算交叉熵(以第一个变量为基准)
def crossEntropy(apple,orange):
    assert apple.shape==orange.shape
    appleLabels=getLabels(apple)
    orangeLabels=getLabels(orange)
    entrySize=apple.shape[0]*apple.shape[1]
    XEnt = 0.00
    infoSet=appleLabels.keys()+orangeLabels.keys()
    for key in infoSet:
        if key in appleLabels.keys():
            prob = float(appleLabels[key])/entrySize
        if key in orangeLabels.keys():
            q_prob=float(orangeLabels[key])/entrySize
        else:
            continue
        XEnt-=prob*log(prob/q_prob,2)
    return XEnt

# 互信息（看不懂啥是归一化）
def interactiveInfo(apple,orange):
    appleLabel=getLabels(apple)
    orangeLabel=getLabels(orange)
    unionLabel=getLabels(orange,addLabel=appleLabel)
    pass