# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import datetime

'''
来自敬忠良，肖刚，李振华《图像融合——理论与分析》P85：基于像素清晰度的融合规则
1，用Laplace金字塔或者是小波变换，将图像分解成高频部分和低频部分两个图像矩阵
2，以某个像素点为中心开窗，该像素点的清晰度定义为窗口所有点((高频/低频)**2).sum()
3，目前感觉主要的问题在于低频
4，高频取清晰度图像中较大的那个图的高频图像像素点
5，算法优化后速度由原来的2min.44s.变成9s.305ms.
补充：书上建议开窗大小10*10，DWT取3层，Laplace金字塔取2层
'''

def imgOpen(img_src1,img_src2):
    apple=Image.open(img_src1).convert('L')
    orange=Image.open(img_src2).convert('L')
    appleArray=np.array(apple)
    orangeArray=np.array(orange)
    return appleArray,orangeArray

# 严格的变换尺寸
def _sameSize(img_std,img_cvt):
    x,y=img_std.shape
    pic_cvt=Image.fromarray(img_cvt)
    pic_cvt.resize((x,y))
    return np.array(pic_cvt)

# 求Laplace金字塔
def getLaplacePyr(img):
    firstLevel=img.copy()
    secondLevel=cv2.pyrDown(firstLevel)
    lowFreq=cv2.pyrUp(secondLevel)
    highFreq=cv2.subtract(firstLevel,_sameSize(firstLevel,lowFreq))
    return lowFreq,highFreq

# 计算对比度，优化后不需要这个函数了，扔在这里看看公式就行了
def _getContrastValue(highWin,lowWin):
    row,col = highWin.shape
    contrastValue = 0.00
    for i in xrange(row):
        for j in xrange(col):
            contrastValue += (float(highWin[i,j])/lowWin[i,j])**2
    return contrastValue

# 先求出每个点的(hi/lo)**2，再用numpy的sum（C语言库）求和
def getContrastImg(low,high):
    row,col=low.shape
    if low.shape!=high.shape:
        low=_sameSize(high,low)
    contrastVal=np.zeros((row,col))
    contrastImg=np.zeros((row,col))
    for i in xrange(row):
        for j in xrange(col):
            contrastVal[i,j]=(float(high[i,j])/low[i,j])**2
    for i in xrange(row):
        for j in xrange(col):
            up=i-5 if i-5>0 else 0
            down=i+5 if i+5<row else row
            left=j-5 if j-5>0 else 0
            right=j+5 if j+5<col else col
            contrastWindow=contrastVal[up:down,left:right]
            contrastImg[i,j]=contrastWindow.sum()
    return contrastImg

# 计算方差权重比
def getVarianceWeight(apple,orange):
    appleMean,appleVar=cv2.meanStdDev(apple)
    orangeMean,orangeVar=cv2.meanStdDev(orange)
    appleWeight=float(appleVar)/(appleVar+orangeVar)
    orangeWeight=float(orangeVar)/(appleVar+orangeVar)
    return appleWeight,orangeWeight

# 函数返回融合后的图像矩阵
def getFusion(apple,orange):
    beginTime=datetime.datetime.now()
    print beginTime
    lowApple,highApple = getLaplacePyr(apple)
    lowOrange,highOrange = getLaplacePyr(orange)
    contrastApple = getContrastImg(lowApple,highApple)
    contrastOrange = getContrastImg(lowOrange,highOrange)
    row,col = lowApple.shape
    highFusion = np.zeros((row,col))
    lowFusion = np.zeros((row,col))
    # 开始处理低频
    # appleWeight,orangeWeight=getVarianceWeight(lowApple,lowOrange)
    for i in xrange(row):
        for j in xrange(col):
            # lowFusion[i,j]=lowApple[i,j]*appleWeight+lowOrange[i,j]*orangeWeight
            lowFusion[i,j] = lowApple[i,j] if lowApple[i,j]<lowOrange[i,j] else lowOrange[i,j]
    # 开始处理高频
    for i in xrange(row):
        for j in xrange(col):
            highFusion[i,j] = highApple[i,j] if contrastApple[i,j] > contrastOrange[i,j] else highOrange[i,j]
    # 开始重建
    fusionResult = cv2.add(highFusion,lowFusion)
    endTime=datetime.datetime.now()
    print endTime
    print 'Runtime: '+str(endTime-beginTime)
    return fusionResult

# 绘图函数
def getPlot(apple,orange,result):
    plt.subplot(131)
    plt.imshow(apple,cmap='gray')
    plt.title('src1')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(orange,cmap='gray')
    plt.title('src2')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(result,cmap='gray')
    plt.title('result')
    plt.axis('off')
    plt.show()


if __name__=='__main__':
    src1='F:\\Python\\try\\BasicImageOperation\\pepsia.jpg'
    src2='F:\\Python\\try\\BasicImageOperation\\pepsib.jpg'
    apple,orange=imgOpen(src1,src2)
    result=getFusion(apple,orange)
    getPlot(apple,orange,result)
