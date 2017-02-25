# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import time

'''
来自敬忠良，肖刚，李振华《图像融合——理论与分析》P85：基于像素清晰度的融合规则
1，用Laplace金字塔或者是小波变换，将图像分解成高频部分和低频部分两个图像矩阵
2，以某个像素点为中心开窗，该像素点的清晰度定义为窗口所有点((高频/低频)**2).sum()
3，书上说低频也和高频用一样的策略，但是鉴于效果很好的小波变换里低频使用的是方差权重比策略，这里低频部分也沿用方差权重比
4，高频取清晰度图像中较大的那个图的高频图像像素点
补充：书上建议开窗大小10*10，DWT取3层，Laplace金字塔取2层
'''

class contrastFusion(object):
    def __init__(self,img_src1,img_src2):
        apple=Image.open(img_src1).convert('L')
        orange=Image.open(img_src2).convert('L')
        self.apple=np.array(apple)
        self.orange=np.array(orange)

    def _sameSize(self,img_std,img_cvt):
        x,y=img_std.shape
        pic_cvt=Image.fromarray(img_cvt)
        pic_cvt.resize((x,y))
        return np.array(pic_cvt)

    def getLaplacePyr(self,img):
        firstLevel=img.copy()
        secondLevel=cv2.pyrDown(firstLevel)
        lowFreq=cv2.pyrUp(secondLevel)
        highFreq=cv2.subtract(firstLevel,self._sameSize(firstLevel,lowFreq))
        return lowFreq,highFreq

    def _getContrastValue(self,highWin,lowWin):
        row,col=highWin.shape
        contrastValue=0.00
        for i in xrange(row):
            for j in xrange(col):
                contrastValue+=(float(highWin[i,j])/lowWin[i,j])**2
        return contrastValue

    def getContrastImg(self,low,high):
        row,col=low.shape
        if low.shape!=high.shape:
            low=self._sameSize(high,low)
        contrastImg=np.zeros((row,col))
        for i in xrange(row):
            for j in xrange(col):
                up=i-3 if i-3>0 else 0
                down=i+3 if i+3<row else row
                left=j-3 if j-3>0 else 0
                right=j+3 if j+3<col else col
                lowWin=low[up:down,left:right]
                highWin=high[up:down,left:right]
                contrastImg[i,j]=self._getContrastValue(highWin,lowWin)
                # print contrastImg[i,j]
        return contrastImg

    def _getVarianceWeight(self,apple,orange):
        appleMean,appleVar=cv2.meanStdDev(apple)
        orangeMean,orangeVar=cv2.meanStdDev(orange)
        appleWeight=float(appleVar)/(appleVar+orangeVar)
        orangeWeight=float(orangeVar)/(appleVar+orangeVar)
        return appleWeight,orangeWeight

    def getFusion(self):
        lowApple,highApple=self.getLaplacePyr(self.apple)
        lowOrange,highOrange=self.getLaplacePyr(self.orange)
        contrastApple=self.getContrastImg(lowApple,highApple)
        contrastOrange=self.getContrastImg(lowOrange,highOrange)
        row,col=lowApple.shape
        highFusion=np.zeros((row,col))
        lowFusion=np.zeros((row,col))
        # 开始处理低频
        appleWeight,orangeWeight=self._getVarianceWeight(lowApple,lowOrange)
        for i in xrange(row):
            for j in xrange(col):
                lowFusion[i,j]=appleWeight*lowApple[i,j]+orangeWeight*lowOrange[i,j]
        # 开始处理高频
        for i in xrange(row):
            for j in xrange(col):
                highFusion[i,j]=highApple[i,j] if contrastApple[i,j]>contrastOrange[i,j] else highOrange[i,j]
        # 开始重建
        fusionResult=cv2.add(highFusion,lowFusion)
        return fusionResult


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
    fusion=contrastFusion(src1,src2)
    result=fusion.getFusion()
    getPlot(fusion.apple,fusion.orange,result)
    time.sleep(10)
    raw_input('unknown bug inside box')