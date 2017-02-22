# -*- coding: utf-8 -*-
import pywt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

'''
文献中的方案永远是代码实现起来最麻烦的方案：
1，对低频分量的所有像素点计算其局部方差，每张图所有点的方差加起来除以两张图所有点加起来，得到两张图的权重
融合图像每个像素点的值为两张图对应像素点的值加权平均，这个权就是上面算出来的权值。
2，对每个高频分量的像素点用canny算子进行边缘提取，再对边缘图像的每一个像素点计算其局部方差，得到方差图像
融合图像每个像素点的值为方差图片中对应像素点取值较大的那张图片的像素点。
3,然而我真的是不会写这个方案，该版本暂时先全部用方差代替吧
4,实验测得效果不是很好，but why？
'''

def imgOpen(path):
    img=Image.open(path).convert('L')
    imgArray=np.array(img)
    return imgArray

# 对于低频分量，计算两图的权重比
def varianceWeight(img1,img2):
    mean1,var1=cv2.meanStdDev(img1)
    mean2,var2=cv2.meanStdDev(img2)
    weight1=var1/(var1+var2)
    weight2=var2/(var1+var2)
    return weight1,weight2

# 不会写canny，暂时先用Sobel算子代替
def calcGradient(img):
    xDiff=cv2.Sobel(img,cv2.CV_16S,1,0)
    yDiff=cv2.Sobel(img,cv2.CV_16S,0,1)
    stdXdiff=cv2.convertScaleAbs(xDiff)
    stdYdiff=cv2.convertScaleAbs(yDiff)
    gradient=np.sqrt(stdXdiff**2+stdYdiff**2)
    return gradient

def testWave(img1,img2):
    transf1=pywt.wavedec2(img1,'haar',level=4)
    transf2=pywt.wavedec2(img2,'haar',level=4)
    assert len(transf1)==len(transf2)
    recWave=[]
    for k in range(len(transf1)):
        # 处理低频分量
        if k==0:
            loWeight1,loWeight2 = varianceWeight(transf1[0],transf2[0])
            lowFreq = np.zeros(transf2[0].shape)
            row,col = transf1[0].shape
            for i in range(row):
                for j in range(col):
                    lowFreq[i,j] = loWeight1*transf1[0][i,j] + loWeight2*transf2[0][i,j]
            recWave.append(lowFreq)
            continue
        # 处理高频分量
        cvtArray=[]
        for array1,array2 in zip(transf1[k],transf2[k]):
            tmp_row,tmp_col = array1.shape
            highFreq = np.zeros((tmp_row,tmp_col))
            hiWeight1,hiWeight2=varianceWeight(array1,array2)
            grad1=calcGradient(array1);grad2=calcGradient(array2)
            for i in range(tmp_row):
                for j in range(tmp_col):
                    # highFreq[i,j] = array1[i,j]*hiWeight1 + array2[i,j]*hiWeight2
                    highFreq[i,j]=array1[i,j] if grad1[i,j]>grad2[i,j] else array2[i,j]
            cvtArray.append(highFreq)
        recWave.append(tuple(cvtArray))
    return pywt.waverec2(recWave,'haar')

def testPlot(org1,org2,img):
    plt.subplot(131)
    plt.imshow(org1,cmap='gray')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(org2,cmap='gray')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(img,cmap='gray')
    plt.axis('off')
    plt.show()

if __name__=='__main__':
    img1=imgOpen('F:\\Python\\try\\BasicImageOperation\\pepsia.jpg')
    img2=imgOpen('F:\\Python\\try\\BasicImageOperation\\pepsib.jpg')
    rec=testWave(img1,img2)
    testPlot(img1,img2,rec)