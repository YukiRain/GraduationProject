# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def openFiles(files):
    IMG=[]
    img_array=[]
    for f in files:
        IMG.append(Image.open(f).convert('L'))
    for k in IMG:
        img_array.append(np.array(k))
    return img_array

#基于平均值的融合
def averageFusion(imgList):
    x, y = imgList[0].shape
    img=np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            img[i,j]=0
            for k in imgList:
                img[i,j]+=k[i,j]/len(imgList)
    return img

# 极大还是极小都在这个函数里了: 'max' / 'min'
def peakFusion(imgList,peak='max'):
    x,y=imgList[0].shape
    img=np.zeros((x,y))
    if peak=='max':
        for i in range(x):
            for j in range(y):
                for pixel in imgList:
                    img[i,j] = pixel[i,j] if pixel[i,j] > img[i,j] else img[i,j]
    elif peak=='min':
        for i in range(x):
            for j in range(y):
                for pixel in imgList:
                    img[i,j] = pixel[i,j] if pixel[i,j] < img[i,j] else img[i,j]
    return img

#显示函数
def img_show(origin,img,axis):
    for i in range(len(origin)):
        plt.subplot(len(origin)+1,1,i+1)
        plt.imshow(origin[i],cmap='gray')
        if not axis:
            plt.axis('off')
    plt.subplot(len(origin)+1,1,len(origin)+1)
    plt.imshow(img,cmap='gray')
    if not axis:
        plt.axis('off')
    plt.show()

if __name__=='__main__':
    img=['F:\\Python\\try\\BasicImageOperation\\apple.jpg',
         'F:\\Python\\try\\BasicImageOperation\\orange.jpg']
    origin=openFiles(img)
    af=averageFusion(origin)
    img_show(origin,af,True)