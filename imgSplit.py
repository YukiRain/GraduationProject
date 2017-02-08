import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os,sys

#1，将一张含有多张图片的图拆成多张图
#2，将同一组图片resize()成相同的大小
#3,按照文件名（数字）分类处理
#4，获取文件夹下所有文件，处理后保存

def imgOpen(path):
    img=Image.open(path).convert('L')
    return np.array(img)

def getYsize(img,miss=150,begin=0):
    x,y=img.shape
    xbegin, xend = 0, 0
    for i in range(begin, x):
        if img[i, miss] != 255:
            xbegin = i
            break
    for k in range(xbegin + 1, x):
        if img[k, miss] == 255:
            if img[k,miss+90]==255 and img[k,miss-90]==255:
                xend = k
                break
    if xend==0:
        xend=x
    return xbegin,xend

def getXsize(img,miss=150,begin=0):
    x, y = img.shape
    ybegin,yend=0,0
    for i in range(begin, y):
        if img[miss, i] != 255:
            ybegin = i
            break
    for k in range(ybegin + 1, y):
        if img[miss, k] == 255:
            if img[miss-90,k]==255 and img[miss+90,k]==255:
                yend = k
                break
    if yend==0:
        yend=y
    return ybegin,yend

def imgSplit(img):
    x,y=img.shape
    ybegin, yend = getYsize(img,150,50)           #修改最后一个参数为分割图片所在的Y轴位置
    print(ybegin), print(yend)
    xbegin,xend=0,0
    splitList=[]
    while len(splitList)<4:
        xbegin, xend = getXsize(img,150,xend)
        if xend-xbegin<150:
            continue
        print(xbegin), print(xend)
        res = np.zeros((yend - ybegin, xend - xbegin))
        dim1, dim2 = res.shape
        for i in range(dim1):
            for j in range(dim2):
                res[i, j] = img[ybegin + i, xbegin + j]
        splitList.append(res)
        xend+=3
    return splitList

def imgSave(imgList):
    for i in range(len(imgList)):
        imgToSave=Image.fromarray(imgList[i])
        imgToSave=imgToSave.convert('RGB')
        imgToSave.save('F:\\GraduationProject\\IMG\\splitTest\\'+str(i+4)+'.jpg')

def imgTest(origin,img):
    plt.subplot(1,len(img)+1,1)
    plt.imshow(origin,cmap='gray')
    for i in range(len(img)):
        plt.subplot(1,len(img)+1,i+2)
        plt.imshow(img[i],cmap='gray')
    plt.show()

if __name__=='__main__':
    imgArray=imgOpen('F:\\GraduationProject\\IMG\\pdfimg\\2.png')
    res=imgSplit(imgArray)
    #imgSave(res)                          #若不保存注释掉这行
    imgTest(imgArray,res)