# -*- coding: utf-8 -*-
import pywt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2,os
from datetime import datetime

'''
论文中的两种方案：
1，对低频分量的所有像素点计算其局部方差，每张图所有点的方差加起来除以两张图所有点加起来，得到两张图的权重
融合图像每个像素点的值为两张图对应像素点的值加权平均，这个权就是上面算出来的权值。
2，对每个高频分量的像素点用canny算子进行边缘提取，再对边缘图像的每一个像素点计算其局部方差，得到方差图像
融合图像每个像素点的值为方差图片中对应像素点取值较大的那张图片的像素点。
3，效果不是很好，在小波分块的边缘有明显的灰度跳变(就是有些论文里说的分块效应)，but why?
4，已找到原因，要求的不是某一个点对全图的方差，而是在某点附近开个小窗口求窗口的局部方差
5，现用局部方差的方法对多聚焦图像效果非常完美
'''

img_dir='F:\\GraduationProject\\IMG\\splitTest\\'
save_dir='F:\\GraduationProject\\IMG\\result\\'

def imgOpen(path):
    img=Image.open(path).convert('L').resize((512,512))
    imgArray=np.array(img)
    return imgArray

# 严格的变换尺寸
def _sameSize(img_std,img_cvt):
    x,y=img_std.shape
    return img_std,img_cvt[:x,:y]

# 对于低频分量，计算两图的权重比
def varianceWeight(img1,img2):
    mean1,var1=cv2.meanStdDev(img1)
    mean2,var2=cv2.meanStdDev(img2)
    weight1=var1/(var1+var2)
    weight2=var2/(var1+var2)
    return weight1,weight2

# 实测这个函数效果非常好！！！
def getVarianceImg(array):
    row,col=array.shape
    varImg=np.zeros((row,col))
    for i in xrange(row):
        for j in xrange(col):
            up=i-5 if i-5>0 else 0
            down=i+5 if i+5<row else row
            left=j-5 if j-5>0 else 0
            right=j+5 if j+5<col else col
            window=array[up:down,left:right]
            mean,var=cv2.meanStdDev(window)
            varImg[i,j]=var
    return varImg

# 不会写canny，暂时先用Sobel算子代替
def calcGradient(img):
    xDiff=cv2.Sobel(img,cv2.CV_16S,1,0)
    yDiff=cv2.Sobel(img,cv2.CV_16S,0,1)
    stdXdiff=cv2.convertScaleAbs(xDiff)
    stdYdiff=cv2.convertScaleAbs(yDiff)
    gradient=np.sqrt(stdXdiff**2+stdYdiff**2)
    return gradient

def testWave(img1,img2):
    transf1=pywt.wavedec2(img1,'haar',level=6)
    transf2=pywt.wavedec2(img2,'haar',level=6)
    assert len(transf1)==len(transf2)
    recWave=[]
    for k in range(len(transf1)):
        # 处理低频分量
        if k==0:
            loWeight1,loWeight2 = varianceWeight(transf1[0],transf2[0])
            lowFreq=loWeight1*transf1[0]+loWeight2*transf2[0]
            recWave.append(lowFreq)
            continue
        # 处理高频分量
        cvtArray=[]
        for array1,array2 in zip(transf1[k],transf2[k]):
            tmp_row,tmp_col = array1.shape
            highFreq = np.zeros((tmp_row,tmp_col))
            var1=getVarianceImg(array1);var2=getVarianceImg(array2)
            for i in range(tmp_row):
                for j in range(tmp_col):
                    highFreq[i,j]=array1[i,j] if var1[i,j]>var2[i,j] else array2[i,j]
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

def runTest(src1=None,src2=None,isPlot=True):
    if src1==None or src2==None:
        img1=imgOpen('F:\\Python\\try\\BasicImageOperation\\pepsia.jpg')
        img2=imgOpen('F:\\Python\\try\\BasicImageOperation\\pepsib.jpg')
    else:
        img1=src1;img2=src2
    beginTime=datetime.now()
    print(beginTime)
    rec=testWave(img1,img2)
    endTime=datetime.now()
    print(endTime)
    print(str(endTime-beginTime))
    if isPlot:
        testPlot(img1,img2,rec)
    return rec


if __name__=='__main__':
    imgList=os.listdir(img_dir)
    imgList.sort()
    for img_index in range(21):
        src_list = filter(lambda x:int(x.split('_')[0]) == img_index, imgList)
        apple=imgOpen(img_dir+src_list[0])
        orange=imgOpen(img_dir+src_list[1])
        wave=runTest(src1=apple,src2=orange,isPlot=False)
        wave *= 255.0/(float(wave.max()))
        Image.fromarray(wave).convert('RGB').save(save_dir+str(img_index)+'_wave_var.jpg')
        print 'SAVING NO %d RESULT' % (img_index)
    print 'FINISHED.'