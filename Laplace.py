# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import datetime

'''
# #########
# 文献中提到了两种融合策略：
# 1.(initial)
# 对两个图像的拉普拉斯金字塔从最底层开始，对每个像素点开窗求窗口的平均梯度/方差
# 新生成的重构金字塔每个对应像素点取值为平均梯度/方差较大的那个像素点
# 2.
# 最高层采用0.5-0.5加权平均融合，下面每层用极大值融合
# 3.(Feb23)
# 我将在小波变换中效果非常好的局部方差图像算法放到Laplace金字塔中，效果依旧不好
# Laplace金字塔的重构图像在图像的灰度会畸变，也可能是我选的图像不合适，可能这种算法不适用于可见光波段图像
# #########
'''

# 使用opencv中的Sobel算子的梯度计算函数
def cvGradient(img):
    xDiff=cv2.Sobel(img,cv2.CV_16S,1,0)
    yDiff=cv2.Sobel(img,cv2.CV_16S,0,1)
    stdXdiff=cv2.convertScaleAbs(xDiff)
    stdYdiff=cv2.convertScaleAbs(yDiff)
    gradient=np.sqrt(stdXdiff**2+stdYdiff**2)
    return np.mean(gradient)

# 计算量太大
def getGradientImg(array):
    row,col=array.shape
    varImg=np.zeros((row,col))
    for i in xrange(row):
        for j in xrange(col):
            up=i-5 if i-5>0 else 0
            down=i+5 if i+5<row else row
            left=j-5 if j-5>0 else 0
            right=j+5 if j+5<col else col
            window=array[up:down,left:right]
            varImg[i,j]=cvGradient(window)
    return varImg

# 计算方差图像
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

# 使得img1的大小与img2相同
def sameSize(img1, img2):
    try:
        rows, cols, dpt = img2.shape
    except:
        rows,cols=img2.shape
    dst = img1[:rows,:cols]
    return dst

# 生成图像金字塔, dep代表金字塔层数
def getGaussPyr(img,dep):
    G = img.copy()
    gp_img = [G]
    for i in xrange(dep):
        # cv2.pyrDown: 用高斯核卷积，去掉偶数行列，实现降采样
        G = cv2.pyrDown(G)
        gp_img.append(G)
    return gp_img

# 求Laplace金字塔
def getLaplacePyr(GaussPyrImg):
    dep=len(GaussPyrImg)
    LaplacePyr=[GaussPyrImg[dep-1]]
    for i in xrange(dep-1,0,-1):
        # cv2.pyrUp: 行列变为原来的2倍，再用高斯核卷积
        GE=cv2.pyrUp(GaussPyrImg[i])
        # 求矩阵差，相当于带通滤波过程
        L=cv2.subtract(GaussPyrImg[i-1],sameSize(GE,GaussPyrImg[i-1]))
        LaplacePyr.append(L)
    return LaplacePyr

# 实现策略1
def reconstruct1(lp1,lp2):
    dep=len(lp1)
    assert dep==len(lp2)
    LS=[]
    for la,lb in zip(lp1,lp2):
        try:
            row,col,dpt=la.shape
        except:
            row,col=la.shape
        tmp=np.zeros((row,col))
        la_gradient=getVarianceImg(la)
        lb_gradient=getVarianceImg(lb)
        for i in range(row):
            for j in range(col):
                try:
                    tmp[i,j] = la[i,j] if la_gradient[i,j] > lb_gradient[i,j] else lb[i,j]
                except:
                    tmp[i,j] = la[i,j][0] if la_gradient[i,j][0] > lb_gradient[i,j][0] else lb[i,j][0]
        LS.append(tmp)
    ls_reconstruct=LS[0]
    for i in xrange(1,dep-1):
        ls_reconstruct=cv2.pyrUp(ls_reconstruct)
        ls_reconstruct=cv2.add(sameSize(ls_reconstruct,LS[i]),LS[i])
    return ls_reconstruct

# 实现策略2
def reconstruct2(lp1,lp2):
    dep=len(lp1)
    assert dep==len(lp2)
    LS=[]
    ta,tb=lp1[0],lp2[0]
    try:
        rowFirst,colFirst,dptFirst=ta.shape
    except:
        rowFirst,colFirst = ta.shape
    tmpFirst=np.zeros((rowFirst,colFirst))
    for i in xrange(rowFirst):
        for j in xrange(colFirst):
            try:
                tmpFirst[i,j] = ta[i,j][0]/2 + tb[i,j][0]/2
            except:
                tmpFirst[i,j] = ta[i,j]/2 + tb[i,j]/2
    LS.append(tmpFirst)
    for k in range(dep):
        if k==0:
            continue
        la=lp1[k]
        lb=lp2[k]
        try:
            row,col,dpt = la.shape
        except:
            row,col=la.shape
        tmp = np.zeros((row,col))
        for i in xrange(row):
            for j in xrange(col):
                try:
                    tmp[i,j] = la[i,j][0] if la[i,j][0] > lb[i,j][0] else lb[i,j][0]
                except:
                    tmp[i,j] = la[i,j] if la[i,j] > lb[i,j] else lb[i,j]
        LS.append(tmp)
    ls_reconstruct=LS[0]
    for i in xrange(1,dep-1):
        ls_reconstruct=cv2.pyrUp(ls_reconstruct)
        ls_reconstruct=cv2.add(sameSize(ls_reconstruct,LS[i]),LS[i])
    return ls_reconstruct

def testPlot(org1,org2,result):
    plt.subplot(131),plt.imshow(org1,cmap='gray')
    plt.title("apple"),plt.xticks([]),plt.yticks([])
    plt.subplot(132),plt.imshow(org2,cmap='gray')
    plt.title("orange"),plt.xticks([]),plt.yticks([])
    plt.subplot(133),plt.imshow(result,cmap='gray')
    plt.title("laplace_pyramid"),plt.xticks([]),plt.yticks([])
    plt.show()

def testFusion():
    apple = Image.open("F:\\Python\\try\\BasicImageOperation\\pepsia.jpg")
    orange = Image.open("F:\\Python\\try\\BasicImageOperation\\pepsib.jpg")
    apple=np.array(apple)
    orange=np.array(orange)
    beginTime=datetime.datetime.now()
    print beginTime
    gp_apple=getGaussPyr(apple,6)
    gp_orange=getGaussPyr(orange,6)
    lp_apple=getLaplacePyr(gp_apple)
    lp_orange=getLaplacePyr(gp_orange)
    result=reconstruct1(lp_apple,lp_orange)
    endTime=datetime.datetime.now()
    print endTime
    print 'Runtime: '+str(endTime-beginTime)
    testPlot(apple,orange,result)


if __name__=='__main__':
    testFusion()