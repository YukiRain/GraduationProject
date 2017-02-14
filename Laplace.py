# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

# 用于求梯度的Prewitt算子和Laplace算子
Prewitt_x=np.array([[-1,0,0],
                   [-1,0,1],
                   [-1,0,1]])
Prewitt_y=np.array([[-1,-1,-1],
                    [0,0,0],
                    [1,1,1]])
Laplace=np.array([[0,1,0],
                 [1,-4,1],
                 [0,1,0]])
Laplace_ex=np.array([[1,1,1],
                    [1,-8,1],
                    [1,1,1]])

def imgConv(imgArray,imgOperator):
    '''计算卷积
        parameter:
        imgArray 原灰度图像矩阵
        imgOperator      算子
        返回变换结果的矩阵
    '''
    img=imgArray.copy()
    dim1,dim2,dst=img.shape
    for x in range(1,dim1-1):
        for y in range(1,dim2-1):
            img[x,y]=(imgArray[(x-1):(x+2),(y-1):(y+2)]*imgOperator).sum()

    img=img*(255.0/img.max())
    return img

# 封装一下求梯度的函数
def getGradient(img):
    xDiff=imgConv(img,Prewitt_x)
    yDiff=imgConv(img,Prewitt_y)
    gradient=np.sqrt(xDiff**2,yDiff**2)
    return gradient

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

'''
# #########
# 文献中提到了两种融合策略：
# 1.
# 对两个图像的拉普拉斯金字塔从最底层开始，对每个像素点周围的一片区域求其平均梯度
# 新生成的重构金字塔每个对应像素点取值为平均梯度较大的那个像素点
# 平均梯度高往往代表了一个像素点很可能是特征点，因此这样重构可以互补地融合两张图里的特征
# 就是代码实现起来非常丧心病狂，而且计算量相当大。。。
# 2.
# 最高层采用0.5-0.5加权平均融合，下面每层用极大值融合
# 代码实现稍微容易一些。。。
# #########
'''

# 实现策略1
def reconstruct1(lp1,lp2):
    dep=len(lp1)
    assert dep==len(lp2)
    LS=[]
    for la,lb in zip(lp1,lp2):
        row,col,dpt=la.shape
        tmp=np.zeros((row,col))
        la_gradient=getGradient(la)
        lb_gradient=getGradient(lb)
        for i in range(row):
            for j in range(col):
                tmp[i,j]=la[i,j][0] if la_gradient[i,j][0]>lb_gradient[i,j][0] else lb[i,j][0]
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
    rowFirst,colFirst,dptFirst=ta.shape
    tmpFirst=np.zeros((rowFirst,colFirst))
    for i in xrange(rowFirst):
        for j in xrange(colFirst):
            tmpFirst[i,j] = ta[i,j][0]/2 + tb[i,j][0]/2
    LS.append(tmpFirst)
    for k in range(dep):
        if k==0:
            continue
        la=lp1[k]
        lb=lp2[k]
        row,col,dpt = la.shape
        tmp = np.zeros((row,col))
        for i in xrange(row):
            for j in xrange(col):
                tmp[i,j] = la[i,j][0] if la[i,j][0] > lb[i,j][0] else lb[i,j][0]
        LS.append(tmp)
    ls_reconstruct=LS[0]
    for i in xrange(1,dep-1):
        ls_reconstruct=cv2.pyrUp(ls_reconstruct)
        ls_reconstruct=cv2.add(sameSize(ls_reconstruct,LS[i]),LS[i])
    return ls_reconstruct

def testFusion(org1,org2,result):
    plt.subplot(131),plt.imshow(cv2.cvtColor(org1,cv2.COLOR_BGR2RGB))
    plt.title("apple"),plt.xticks([]),plt.yticks([])
    plt.subplot(132),plt.imshow(cv2.cvtColor(org2,cv2.COLOR_BGR2RGB))
    plt.title("orange"),plt.xticks([]),plt.yticks([])
    plt.subplot(133),plt.imshow(result,cmap='gray')
    plt.title("laplace_pyramid"),plt.xticks([]),plt.yticks([])
    plt.show()

if __name__=='__main__':
    apple = cv2.imread("F:\\Python\\try\\BasicImageOperation\\apple.jpg")
    orange = cv2.imread("F:\\Python\\try\\BasicImageOperation\\orange.jpg")
    gp_apple=getGaussPyr(apple,6)
    gp_orange=getGaussPyr(orange,6)
    lp_apple=getLaplacePyr(gp_apple)
    lp_orange=getLaplacePyr(gp_orange)
    result=reconstruct1(lp_apple,lp_orange)
    testFusion(apple,orange,result)
