# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt,log

# 最小二乘回归
def standRegression(xArr,yArr):
    xMat=np.mat(xArr); yMat=np.mat(yArr).T
    xTx=xMat.T*xMat
    assert np.linalg.det(xTx)!=0
    '''
        矩阵必须保证可逆，即xMat数据量大小必须大于等于xMat的维度（xMat线性无关时可以取等号）
    '''
    ws=xTx.I * (xMat.T*yMat)
    return ws

# 局部加权线性回归
def LWLR(testPoint,xArr,yArr,k=1.0):
    xMat=np.mat(xArr); yMat=np.mat(yArr).T
    row,col=xMat.shape
    weights = np.mat(np.eye((row)))
    for i in range(row):
        diffMat=testPoint-xMat[i,:]
        weights[i,i]=np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx=xMat.T*(weights*xMat)
    assert np.linalg.det(xTx)!=0
    '''
        矩阵必须保证可逆，即xMat数据量大小必须大于等于xMat的维度（xMat线性无关时可以取等号）
    '''
    ws=xTx.I*(xMat.T*(weights*yMat))
    return testPoint*ws