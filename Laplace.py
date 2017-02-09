import cv2
import numpy as np
from matplotlib import pyplot as plt


def sameSize(img1, img2):
    """
    使得img1的大小与img2相同
    """
    rows, cols, dpt = img2.shape
    dst = img1[:rows,:cols]
    return dst


apple = cv2.imread("E:\python\Python Project\opencv_showimage\images\\apple.jpg")
orange = cv2.imread("E:\python\Python Project\opencv_showimage\images\\orange.jpg")

# 对apple进行6层高斯降采样
G = apple.copy()
gp_apple = [G]
for i in xrange(6):
    G = cv2.pyrDown(G)
    gp_apple.append(G)

# 对orange进行6层高斯降采样
G = orange.copy()
gp_orange = [G]
for j in xrange(6):
    G = cv2.pyrDown(G)
    gp_orange.append(G)

# 求apple的Laplace金字塔
lp_apple = [gp_apple[5]]
for i in xrange(5,0,-1):
    GE = cv2.pyrUp(gp_apple[i])
    L = cv2.subtract(gp_apple[i-1], sameSize(GE,gp_apple[i-1]))
    lp_apple.append(L)

# 求orange的Laplace金字塔
lp_orange = [gp_orange[5]]
for i in xrange(5,0,-1):
    GE = cv2.pyrUp(gp_orange[i])
    L = cv2.subtract(gp_orange[i-1], sameSize(GE,gp_orange[i-1]))
    lp_orange.append(L)

# 对apple和orange的Laplace金字塔进行1/2拼接
LS = []
for la,lb in zip(lp_apple,lp_orange):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:cols/2],lb[:,cols/2:]))
    LS.append(ls)

# 对拼接后的Laplace金字塔重建获取融合后的结果
ls_reconstruct = LS[0]
for i in xrange(1,6):
    ls_reconstruct = cv2.pyrUp(ls_reconstruct)
    ls_reconstruct = cv2.add(sameSize(ls_reconstruct,LS[i]), LS[i])

# 各取1/2直接拼接的结果
r,c,depth = apple.shape
real = np.hstack((apple[:,0:c/2],orange[:,c/2:]))

plt.subplot(221), plt.imshow(cv2.cvtColor(apple,cv2.COLOR_BGR2RGB))
plt.title("apple"),plt.xticks([]),plt.yticks([])
plt.subplot(222), plt.imshow(cv2.cvtColor(orange,cv2.COLOR_BGR2RGB))
plt.title("orange"),plt.xticks([]),plt.yticks([])
plt.subplot(223), plt.imshow(cv2.cvtColor(real,cv2.COLOR_BGR2RGB))
plt.title("real"),plt.xticks([]),plt.yticks([])
plt.subplot(224), plt.imshow(cv2.cvtColor(ls_reconstruct,cv2.COLOR_BGR2RGB))
plt.title("laplace_pyramid"),plt.xticks([]),plt.yticks([])
plt.show()

