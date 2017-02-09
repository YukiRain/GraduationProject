import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import signal
from PIL import Image

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

def Gauss_func(x,y,sigma=1):
    return 100*(1/(2*np.pi*sigma))*np.exp(-((x-2)**2+(y-2)**2)/(2.0*sigma**2))
Gauss=np.fromfunction(Gauss_func,(5,5),sigma=5)

def imgconv(imgArray,Prewitt):
    '''计算卷积
        parameter:
        imgArray 原灰度图像矩阵
        Prewitt      算子
        返回变换结果的矩阵
    '''
    img=imgArray.copy()
    dim1,dim2=img.shape
    for x in range(1,dim1-1):
        for y in range(1,dim2-1):
            img[x,y]=(imgArray[(x-1):(x+2),(y-1):(y+2)]*Prewitt).sum()

    img=img*(255.0/img.max())
    return img

def Prewitt_plt(path):
    img=Image.open(path).convert('L')
    imgArray=np.array(img)
    img_x=imgconv(imgArray,Prewitt_x)      #x-axis difference
    img_y=imgconv(imgArray,Prewitt_y)      #y-axis difference
    grad=np.sqrt(img_x**2,img_y**2)        #calculate gradient
    grad=grad*(255.0/grad.max())           #adjust to 0-255 gray-img
    #start to plot
    plt.subplot(221)
    plt.imshow(imgArray,cmap='gray')
    plt.axis("off")
    plt.subplot(222)
    plt.imshow(img_x,cmap='gray')
    plt.axis("off")
    plt.subplot(223)
    plt.imshow(img_y,cmap='gray')
    plt.axis("off")
    plt.subplot(224)
    plt.imshow(grad,cmap='gray')
    plt.axis("off")
    plt.show()

def Laplace_plt(path):                                                   #直接使用scipy.signal.convolve()函数
    img=Image.open(path).convert('L')
    imgArray=np.array(img)
    imgBlur=signal.convolve2d(imgArray,Gauss,mode='same')
    img2=signal.convolve2d(imgBlur,Laplace_ex,mode='same')
    img2=(img2/float(img2.max()))*255.0
    img2[img2>img2.mean()]=255
    # convolve=signal.convolve2d(imgArray,Laplace,mode='same')             #para1:img matrix
    # convolve_ex=signal.convolve2d(imgArray,Laplace_ex,mode='same')       #para2:算子
    # convolve=(convolve/float(convolve.max()))*255
    # convolve_ex=(convolve_ex/float(convolve_ex.max()))*255
    # convolve[convolve>convolve.mean()]=255
    # convolve_ex[convolve_ex>convolve_ex.mean()]=255
    #start to plot
    plt.subplot(211)
    plt.imshow(imgArray,cmap='gray')
    plt.axis('off')
    plt.subplot(212)
    plt.imshow(img2,cmap='gray')
    plt.axis('off')
    # plt.subplot(224)
    # plt.imshow(convolve_ex,cmap='gray')
    # plt.axis('off')
    plt.show()

path="example.png"
Laplace_plt(path)