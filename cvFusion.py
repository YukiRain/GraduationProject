import numpy as np
import matplotlib.pyplot as plt
import cv2

def cvCall(path1,path2):
    src1=cv2.imread(path1)
    src2=cv2.imread(path2)
    dst=cv2.addWeighted(src1,0.5,src2,0.5,0)
    plt.subplot(311)
    plt.imshow(src1,cmap='gray')
    plt.subplot(312)
    plt.imshow(src2,cmap='gray')
    plt.subplot(313)
    plt.imshow(cv2.cvtColor(dst,cv2.COLOR_BGR2RGB))
    plt.show()

if __name__=='__main__':
    apple='F:\\GraduationProject\\IMG\\test\\apple.jpg'
    orange='F:\\GraduationProject\\IMG\\test\\orange.jpg'
    cvCall(apple,orange)
