# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from math import log,sqrt
import cv2,pywt,os
from PIL import Image
import fitting

src1_path = 'F:\\Python\\try\\BasicImageOperation\\disk1.jpg'
src2_path = 'F:\\Python\\try\\BasicImageOperation\\disk2.jpg'
src_dir='F:\\GraduationProject\\IMG\\splitTest\\'
target_dir='F:\\GraduationProject\\IMG\\result\\'

# 测试绘图用的数据
# entropies=[1,2,3,4,5,6,7,8]
# variances=[9,8,7,6,5,4,3,2]
# gradients=[1,3,5,7,9,2,4,6]
# frequencies=[0,8,6,4,2,1,3,5]

'''
    2017.Apr.2nd:
    上层跑数据的东西基本完成了，用get_party()函数可以绘制出每组图像的参数柱状图
    现在的问题是，各种算法对于不同图片融合的参数规律分布十分杂乱无章，从上面绘制的柱状图来看很难发现特别的规律
    所以考虑用scatter绘图，如果顺利的话可以更清晰的看到不同算法的参数规律（维度不得不限制在二维）
    如果这样还是看不出规律，就考虑用LWLR来拟合一下，这样可以选择更高的特征维度，或许可以得到一个更接近真相的结果
    选择LWLR的原因是：如果源图像参数较为接近的话，就可以在一定程度上认为这些图片近似
    近似的图像在相同的算法下理论上讲应当表现出近似的参数，所以应该赋一个较大的权值，而在样本空间里离得较远的点会被赋一个较小的权值
    希望这样可以得到一个比较好的结果吧，如果不行我也没办法了OTZ
'''

# 为降低程序耦合，把计算熵之前的统计过程封装起来
def getLabels(img,addLabel=None):
    labels={} if addLabel==None else addLabel
    row,col=img.shape
    for i in xrange(row):
        for j in xrange(col):
            # 限定len(labels)最大只能是26，否则算起来太零碎了
            greyLabel=int(img[i,j]/10)
            if greyLabel not in labels.keys():
                labels[greyLabel]=0
            labels[greyLabel]+=1
    return labels

# 计算熵
def entropy(img):
    labels=getLabels(img)
    row,col=img.shape
    ent=0.00
    entrySize=row*col
    for key in labels:
        prob=float(labels[key])/entrySize
        ent-=prob*log(prob,2)
    return ent

# 计算两张图熵的差值
def entropyDiff(apple,orange):
    appleEnt=entropy(apple)
    orangeEnt=entropy(orange)
    return appleEnt-orangeEnt

# 计算交叉熵(以第一个变量为基准)
def crossEntropy(apple,orange):
    assert apple.shape==orange.shape
    appleLabels=getLabels(apple)
    orangeLabels=getLabels(orange)
    entrySize=apple.shape[0]*apple.shape[1]
    XEnt = 0.00
    infoSet=appleLabels.keys()+orangeLabels.keys()
    for key in infoSet:
        if key in appleLabels.keys():
            prob = float(appleLabels[key])/entrySize
        if key in orangeLabels.keys():
            q_prob=float(orangeLabels[key])/entrySize
        else:
            continue
        XEnt-=prob*log(prob/q_prob,2)
    return XEnt

# 平均梯度
def avrGradient(img):
    row,col=img.shape
    size=row*col
    ans=0
    xDiff=cv2.Sobel(img,cv2.CV_16S,1,0)
    yDiff=cv2.Sobel(img,cv2.CV_16S,0,1)
    stdXdiff = cv2.convertScaleAbs(xDiff)
    stdYdiff = cv2.convertScaleAbs(yDiff)
    gradient=np.sqrt(stdXdiff**2+stdYdiff**2)
    for i in xrange(row):
        for j in xrange(col):
            ans+=gradient[i,j]/float(size)
    return ans

# 计算空间平均频率
def avrFreq(img):
    row,col=img.shape
    size=row*col
    xFreq=0;yFreq=0
    for i in range(row-1):
        for j in range(col-1):
            x_tmp=abs(float(img[i+1,j])-float(img[i,j]))
            y_tmp=abs(float(img[i,j+1])-float(img[i,j]))
            xFreq+=x_tmp/size
            yFreq+=y_tmp/size
    return sqrt(xFreq**2+yFreq**2)

# 平均对比度
def getContrast(img):
    lowFreq,highFreq=pywt.wavedec2(img,'haar',level=1)
    row,col=img.shape
    ctr=np.zeros((row,col))
    size=len(highFreq)
    for array in highFreq:
        ctr+=(array/lowFreq)**2/size
    return ctr.sum()*100/(row*col)

# 均方根误差
def avrSquareError(src,std):
    pass
    return

# 用matplotlib画灰度分布图
def greyDistributionPlot(img,mode='bar'):
    labels=getLabels(img)
    keys=labels.keys()
    vals=labels.values()
    if 'bar' in mode.lower():
        plt.plot(keys,vals,color='r')
        plt.bar(keys,vals,color='b')
    elif 'pie' in mode.lower():
        plt.pie(vals,labels=keys,shadow=True)
    else:
        print 'Wrong Mode'
        return None
    plt.title('GREY DISTRIBUTION')
    plt.show()

# 这个类里包括了所有图像的参数，融合完可以搞一个这个类的list方便画直方图
class imgParty(object):
    def __init__(self,path=None,array=None):
        if path!=None:
            self.img=Image.open(path).convert('L').resize((512,512))
        if array!=None:
            self.array=array
        else:
            self.array = np.array(self.img)
        self.entropy=None
        self.stdVar=None
        self.avrGrey=None
        self.var=None
        self.gradient=None
        self.greyDist=None
        self.avrFreq=None
        self.labels=None
        self.values=None

    def getPara(self):
        self.entropy=entropy(self.array)
        self.var=self.array.var()
        # self.stdVar=sqrt(self.var)
        self.gradient=avrGradient(self.array)
        # self.greyDist=getLabels(self.array)
        self.avrFreq=avrFreq(self.array)
        # self.labels=self.greyDist.keys()
        # self.values=self.greyDist.values()
        return self

# 将一组图片作为一组保存
class family(object):
    def __init__(self, src1=None, src2=None,
                 pyr_contrast=None, pyr1=None, pyr2=None,
                 var_weighted=None, wave_contrast=None, wave_var=None):
        self.apple=imgParty(path=src1).getPara()
        self.orange=imgParty(path=src2).getPara()
        self.pyr_contrast=imgParty(path=pyr_contrast).getPara()
        self.pyr_result1=imgParty(path=pyr1).getPara()
        self.pyr_result2=imgParty(path=pyr2).getPara()
        self.avr_weighted=imgParty(path=var_weighted).getPara()
        self.wave_contrast=imgParty(path=wave_contrast).getPara()
        self.wavelet=imgParty(path=wave_var).getPara()

    # 可以直接print该类在屏幕上，输出所有参数的表格
    def __str__(self):
        ret = '''
            ************************************************************************************************
             PICTURE             ENTROPY            VARIANCE          AVR_GRADIENT            AVR_FREQUENCY
             SRC1                %f          %f           %f               %f
             SRC2                %f          %f           %f               %f
             PYR_CONTRAST        %f          %f           %f               %f
             WAVE_CONTRAST       %f          %f           %f               %f
             LAPLACE_PYR1        %f          %f           %f               %f
             LAPLACE_PYR2        %f          %f           %f               %f
             AVR_WEIGHTED        %f          %f           %f               %f
             WAVELET             %f          %f           %f               %f
            ************************************************************************************************
        '''
        return ret % (self.apple.entropy, self.apple.var, self.apple.gradient, self.apple.avrFreq,
                      self.orange.entropy, self.orange.var, self.orange.gradient, self.orange.avrFreq,
                      self.pyr_contrast.entropy,self.pyr_contrast.var,
                      self.pyr_contrast.gradient,self.pyr_contrast.avrFreq,
                      self.wave_contrast.entropy, self.wave_contrast.var,
                      self.wave_contrast.gradient, self.wave_contrast.avrFreq,
                      self.pyr_result1.entropy, self.pyr_result1.var,
                      self.pyr_result1.gradient, self.pyr_result1.avrFreq,
                      self.pyr_result2.entropy, self.pyr_result2.var,
                      self.pyr_result2.gradient, self.pyr_result2.avrFreq,
                      self.avr_weighted.entropy, self.avr_weighted.var,
                      self.avr_weighted.gradient, self.avr_weighted.avrFreq,
                      self.wavelet.entropy, self.wavelet.var, self.wavelet.gradient, self.wavelet.avrFreq)

# 返回一个包含family类的list，其中包含20组图片
def get_family():
    src_list=os.listdir(src_dir)
    target_list=os.listdir(target_dir)
    group_list=list()
    for group_index in range(21):
        group_src=filter(lambda x: int(x.split('_')[0])==group_index, src_list)
        group_src=map(lambda x: src_dir+x, group_src)
        group_target=filter(lambda x: int(x.split('_')[0])==group_index, target_list)
        group_target=map(lambda x: target_dir+x, group_target)
        group_target.sort()
        member=family(src1=group_src[0],src2=group_src[1],
                      pyr_contrast=group_target[0],pyr1=group_target[1],pyr2=group_target[2],
                      var_weighted=group_target[3],wave_contrast=group_target[4],wave_var=group_target[5])
        print member
        group_list.append(member)
    return group_list

# 返回一个包含imgParty类的list，里面有20*8张图片
def get_party():
    src_list = os.listdir(src_dir)
    target_list = os.listdir(target_dir)
    ret=list()
    for img_index in range(21):
        party_list = list()
        party_src=filter(lambda x: int(x.split('_')[0])==img_index, src_list)
        party_src=map(lambda x: src_dir+x, party_src)
        party_src.sort()
        for item in party_src:
            party_list.append(imgParty(path=item).getPara())
        party_target=filter(lambda x: int(x.split('_')[0])==img_index, target_list)
        party_target=map(lambda x: target_dir+x, party_target)
        party_target.sort()
        for item in party_target:
            party_list.append(imgParty(path=item).getPara())
        get_bar_plot(party_list)
        ret+=party_list
    return ret

# 每计算出一组图像的参数，绘图一次
def get_bar_plot(party_list):
    entropies=[item.entropy for item in party_list]
    variances=[item.var/500 for item in party_list]
    gradients=[item.gradient/2 for item in party_list]
    frequencies=[item.avrFreq for item in party_list]
    length=np.arange(len(entropies))
    labels=(u'SRC1',u'SRC2',u'PYR_CST',u'WAVE_CST',u'LP_PYR1',u'LP_PYR2',u'WEIGHTED',u'WAVELET')
    plt.figure()
    plt.title('PARAMETERS')
    plt.bar(length, entropies, width=0.2, color='r')
    plt.bar(length+0.2, variances, width=0.2, color='g')
    plt.bar(length + 0.4,gradients,width=0.2,color='b')
    plt.bar(length + 0.6,frequencies,width=0.2,color='y')
    plt.xticks(np.arange(len(labels)),labels)
    plt.show()

# 数据感觉像是随机的一样，考虑用scatter画出来看看有没有规律
def get_scatter():
    groups=get_family()
    src1_ent=[item.apple.entropy for item in groups]
    src2_ent=[item.orange.entropy for item in groups]
    avr_src_ent=map(lambda first,second: (first+second)/2.0, src1_ent, src2_ent)
    avr_src_ent=np.array(avr_src_ent)

    pyr_contrast_ent=np.array([item.pyr_contrast.entropy for item in groups])
    pyr1_ent=np.array([item.pyr_result1.entropy for item in groups])
    pyr2_ent=np.array([item.pyr_result2.entropy for item in groups])
    weighted_ent=np.array([item.avr_weighted.entropy for item in groups])
    wave_contrast_ent=np.array([item.wave_contrast.entropy for item in groups])
    wavelet_ent=np.array([item.wavelet.entropy for item in groups])

    avr_src_ent *= 100.0/avr_src_ent.max()
    pyr_contrast_ent *= 100.0/pyr_contrast_ent.max()
    pyr1_ent *= 100.0/pyr1_ent.max()
    pyr2_ent *= 100.0/pyr2_ent.max()
    weighted_ent *= 100.0/weighted_ent.max()
    wave_contrast_ent *= 100.0/wave_contrast_ent.max()
    wavelet_ent *= 100.0/wavelet_ent.max()

    plt.figure(1)
    plt.title('ENTROPY')
    plt.scatter(avr_src_ent, pyr1_ent, c='b')
    plt.scatter(avr_src_ent, pyr_contrast_ent, c='r')
    plt.scatter(avr_src_ent, pyr2_ent, c='g')
    plt.scatter(avr_src_ent, wavelet_ent, c='y')
    plt.scatter(avr_src_ent, weighted_ent, c='k')
    plt.scatter(avr_src_ent, wave_contrast_ent, c='c')
    plt.show()

    # pyr_contrast_var=np.array([item.pyr_contrast.var for item in groups])
    # pyr1_var=np.array([item.pyr_result1.var for item in groups])
    # pyr2_var=np.array([item.pyr_result2.var for item in groups])
    # weighted_var=np.array([item.avr_weighted.var for item in groups])
    # wave_contrast_var=np.array([item.wave_contrast.var for item in groups])
    # wavelet_var=np.array([item.wavelet.var for item in groups])

    # plt.figure(2)
    # plt.title('VARIANCE')
    # plt.scatter(avr_src_ent, pyr1_ent, 'b')
    # plt.scatter(avr_src_ent, pyr_contrast_ent, 'r')
    # plt.scatter(avr_src_ent, pyr2_ent, 'g')
    # plt.scatter(avr_src_ent, wavelet_ent, 'y')
    # plt.scatter(avr_src_ent, weighted_ent, 'w')
    # plt.scatter(avr_src_ent, wave_contrast_ent, 'c')




# 实在不行我们还可以拟合一条直线出来，考虑一下，参数相近的图片可以在一定程度上认为他们是相近的，所以用LWLR拟合比较合适


if __name__ == '__main__':
    get_scatter()