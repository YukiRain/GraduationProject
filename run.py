# -*- coding: utf-8 -*-
import contrast,wavelet,quality
import Laplace
import os
from PIL import Image
import numpy as np

img_dir='F:\\GraduationProject\\IMG\\splitTest\\'
save_dir='F:\\GraduationProject\\IMG\\result\\'

# 每两张图像放到一起作为一组，使用quality.imgParty组合来表示
class union_img(object):
    def __init__(self,apple,orange):
        self.src1=quality.imgParty(path=apple)
        self.src2=quality.imgParty(path=orange)
        self.cst_wave_result=None
        self.lp_result=None
        self.avr_result=None
        self.wvlt_result=None

    def get_src_para(self):
        self.src1.getPara()
        self.src2.getPara()
        return self

    def fusion(self,cst=True,laplace=True,wvlt=True):
        self.get_src_para()
        if cst:
            self.cst_wave_result,self.cst_pyr_result=contrast.runTest(src1=self.src1.array,
                                                                 src2=self.src2.array,
                                                                 isplot=False)
        if laplace:
            self.lp_result=Laplace.runTest(src1=self.src1.array,src2=self.src2.array,isPlot=False)
        if wvlt:
            self.wvlt_result=wavelet.runTest(src1=self.src1.array,src2=self.src2.array)
        return self

    def get_target_para(self):
        self.src1.getPara()
        self.src2.getPara()
        self.cst_wave=quality.imgParty(array=self.cst_wave_result).getPara()
        self.cst_pyr=quality.imgParty(array=self.cst_pyr_result).getPara()
        self.lp=quality.imgParty(array=self.lp_result).getPara()
        self.avr=quality.imgParty(array=self.avr_result).getPara()
        self.wave=quality.imgParty(array=self.wvlt_result).getPara()
        return self

    def __unicode__(self):
        ret='''
            **************************************************************************************************
            PARAMETERS:             ENTROPY     AVRAGE GRADIENT   AVERAGE VARIANCE   STD VARIANCE    WAVE
            SOURCE1:                %f          %f                %f                 %f              %f
            SOURCE2:                %f          %f                %f                 %f              %f
            LAPLACE PYRAMID:        %f          %f                %f                 %f              %f
            CONTRAST PYRAMID:       %f          %f                %f                 %f              %f
            WAVELET:                %f          %f                %f                 %f              %f
            **************************************************************************************************
        '''
        return ret % (self.src1.entropy,self.src1.gradient,self.src1.var,self.src1.stdVar,
                      self.src2.entropy,self.src2.gradient,self.src2.var,self.src2.stdVar,
                      self.lp.entropy,self.lp.gradient,self.lp.var,self.lp.stdVar,
                      self.cst_wave.entropy,self.cst_wave.gradient,self.cst_wave.var,self.cst_wave.stdVar,
                      self.wave.entropy,self.wave.gradient,self.wave.var,self.wave.stdVar)


def resize_img_list():
    img_list=os.listdir(img_dir)
    for img_index in range(21):
        src_list = filter(lambda x:int(x.split('_')[0]) == img_index,img_list)
        src1 = Image.open(img_dir + '\\' + src_list[0]).convert('L')
        src2 = Image.open(img_dir + '\\' + src_list[1]).convert('L')
        print 'PREVIOUS (%d, %d) ' %(src2.width, src2.height)
        arr = np.array(src1)
        new_src=src2.resize((arr.shape[1], arr.shape[0]),Image.BILINEAR)
        new_src.convert('RGB').save(img_dir + '\\' + src_list[1])
        # src2.resize((arr.shape[1], arr.shape[0])).convert('RGB').save(img_dir+'\\'+src_list[1])
        print 'RESIZE TO (%d, %d))' % (new_src.width, new_src.height)
    print 'FINISHED'

def load_img(isFusion=True):
    img_list=os.listdir(img_dir)
    union_list=list()
    for img_index in range(21):
        src_list=filter(lambda x:int(x.split('_')[0])==img_index, img_list)
        tmp = union_img(img_dir + '\\' + src_list[0],img_dir + '\\' + src_list[1])
        union_list.append(tmp)
    if isFusion:
        union_list=map(lambda item:item.fusion().get_target_para(), union_list)
    return union_list

def run():
    fusion_list=load_img()
    for item in fusion_list:
        print item

def var_weighted_fusion(apple,orange):
    apple_var=apple.var()
    apple_weight=apple_var/(apple_var+orange.var())
    orange_weight=1-apple_weight
    return apple*apple_weight+orange*orange_weight

def var_weight_out():
    img_list=os.listdir(img_dir)
    for img_index in range(21):
        src_list = filter(lambda x: int(x.split('_')[0]) == img_index, img_list)
        apple,orange=contrast.imgOpen(img_dir+src_list[0], img_dir+src_list[1])
        result=var_weighted_fusion(apple,orange)
        Image.fromarray(result).convert('RGB').save(save_dir+str(img_index)+'_var_weighted.jpg')
        print 'SAVING TO %d' % (img_index)
    print 'FINISHED'

if __name__=='__main__':
    pass

    # resize_img_list()
    # run()