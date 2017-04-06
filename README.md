# GraduationProject

contrast.py：基于像素清晰度的融合算法
gradient.py：梯度计算的实验，看看哪个库算梯度算起来比较快
ImageFusion.py：基本融合算法（不包含带通滤波）
imgSplit.py：从论文里找出来的图像很多都是很多张在一个图里，这个小脚本实现了自动切图的基本功能
Laplace.py：Laplace金字塔，两种融合策略
quality.py：融合图像质量的量化评估
run.py：对图像数据库中所有图像跑一遍所有的算法，并且向另一个文件夹输出结果
wavelet.py：基于小波变换的融合策略实现
source.py：使用PyQt5写的一个简单的带GUI的融合窗口
fitting.py：最小二乘回归与局部加权线性回归