# GraduationProject
ImageFusion

学机器学习的时候冒出一个想法：
很多论文指出判断融合图像质量的好坏不应当总以肉眼识别为标准，应当建立一种可以量化融合图片质量的数学模型
但是论文里同时也指出这种数学模型现在还没有统一的标准
也就是说，对所有需要融合的图片，可以用机器学习的方法，先将图片的一些指标生成数据表格（就是sklearn里可以用的list[list[]]格式）
这个表格可以包括融合前与融合后图像的均方根误差，平均方差，平均梯度，结构相似度
然后把数据全部扔进决策树（可以用ID3算法，虽然我还不太能看懂），前期人工确定图片融合的质量好坏，由此来训练决策树
当数据量足够大的时候，或许就可以由决策树来自己判断对于某种指标的图片，应当优先使用哪种feature来判断图片适合使用的算法了