# RockClassifier

#### 介绍
基于深度学习的岩石图像分类

#### 软件架构
深度学习框架：Pytorch
深度学习网络模型ResNet50


#### 安装教程

1. 安装python3.8环境
2. 安装pytorch1.7
3. 安装cuda10.2

#### 使用说明

1. 进入项目目录
2. 执行train.py
3. TensorboardX的默认目录是项目根目录下的log文件夹
4. 数据集的目录格式为：
训练集：项目根目录/data/rock_dataset/rock_trans/1/TRAIN/*(每个类别的编号从0到6）/*.jpg
验证集：项目根目录/data/rock_dataset/rock_trans/1/EVAL/*(每个类别的编号从0到6）/*.jpg
测试集：项目根目录/data/rock_dataset/rock_trans/1/TEST/*(每个类别的编号从0到6）/*.jpg
#### 参与贡献

1. Fork 本仓库
2. 新建 Feat_xxx 分支
3. 提交代码
4. 新建 Pull Request


#### 码云特技

1. 使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2. 码云官方博客 [blog.gitee.com](https://blog.gitee.com)
3. 你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解码云上的优秀开源项目
4. [GVP](https://gitee.com/gvp) 全称是码云最有价值开源项目，是码云综合评定出的优秀开源项目
5. 码云官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6. 码云封面人物是一档用来展示码云会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)