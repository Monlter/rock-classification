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


