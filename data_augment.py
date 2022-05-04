# @Time    : 2021/3/20 15:28
# @Author  : 我丶老陈 
# @FileName: data_augment.py
# @Software: PyCharm

import numpy as np
import Augmentor
from os import listdir
from PIL import Image
import os

def augmentation(img_path,mask_path):
    imglist = listdir(img_path)
    img0 = Image.open(os.path.join(img_path, imglist[0]))
    width, height = img0.size[0], img0.size[1]
    p = Augmentor.Pipeline(img_path)
    p.ground_truth(mask_path)
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.zoom(probability=0.3, min_factor=1.1, max_factor=1.6)
    p.resize(probability=1, height=height, width=width)
    p.sample(10)


if __name__ == '__main__':
    img_path = 'dataset/test/1'
    mask_path = 'dataset/test/2'
    augmentation(img_path, mask_path)