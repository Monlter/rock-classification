# @Time    : 2021/3/19 21:37
# @Author  : 我丶老陈 
# @FileName: data_trans.py
# @Software: PyCharm

import os
from PIL import Image


def bmp_to_jpg(file_path):
    """创建转换后的文件夹"""
    pre_path = file_path + "_trans"
    if not os.path.exists(pre_path):
        os.mkdir(pre_path)

    """生成jpg文件并保存"""
    for filename in os.listdir(file_path):
        filename_trans = os.path.splitext(filename)[0]
        class_num = filename_trans.split('-')[1]
        save_path = os.path.join(pre_path, class_num)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        img = Image.open(os.path.join(file_path, filename), 'r')
        new_img_name = filename_trans.split('-')[0] + '.jpg'
        print(new_img_name)
        img.save(os.path.join(save_path, new_img_name))


if __name__ == '__main__':
    file_path = 'dataset/rock'
    bmp_to_jpg(file_path)

