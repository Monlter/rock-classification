import argparse
import cv2
import numpy as np

from tools.allmethods import thresh_grabcut, mouse_grabcut,watershed,extract_oil,cv_show,calculate_oleaginousness
from tools.plot_tools import plotOddHist,plotHist,plotRGBImg
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='./data/2/350.jpg', dest='input')
    parser.add_argument('-o', '--output', default="./OUT", dest='output_dir')
    parser.add_argument('-m', '--method', default="1", dest='method')
    args = parser.parse_args()
    return args


def open_img(img_name):
    img = cv2.imread(img_name)
    img_width = img.shape[0]
    img_height = img.shape[1]
    scale_size = 800 * img_width // img_height
    if img_width > 800:
        img = cv2.resize(img, (800, scale_size), interpolation=cv2.INTER_AREA)
    return img


if __name__ == '__main__':
    # 方法0，1，2
    methods = [thresh_grabcut, mouse_grabcut, watershed]
    args = get_args()
    # 加载图片
    img = open_img(args.input)
    seg_mask = methods[int(args.method)](img)
    # 计算油气分量
    mask = seg_mask / 255
    cv_show("mask", seg_mask)
    seg_img = (img * mask[:, :, np.newaxis]).astype("uint8")
    cv_show("seg_img", seg_img)
    dst = extract_oil(seg_img)
    cv_show("mask", dst)
    oil_percent = calculate_oleaginousness(seg_mask, dst)
    print("oil_precent:", oil_percent)

