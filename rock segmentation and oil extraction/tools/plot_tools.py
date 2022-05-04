# @Time    : 2021/4/14 19:18
# @Author  : Monlter
# @FileName: plot_tools.py
# @Software: PyCharm

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def img_save(img_arr, file_name):
    img = Image.fromarray(img_arr)
    img.save(file_name)


def plotRGBHist(img_arr):
    r, g, b = np.split(img_arr, 3, axis=2)    # 分离RGB
    plt.figure("RGBHist")
    ar = np.array(r).flatten()
    ag = np.array(g).flatten()
    ab = np.array(b).flatten()
    plt.subplot(2, 2, 1)
    plt.imshow(img_arr)
    plt.subplot(2, 2, 2)
    plt.hist(ar, bins=256, density=1, facecolor='red', edgecolor='red')
    plt.subplot(2, 2, 3)
    plt.hist(ag, bins=256, density=1, facecolor='green', edgecolor='green', alpha=0.6)
    plt.subplot(2, 2, 4)
    plt.hist(ab, bins=256, density=1, facecolor='blue', edgecolor='blue', alpha=0.3)
    plt.show()


def plotOddHist(img_arr, color):
    r, g, b = np.split(img_arr, 3, axis=2)    # 分离RGB
    plt.figure("OddHist")
    odd_channel = r if color == 'red' else(g if color == 'green' else b)
    a = np.array(odd_channel).flatten()
    plt.subplot(2, 1, 1)
    plt.imshow(odd_channel)
    plt.subplot(2, 1, 2)
    plt.hist(a, bins=256, density=1, facecolor=color, edgecolor=color)
    plt.show()

def plotHist(img_gray):
    plt.hist(img_gray.ravel(), bins=255, color='g')
    plt.show()


def plotRGBImg(img):
    img_arr = np.array(img)

    r = img_arr.copy()
    r[:, :, 1:3] = 0
    g = img_arr.copy()
    g[:, :, ::2] = 0
    b = img_arr.copy()
    b[:, :, :2] = 0

    plt.figure("BGBImg")
    plt.subplot(2, 2, 1)
    plt.imshow(img_arr)
    plt.subplot(2, 2, 2)
    plt.imshow(r)
    plt.subplot(2, 2, 3)
    plt.imshow(g)
    plt.subplot(2, 2, 4)
    plt.imshow(b)
    plt.show()

    return r, g, b

