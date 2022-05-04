# @Time    : 2021/4/14 19:52
# @Author  : Monlter
# @FileName: segfunction.py
# @Software: PyCharm
import numpy as np


def iterThresh(img_gray):
    img_gray = np.array(img_gray)
    zmax = np.max(img_gray)
    zmin = np.min(img_gray)
    ith_old = 0
    ith_new = (zmax + zmin) / 2
    while ith_old != ith_new:
        zo = np.mean(img_gray[np.where(img_gray > ith_new)])
        zb = np.mean(img_gray[np.where(img_gray < ith_new)])
        ith_old = ith_new
        ith_new = (zo + zb) / 2
    print("old:",ith_old,"new:",ith_new)
    print('iter th:', ith_new)

    return ith_new


def threshSegImg(img_gray, thresh):
    img_bool = img_gray > thresh
    img_gray = np.array(img_gray)
    img_gray[img_bool] = 255
    img_gray[~img_bool] = 0
    return img_gray


def delete_contours(contours, delete_list):
    delta = 0
    for i in range(len(delete_list)):
        # print("i= ", i)
        del contours[delete_list[i] - delta]
        delta = delta + 1
    return contours
