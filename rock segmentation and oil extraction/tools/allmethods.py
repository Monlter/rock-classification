"""
 @Time    : 2021/4/20 16:55
 @Author  : Monlter
 @FileName: allmethods.py
 @Software: PyCharm
"""
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import morphology, color, data, filters,segmentation
import numpy as np
import cv2
import time


drawing = False
mode = False


def cv_show(name, img):
    cv2.namedWindow(name, 0)
    cv2.resizeWindow(name, 640, 640)
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


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


def extract_oil(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([35, 43, 46])
    upper_hsv = np.array([77, 255, 255])
    mask = cv2.inRange(hsv_img, lowerb=lower_hsv, upperb=upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dst = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return dst


def calculate_oleaginousness(ori_img, seg_img):
    num_all = len(ori_img[ori_img == 255])
    num_oil = len(seg_img[seg_img == 255])
    return num_oil/num_all


class GrabCut:
    def __init__(self, t_img):
        self.img = t_img
        self.img_raw = self.img.copy()
        self.img_width = self.img.shape[0]
        self.img_height = self.img.shape[1]
        # self.scale_size = 640 * self.img_width // self.img_height
        # if self.img_width > 640:
        #     self.img = cv2.resize(self.img, (640, self.scale_size), interpolation=cv2.INTER_AREA)
        self.img_show = self.img.copy()
        self.img_gc = self.img.copy()
        self.img_gc = cv2.GaussianBlur(self.img_gc, (3, 3), 0)
        self.lb_up = False
        self.rb_up = False
        self.lb_down = False
        self.rb_down = False
        self.mask = np.full(self.img.shape[:2], 2, dtype=np.uint8)
        self.firt_choose = True

def re_max_area(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # hierarchy[i]: [Next，Previous，First_Child，Parent]
    # 要求没有父级轮廓
    delete_list = []
    c, row, col = hierarchy.shape
    for i in range(row):
        if hierarchy[0, i, 2] > 0 or hierarchy[0, i, 3] > 0:  # 有父轮廓或子轮廓
            pass
        else:
            delete_list.append(i)

    # 根据列表序号删除不符合要求的轮廓
    contours = delet_contours(contours, delete_list)
    temp = np.ones(mask.shape, dtype=np.uint8) * 255
    result = cv2.drawContours(temp, contours, -1, (0, 0, 0), 2)
    area = []

    # 找到最大的轮廓
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))

    # 填充最大的轮廓
    mask = cv2.drawContours(result, contours, max_idx, 0, cv2.FILLED)
    out_mask = np.ones(mask.shape, dtype=np.uint8) * 255
    out_mask[mask == 255] = 0
    return out_mask

def delet_contours(contours, delete_list):
    delta = 0
    for i in range(len(delete_list)):
        # print("i= ", i)
        del contours[delete_list[i] - delta]
        delta = delta + 1
    return contours

# 鼠标的回调函数
def mouse_event2(event, x, y, flags, param):
    global drawing, last_point, start_point
    # 左键按下：开始画图
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_point = (x, y)
        start_point = last_point
        param.lb_down = True
        print('mouse lb down')
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = True
        last_point = (x, y)
        start_point = last_point
        param.rb_down = True
        print('mouse rb down')
    # 鼠标移动，画图
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if param.lb_down:
                cv2.line(param.img_show, last_point, (x, y), (0, 0, 255), 2, -1)
                cv2.rectangle(param.mask, last_point, (x, y), 1, -1, 4)
            else:
                cv2.line(param.img_show, last_point, (x, y), (255, 0, 0), 2, -1)
                cv2.rectangle(param.mask, last_point, (x, y), 0, -1, 4)
            last_point = (x, y)
    # 左键释放：结束画图
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        param.lb_up = True
        param.lb_down = False
        cv2.line(param.img_show, last_point, (x, y), (0, 0, 255), 2, -1)
        if param.firt_choose:
            param.firt_choose = False
        cv2.rectangle(param.mask, last_point, (x, y), 1, -1, 4)
        print('mouse lb up')
    elif event == cv2.EVENT_RBUTTONUP:
        drawing = False
        param.rb_up = True
        param.rb_down = False
        cv2.line(param.img_show, last_point, (x, y), (255, 0, 0), 2, -1)
        if param.firt_choose:
            param.firt_choose = False
            param.mask = np.full(param.img.shape[:2], 3, dtype=np.uint8)
        cv2.rectangle(param.mask, last_point, (x, y), 0, -1, 4)
        print('mouse rb up')


def watershed(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = filters.rank.median(image, morphology.disk(3))  # 过滤噪声

    # 将梯度值低于12的作为开始标记点
    markers = filters.rank.gradient(denoised, morphology.disk(6)) < 12
    markers = ndi.label(markers)[0]

    gradient = filters.rank.gradient(denoised, morphology.disk(5))  # 计算梯度

    labels = segmentation.watershed(gradient, markers, mask=image)  # 基于梯度的分水岭算法

    labels = labels.astype(np.uint8)
    ret, thresh = cv2.threshold(labels, 20, 225, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # hierarchy[i]: [Next，Previous，First_Child，Parent]
    # 要求没有父级轮廓
    delete_list = []
    c, row, col = hierarchy.shape
    for i in range(row):
        if hierarchy[0, i, 2] > 0 or hierarchy[0, i, 3] > 0:  # 有父轮廓或子轮廓
            pass
        else:
            delete_list.append(i)

    # 根据列表序号删除不符合要求的轮廓
    contours = delet_contours(contours, delete_list)
    temp = np.ones(thresh.shape, dtype=np.uint8) * 255
    result = cv2.drawContours(temp, contours, -1, (0, 0, 0), 2)
    area = []

    # 找到最大的轮廓
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))

    # 填充最大的轮廓
    mask = cv2.drawContours(result, contours, max_idx, 0, cv2.FILLED)
    out_mask = np.ones(mask.shape, dtype=np.uint8) * 255
    out_mask[mask == 255] = 0
    return out_mask



def thresh_grabcut(img):
    old_img = img.copy()
    out_img = img.copy()
    # 增强图像的对比度
    img_bright = cv2.convertScaleAbs(img, alpha=2, beta=0)
    # 生成mask
    img_bright_gray = cv2.cvtColor(img_bright, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img_bright_gray, 30, 255, cv2.THRESH_BINARY_INV)
    mask[mask == 255] = 1
    mask[mask == 0] = 2

    # 生成bgdModel, fgdModlel
    size = (1, 65)
    bgdModel = np.zeros(size, np.float64)
    fgdModel = np.zeros(size, np.float64)
    # 生成rect
    rect = (1, 1, img.shape[1], img.shape[0])

    # cv2.grabCut
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 20, cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    out_img *= mask[:, :, np.newaxis]
    mask1 = np.where((mask == 1), 255, 0).astype("uint8")
    out_mask = re_max_area(mask1)
    return out_mask


def mouse_grabcut(img):
    g_img = GrabCut(img)
    cv2.namedWindow('image')
    # 定义鼠标的回调函数
    cv2.setMouseCallback('image', mouse_event2, g_img)
    while (True):
        cv2.imshow('image', g_img.img_show)
        if g_img.lb_up or g_img.rb_up:
            g_img.lb_up = False
            g_img.rb_up = False
            start = time.process_time()
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)

            rect = (1, 1, g_img.img.shape[1], g_img.img.shape[0])
            print(g_img.mask)
            mask = g_img.mask
            g_img.img_gc = g_img.img.copy()
            cv2.grabCut(g_img.img_gc, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            elapsed = (time.process_time() - start)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')  # 0和2做背景
            g_img.img_gc = g_img.img_gc * mask2[:, :, np.newaxis]  # 使用蒙板来获取前景区域
            cv2.imshow('result', g_img.img_gc)
            mask1 = np.where((mask2 == 1), 0, 255).astype('uint8')
            out_mask = re_max_area(mask1)
        # 按下ESC键退出
        if cv2.waitKey(20) == 27:
            return out_mask





