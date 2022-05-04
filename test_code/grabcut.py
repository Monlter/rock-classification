
"""
 @Time    : 2021/4/23 23:43
 @Author  : Monlter
 @FileName: grabcut.py
 @Software: PyCharm
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../data/2/350.jpg')
img = cv2.convertScaleAbs(img, alpha=2, beta=0)
OLD_IMG = img.copy()
mask = np.zeros(img.shape[:2], np.uint8)
SIZE = (1, 65)
bgdModle = np.zeros(SIZE, np.float64)
fgdModle = np.zeros(SIZE, np.float64)
rect = (5, 5, img.shape[1]-5, img.shape[0]-5)
cv2.grabCut(img, mask, rect, bgdModle, fgdModle, 100, cv2.GC_INIT_WITH_RECT)

mask1 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img1 = img * mask1[:, :, np.newaxis]
mask2 = np.where((mask == 2), 0, 1).astype('uint8')
img2 = img * mask2[:, :, np.newaxis]
mask3 = np.where((mask == 1), 0, 1).astype('uint8')
img3 = img * mask3[:, :, np.newaxis]

plt.subplot(221), plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title("grabcut"), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(cv2.cvtColor(OLD_IMG, cv2.COLOR_BGR2RGB))
plt.title("original"), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title("maybe"), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
plt.title("must"), plt.xticks([]), plt.yticks([])
plt.show()