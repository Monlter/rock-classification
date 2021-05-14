import glob
import numpy as np
import pandas as pd
import os
import shutil

label_name = [
    "黑色煤",
    "灰黑色泥岩",
    "灰色泥质粉砂岩",
    "灰色细砂岩",
    "浅灰色细砂岩",
    "深灰色粉砂质泥岩",
    "深灰色泥岩"
]


data = pd.read_csv("data/rock_dataset/rock_label.csv", encoding="GBK")
data = np.array(data)
for row in data:
    row[1] = label_name.index(row[1])

im_list = glob.glob("data/rock_dataset/Rock_trans/1/*.jpg")
file_path = "data/rock_dataset/rock_trans/1/TRAIN"

for index in range(len(im_list)):
    img_path = im_list[index]
    img_name = img_path.split("/")[-1]
    img_name = img_name.split(".")[-2]
    # img_name = img_name.split("-")[-2]
    img_name = int(img_name)
    col1 = data[:, 0]
    col1 = col1.tolist()
    col2 = data[:, 1]
    label_index = col1.index(img_name)
    img_label = col2[label_index]
    img_file = os.path.join(file_path, str(img_label))
    if not os.path.exists(img_file):
        os.mkdir(img_file)
    shutil.move(img_path, img_file)
