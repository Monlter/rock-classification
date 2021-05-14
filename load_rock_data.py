from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rock_transforms import train_transform, test_transform
import pandas as pd
import numpy as np
import glob
import random
import PIL.Image as Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# label_name = [
#     "黑色煤",
#     "灰黑色泥岩",
#     "灰色泥质粉砂岩",
#     "灰色细砂岩",
#     "浅灰色细砂岩",
#     "深灰色粉砂质泥岩",
#     "深灰色泥岩"
# ]


# data = pd.read_csv("data/rock_dataset/rock_label_1.csv", encoding="GBK")
# data = np.array(data)
# for row in data:
#     row[1] = label_name.index(row[1])


train_list = glob.glob("./data/rock_dataset/rock_trans/1/TRAIN/*/*.jpg")
test_list = glob.glob("./data/rock_dataset/rock_trans/1/TEST/*/*.jpg")
eval_list = glob.glob("./data/rock_dataset/rock_trans/1/EVAL/*/*.jpg")

# random.shuffle(img_list)
# imgs_len = len(img_list)
# test_len = imgs_len * 0.2
# test_len = int(test_len)
# test_list = img_list[0:test_len]
# train_list = img_list[test_len:imgs_len]


class RockData(Dataset):
    def __init__(self, image_list, transform=None):
        super(RockData, self).__init__()
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        img_label = img_path.split("/")[-2]
        img_label = int(img_label)
        # img_name = img_name.split(".")[-2]
        # img_name = img_name.split("-")[-2]
        # img_name = int(img_name)
        # col1 = data[:, 0]
        # col1 = col1.tolist()
        # col2 = data[:, 1]
        # label_index = col1.index(img_name)
        # img_label = col2[label_index]
        image = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, img_label


train_set = RockData(train_list, transform=train_transform)
test_set = RockData(test_list, transform=test_transform)
eval_set = RockData(eval_list, transform=test_transform)
print("length of train_set:", len(train_set))
print("length of eval_set:", len(eval_set))
print("length of test_set:", len(test_set))
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
eval_loader = DataLoader(eval_set, batch_size=32, shuffle=False)
print("length of train_loader:", len(train_loader))
print("length of eval_loader:", len(eval_loader))
print("length of test_loader:", len(test_loader))
