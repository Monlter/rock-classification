import torch
import numpy as np
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import glob
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True
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


data = pd.read_csv("data/rock_dataset/rock_label_1.csv", encoding="GBK")
data = np.array(data)
for row in data:
    row[1] = label_name.index(row[1])

def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C x H x W  ---> H x W x C

    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy() * 255

    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()

    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

    return img


listdir = glob.glob("data/rock_dataset/rock_trans/1/*.jpg")
path = 'data/rock_dataset/rock_trans/1'
# img = Image.open('data/rock_dataset/rock_trans/1/1-0.jpg')

ToTensor_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
    # transforms.RandomGrayscale(0.2),
    # transforms.RandomAffine(90),
    transforms.RandomCrop(1024),
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                      (0.2023, 0.1994, 0.2010)),
])
for di in listdir:
    img_name = di.split("/")[-1]
    img_name = img_name.split(".")[-2]
    img_name = img_name.split("-")[-2]
    col1 = data[:, 0]
    col1 = col1.tolist()
    col2 = data[:, 1]
    label_index = col1.index(int(img_name))
    img_label = col2[label_index]
    for i in range(0, 5):
        img = Image.open(di)
        img_tensor = ToTensor_transform(img)
        img = transform_convert(img_tensor, ToTensor_transform)
        save_path = os.path.join(path, str(img_label) + '/')
        img_path = os.path.join(path, str(img_label) + '/' +str(img_name) + '-' + str(i+1) + '.jpg')
        # img_path = os.path.join(path, str(img_label) + '/' +str(img_name) + '-0.jpg')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        img.save(img_path)
        # plt.axis("off")
        # plt.imshow(img)
        # plt.savefig(os.path.join(path, '1-' + str(i+1) + '.jpg'))
