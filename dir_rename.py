import os
import glob

listdir = glob.glob("./data/rock_dataset/rock_trans/1/*.jpg")
path = 'data/rock_dataset/rock_trans/1/'
for di in listdir:
    img_name = di.split("/")[-1].split(".")[-2]
    os.rename(os.path.join(path, img_name + '.jpg'),
              os.path.join(path, img_name + '-0' + '.jpg'))
