import torch
from load_rock_data import train_loader, test_loader, eval_loader
from vgg16 import vgg16
from pre_resnet import resnet50, resnet101
from pre_vgg import pytorch_vgg16
from pre_mobilenet import mobilenet
from pre_shufflenet import shufflenet
# from torchvision.models import vgg16
import torch.nn as nn
import torch.optim as optim
import os
import tensorboardX
from tqdm import tqdm, trange
import torchvision
import matplotlib.pyplot as plt
import numpy as np


label_name = [
    "黑色煤",
    "灰黑色泥岩",
    "灰色泥质粉砂岩",
    "灰色细砂岩",
    "浅灰色细砂岩",
    "深灰色粉砂质泥岩",
    "深灰色泥岩"
]

if not os.path.exists("log"):
    os.mkdir("log")
writer = tensorboardX.SummaryWriter("log")

lr = 0.01
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = resnet50()
net_name = "resnet"
# net.to(device)
optimizer = optim.SGD(net.parameters(), lr=lr)
# lr exponentia damping
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# training


def train():
    step_n = 0
    net.to(device)
    for epoch in trange(50):
        net.train()
        for i, d in enumerate(train_loader):
            optimizer.zero_grad()
            # print("epoch is:", epoch, end=';')
            # print("step is:", i)
            inputs, labels = d
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            running_loss = loss.item()
            train_correct = (predicted == labels).sum().item()
            train_correct = train_correct / labels.size(0)
            # print("training loss is:%.3f" % running_loss)
            # print("train correct is:%.3f" % train_correct)
            writer.add_scalar("train loss", running_loss, global_step=step_n)
            writer.add_scalar("train correct", train_correct, global_step=step_n)
            step_n += 1
            running_loss = 0
            train_correct = 0
            loss.backward()
            optimizer.step()
        # 保存结果 every epoch
        if not os.path.exists(net_name + "_models"):
            os.mkdir(net_name + "_models")
        model_path = net_name + "_models/{}.pth"
        torch.save(net.state_dict(), model_path.format(epoch + 1))
        # 更新学习率
        # scheduler.step()

        # eval
        total = 0
        eval_correct = 0
        mean_correct = 0
        net.eval()
        for i, data in enumerate(eval_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            eval_correct += (predicted == labels).sum().item()
            total += labels.size(0)
            mean_correct = eval_correct * 100 / total
        # print("epoch is:", epoch)
        # print("eval correct is:%d %%" % mean_correct)
        writer.add_scalar("eval correct", mean_correct, epoch)

    print("training finished!")

def test():
# testing
#     net.load_state_dict(torch.load("./resnet50_models/50.pth"))
#     test_data = iter(test_loader)
#     inputs, labels = test_data.next()
    # inputs, labels = inputs.to(device), labels.to(device)
    # def imshow(img):
        # img = img / 2 + 0.5     # unnormalize
        # npimg = img.numpy()
        # plt.imshow(np.transpose(npimg, (1, 2, 0)))
        # plt.show()
    # images = inputs.numpy()
    # imshow(torchvision.utils.make_grid(inputs))
    # print('GroundTruth: ', ' '.join('%5s' % label_name[labels[j]] for j in range(4)))

    # outputs = net(inputs)
    # _, predicted = torch.max(outputs.data, 1)
    # print('Predicted: ', ' '.join('%5s' % label_name[predicted[j]] for j in range(4)))
    # acc = (labels == predicted).sum().item()
    # print('Accuracy on the images is: %d%%' % (100 * acc/len(inputs)))
    # print("GroundTruth:")
    # for label_num in labels:
    #     print(label_name[label_num], end=',')
    # print('\r')
    # print("Predicted:")
    # for p in predicted:
    #     print(label_name[p], end=',')
    # print('\r')

    # 整个测试集上
    net.to(device)
    total = 0
    test_correct = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
        print("Accuracy of the network on the test image is:%d %%" % (test_correct * 100 / total))


train()
test()
