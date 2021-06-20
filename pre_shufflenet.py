import torch.nn as nn
from torchvision import models


class ShuffleNet(nn.Module):
    def __init__(self):
        super(ShuffleNet, self).__init__()
        self.model = models.shufflenet_v2_x1_0(pretrained=True)
        self.fc = nn.Linear(self.model._stage_out_channels[-1], 7)

    def forward(self, x):
        out = self.model(x)
        return out


def shufflenet():
    return ShuffleNet()

