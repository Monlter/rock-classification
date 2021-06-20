import torch.nn as nn
from torchvision import models


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.last_channel, 7),
        )

    def forward(self, x):
        out = self.model(x)
        return out


def mobilenet():
    return MobileNet()

