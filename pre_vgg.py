import torch.nn as nn
from torchvision import models


class vgg16(nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 7),
        )

    def forward(self, x):
        out = self.model(x)
        return out

def pytorch_vgg16():
    return vgg16()
