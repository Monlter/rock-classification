import torch.nn as nn
from torchvision import models


class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features, 10)

    def forward(self, x):
        out = self.model(x)
        return out
class resnet50(nn.Module):
    def __init__(self):
        super(resnet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features, 7)

    def forward(self, x):
        out = self.model(x)
        return out

class resnet101(nn.Module):
    def __init__(self):
        super(resnet101, self).__init__()
        self.model = models.resnet101(pretrained=True)
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features, 7)

    def forward(self, x):
        out = self.model(x)
        return out

def pytorch_resnet18():
    return resnet18()

def pytorch_resnet50():
    return resnet50()

def pytorch_resnet101():
    return resnet101()
