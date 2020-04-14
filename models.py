import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from constants import CLASSES


class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 18, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(18)
        self.pool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(18, 36, 3, bias=False)
        self.bn3 = nn.BatchNorm2d(36)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(36 * 10 * 10, 100)
        self.fc2 = nn.Linear(100, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.bn3(x)
        x = x.view(-1, 36 * 10 * 10)
        x = self.dropout(x)
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def vgg16():
    net = torchvision.models.vgg16()
    net.features[0] = nn.Conv2d(1, 64, 3, 1, padding=1, padding_mode='zeros')
    net.classifier[6] = nn.Linear(in_features=4096, out_features=len(CLASSES), bias=True)
    return net


def resnet18():
    net = torchvision.models.resnet18()
    net.conv1 = nn.Conv2d(1, 64, 7, 2, padding=3, padding_mode='zeros', bias=False)
    net.fc = nn.Linear(in_features=512, out_features=len(CLASSES), bias=True)
    return net