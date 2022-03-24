"""
 @Description: model.py in Handwritten-Digit-Recognition
 @Author: Jerry Huang
 @Date: 3/24/22 8:41 PM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class lenet5(nn.Module):
    def __init__(self):
        super(lenet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        in_size = x.size(0)
        out = self.relu(self.mp(self.conv1(x)))
        out = self.relu(self.mp(self.conv2(out)))
        out = self.relu(self.conv3(out))
        out = out.view(in_size, -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return self.logsoftmax(out)

class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.mp    = nn.MaxPool2d(2)
        self.relu  = nn.ReLU()
        self.fc1   = nn.Linear(32*6*6, 2048)
        self.fc2   = nn.Linear(2048, 10)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        in_size = x.size(0)
        out = self.relu(self.mp(self.conv1(x)))
        out = self.relu(self.mp(self.conv2(out)))
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out
