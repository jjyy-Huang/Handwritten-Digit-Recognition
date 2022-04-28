"""
 @Description: try.py in Handwritten-Digit-Recognition
 @Author: Jerry Huang
 @Date: 4/27/22 8:35 PM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import DEVICE,EPOCHS,NUM_WORKERS,BATCH_SIZE
from model import lenet5, myNet
import torch.nn.utils.prune as prune

model = torch.load('./LeNet5.pt').to(DEVICE)

# 首先打印初始化模型的状态字典
print(model.state_dict().keys())
print('*'*50)

# 构建参数集合, 决定哪些层, 哪些参数集合参与剪枝
parameters_to_prune = (
            (model.conv1, 'weight'),
            (model.conv2, 'weight'),
            (model.conv3, 'weight'),
            (model.fc1, 'weight'),
            (model.fc2, 'weight'))

# 调用prune中的全局剪枝函数global_unstructured执行剪枝操作, 此处针对整体模型中的20%参数量进行剪枝
prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.4)

# 最后打印剪枝后的模型的状态字典
print(model.state_dict().keys())

print(
    "Sparsity in conv1.weight: {:.2f}%".format(
    100. * float(torch.sum(model.conv1.weight == 0))
    / float(model.conv1.weight.nelement())
    ))

print(
    "Sparsity in conv2.weight: {:.2f}%".format(
    100. * float(torch.sum(model.conv2.weight == 0))
    / float(model.conv2.weight.nelement())
    ))

print(
    "Sparsity in conv3.weight: {:.2f}%".format(
    100. * float(torch.sum(model.conv3.weight == 0))
    / float(model.conv3.weight.nelement())
    ))


print(
    "Sparsity in fc1.weight: {:.2f}%".format(
    100. * float(torch.sum(model.fc1.weight == 0))
    / float(model.fc1.weight.nelement())
    ))

print(
    "Sparsity in fc2.weight: {:.2f}%".format(
    100. * float(torch.sum(model.fc2.weight == 0))
    / float(model.fc2.weight.nelement())
    ))

print(
    "Global sparsity: {:.2f}%".format(
    100. * float(torch.sum(model.conv1.weight == 0)
               + torch.sum(model.conv2.weight == 0)
               + torch.sum(model.conv3.weight == 0)
               + torch.sum(model.fc1.weight == 0)
               + torch.sum(model.fc2.weight == 0))
         / float(model.conv1.weight.nelement()
               + model.conv2.weight.nelement()
               + model.conv3.weight.nelement()
               + model.fc1.weight.nelement()
               + model.fc2.weight.nelement())
    ))

prune.remove(model.conv1, 'weight')

print(model.state_dict().keys())
pass
# 当采用全局剪枝策略的时候(假定20%比例参数参与剪枝),
# 仅保证模型总体参数量的20%被剪枝掉,
# 具体到每一层的情况则由模型的具体参数分布情况来定.
