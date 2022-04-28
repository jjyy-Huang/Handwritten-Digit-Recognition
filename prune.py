"""
 @Description: prune.py in Handwritten-Digit-Recognition
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
from torchsummary import summary

def Prune(model=None, mode=None):
    if model is None:
        model = torch.load('./LeNet5.pt').to(DEVICE)

    summary(model)
    if mode == 'global':
        parameters_to_prune = (
                    (model.conv1, 'weight'),
                    (model.conv2, 'weight'),
                    (model.conv3, 'weight'),
                    (model.fc1, 'weight'),
                    (model.fc2, 'weight'))

        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.2)

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

    elif mode == 'part':
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name="weight", amount=0.2)
            elif isinstance(module, torch.nn.Linear):
                prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)


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

    return model
