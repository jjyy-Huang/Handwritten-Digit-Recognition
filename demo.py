"""
 @Description: demo.py in Handwritten-Digit-Recognition
 @Author: Jerry Huang
 @Date: 3/24/22 8:35 PM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import DEVICE, EPOCHS, NUM_WORKERS, BATCH_SIZE, INFER_MODE
from torch.cuda.amp import autocast
from model import lenet5, myNet
from prune import Prune
import time


def loadData():
    train_loader = DataLoader(
        datasets.MNIST('data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1037,), (0.3081,))
                           ])),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    test_loader = torch.utils.data.DataLoader(
                        datasets.MNIST('data', train=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1037,), (0.3081,))
                        ])),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    return train_loader, test_loader

def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # if (batch_idx + 1) % 30 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, mode='float32'):
    if mode == 'float16':
        model.half()
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with autocast():
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum')
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    # print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \n".format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)
    # ))
    return 100. * correct / len(test_loader.dataset)


if __name__ == '__main__':
    train_loader, test_loader = loadData()

    model = torch.load('./LeNet5.pt').to(DEVICE)
    start = time.time()
    acc = test(model, DEVICE, test_loader, mode=INFER_MODE)
    end = time.time()
    print("Inference cost: {:.5}s".format(end - start))
    print("Before pruning, accuracy is {:.4f}%".format(
        acc
    ))
    model = Prune(model, mode='part')
    start = time.time()
    acc = test(model, DEVICE, test_loader, mode=INFER_MODE)
    end = time.time()
    print("Inference cost: {:.5}s".format(end - start))
    print("After pruning, accuracy is {:.4f}%".format(
        acc
    ))


'''
    acc_aver = 0
    model = myNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters())
    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer)
        acc_aver += test(model, DEVICE, test_loader)

    print("myNet average accuracy is {:.5f}%".format(
        acc_aver/EPOCHS
    ))
    torch.save(model, './myNet.pt')
    
    acc_aver = 0
    model = lenet5().to(DEVICE)
    optimizer = optim.Adam(model.parameters())
    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer)
        acc_aver += test(model, DEVICE, test_loader)

    print("Lenet5 average accuracy is {:.5f}%".format(
        acc_aver/EPOCHS
    ))
    torch.save(model, './LeNet5.pt')
    
    '''

