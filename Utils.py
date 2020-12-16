import os
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn


def MnistLoadData(image_size, batch_size, train):
    os.makedirs("images", exist_ok=True)

    if image_size is None:
        transform = transforms.ToTensor()
    else:
        transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=train,
            download=True,
            transform=transform),
        batch_size=batch_size,
        shuffle=True,
    )

    return dataloader


def conv_3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv_1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


