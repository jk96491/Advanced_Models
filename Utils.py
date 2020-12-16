import os
import torch
from torchvision import datasets
import torchvision.transforms as transforms


def MnistLoadData(image_size, batch_size):
    os.makedirs("images", exist_ok=True)

    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    return dataloader