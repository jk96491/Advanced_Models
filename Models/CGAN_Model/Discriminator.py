import torch
import torch.nn as nn
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, image_shape):
        super(Discriminator, self).__init__()
        self.image_shape = image_shape

        self.Layer1_1 = nn.Linear(int(np.prod(self.image_shape)), 256)
        self.Layer1_2 = nn.Linear(10, 256)

        self.Layer2 = nn.Sequential(nn.Linear(512, 512),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Linear(512, 256),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Linear(256, 1),
                                   nn.Sigmoid())

    def forward(self, image, labels):
        image_flatten = image.view(image.size(0), -1)
        x1 = self.Layer1_1(image_flatten)
        x2 = self.Layer1_2(labels)

        validity = torch.cat([x1, x2], 1)

        validity = self.Layer2(validity)

        return validity

