import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, image_shape, args):
        super(Generator, self).__init__()
        self.image_shape = image_shape

        self.model = nn.Sequential(self.block(args.latent_dim, 128),
                                   self.block(128, 256),
                                   self.block(256, 512),
                                   self.block(512, 1024),
                                   nn.Linear(1024, int(np.prod(self.image_shape))),
                                   nn.Tanh())

    def forward(self, latent):
        latent = self.model(latent)
        latent = latent.view(latent.size(0), *self.image_shape)
        return latent

    def block(self, input_feature, output_feature):
        layers = nn.Sequential(nn.Linear(input_feature, output_feature),
                               nn.LeakyReLU(0.2, inplace=True))
        return layers
