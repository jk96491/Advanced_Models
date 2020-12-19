import torch.nn as nn
from Models.SAGAN_Model.Attention_Layer import self_attention

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.noise_dim = args.noise_dim
        self.channels = args.channels
        self.noise_filter = args.noise_filter_genenrator

        self.layer1 = nn.Sequential(nn.ConvTranspose2d(self.noise_dim, self.noise_filter * 8, 4, 1, 0, bias=False),
                                    nn.BatchNorm2d(self.noise_filter * 8),
                                    nn.ReLU(True))

        self.layer2 = nn.Sequential(nn.ConvTranspose2d(self.noise_filter * 8, self.noise_filter * 4, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(self.noise_filter * 4),
                                    nn.ReLU(True))

        self.layer3 = nn.Sequential(nn.ConvTranspose2d(self.noise_filter * 4, self.noise_filter * 2, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(self.noise_filter * 2),
                                    nn.ReLU(True))

        self.layer4 = nn.Sequential(nn.ConvTranspose2d(self.noise_filter * 2, self.noise_filter, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(self.noise_filter),
                                    nn.ReLU(True))

        self.layer5 = nn.Sequential(nn.ConvTranspose2d(self.noise_filter, self.channels, 4, 2, 1, bias=False),
                                    nn.Tanh())

        self.self_attention_layer = self_attention(self.noise_filter)

    def forward(self, noise):
        x = self.layer1(noise)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x, attention_score = self.self_attention_layer(x)

        latent = self.layer5(x)

        return latent

