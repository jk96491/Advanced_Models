import torch.nn as nn
from Utils import conv_3x3


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, down_sample=None):
        super(BasicBlock, self).__init__()
        self.conv_layer1 = nn.Sequential(conv_3x3(in_planes, planes, stride),
                                         nn.BatchNorm2d(planes),
                                         nn.ReLU())
        self.conv_layer2 = nn.Sequential(conv_3x3(planes, planes, stride),
                                         nn.BatchNorm2d(planes))

        self.down_sample = down_sample
        self.stride = stride

    def forward(self, inputs):
        identity = inputs

        out = self.conv_layer1(inputs)
        out = self.conv_layer2(out)

        if self.downsample is not None:
            identity = self.downsample(inputs)

        out += identity
        out = self.relu(out)

        return out
