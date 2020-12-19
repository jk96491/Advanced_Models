import torch.nn as nn
from Utils import conv_3x3
from Utils import conv_1x1


class Bottle_neck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, down_sample=None):
        super(Bottle_neck, self).__init__()

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # 1 X 1
        self.Layer1 = nn.Sequential(conv_1x1(in_planes, planes),
                                    self.bn1,
                                    self.relu)
        # 3 X 3
        self.Layer2 = nn.Sequential(conv_3x3(planes, planes, stride),
                                    self.bn2,
                                    self.relu)
        # 1 X 1
        self.Layer3 = nn.Sequential(conv_1x1(planes, planes * self.expansion),
                                    self.bn3)

        self.down_sample = down_sample
        self.stride = stride

    def forward(self, inputs):
        identity = inputs

        out = self.Layer1(inputs)
        out = self.Layer2(out)
        out = self.Layer3(out)

        if self.down_sample is not None:
            identity = self.down_sample(inputs)

        out += identity
        out = self.relu(out)

        return out