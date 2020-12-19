import torch.nn as nn
import torch.optim as optim
from Models.Resnet_Model.BasicBlock import BasicBlock
from Models.Resnet_Model.Bottle_neck import Bottle_neck
from Utils import conv_1x1


class ResNet(nn.Module):
    def __init__(self, input_channel, layers, num_classes=10, zero_init_residual=True):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.inputLayer = nn.Sequential(nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.inputLayer2 = nn.Sequential(self.make_layer(Bottle_neck, 64, layers[0]),
                                         self.make_layer(Bottle_neck, 128, layers[1], stride=2),
                                         self.make_layer(Bottle_neck, 256, layers[2], stride=2),
                                         self.make_layer(Bottle_neck, 512, layers[3], stride=2))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottle_neck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottle_neck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.cross_entropy = nn.CrossEntropyLoss()

    def make_layer(self, block, planes, blocks, stride=1):
        down_sample = None

        if stride != 1 or self.in_planes != planes * block.expansion:
            down_sample = nn.Sequential(
                conv_1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, down_sample))

        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        final_layers = nn.Sequential(*layers)
        return final_layers

    def forward(self, x):
        x = self.inputLayer(x)
        x = self.inputLayer2(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def Learn(self, inputs, labels):
        predict = self.forward(inputs)
        loss = self.cross_entropy(predict, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

