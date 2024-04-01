import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from lq_layers import *

__all__ = ['ResNet']

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, nonlinearity='relu')

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, w_nbits=None, a_nbits=None):
        super(BasicBlock, self).__init__()
        self.conv1 = LQConv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, nbits=w_nbits)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activ1 = nn.Sequential(nn.ReLU(True), LQActiv(nbits=a_nbits))
        self.conv2 = LQConv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, nbits=w_nbits)
        self.bn2 = nn.BatchNorm2d(planes)
        self.activ2 = nn.Sequential(nn.ReLU(True), LQActiv(nbits=a_nbits))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))

    def forward(self, x):
        out = self.activ1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activ2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_layers=20, num_classes=10, w_nbits=None, a_nbits=None):
        super(ResNet, self).__init__()
        self.in_planes = 16

        num_blocks = (num_layers - 2) // 6

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, num_blocks, stride=1, w_nbits=w_nbits, a_nbits=a_nbits)
        self.layer2 = self._make_layer(32, num_blocks, stride=2, w_nbits=w_nbits, a_nbits=a_nbits)
        self.layer3 = self._make_layer(64, num_blocks, stride=2, w_nbits=w_nbits, a_nbits=a_nbits)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, planes, num_blocks, stride, w_nbits, a_nbits):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride, w_nbits, a_nbits))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
