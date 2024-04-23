import torch
from torch import nn, autograd
import torch.nn.functional as F

class LinearQuantization(autograd.Function):
    @staticmethod
    def forward(ctx, w, nbits):
        if nbits is None:
            return w
        assert 0 < nbits <= 8
        w_min = torch.min(w)
        w_max = torch.max(w)
        alpha = w_max - w_min
        beta = w_min
        ws = (w - beta) / alpha
        step = 2 ** nbits - 1
        R = torch.round(step * ws) / step
        wq = R * alpha + beta
        return wq

    @staticmethod
    def backward(ctx, g):
        return g, None

class LiQConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, nbits=None):
        super(LiQConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.nbits = nbits

    def forward(self, x):
        q_weight = LinearQuantization.apply(self.conv.weight, self.nbits)
        return F.conv2d(x, q_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)

class LiQActiv(nn.Module):
    def __init__(self, nbits=None):
        super(LiQActiv, self).__init__()
        self.nbits = nbits

    def forward(self, x):
        return LinearQuantization.apply(x, self.nbits)
