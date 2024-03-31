import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def learned_quantization(nbits: int | None, q_T: int, q_alpha: float):
    if nbits is None:
        def _forward(ctx, w, *args):
            return w, None
    elif nbits == 0:
        def _forward(ctx, w, *args):
            return torch.zeros_like(w, requires_grad=True), None
    else:
        assert nbits <= 8
        bitvecs = np.unpackbits(np.arange(2 ** nbits, dtype=np.uint8).reshape(-1, 1), axis=1)[:,-nbits:]
        # {0, 1} to {-1, 1}
        encodings = torch.from_numpy(bitvecs.astype(np.float32) * 2 - 1)

        def _forward(ctx, w: torch.Tensor, basis: torch.Tensor, training: bool):
            nonlocal encodings
            encodings = encodings.to(basis.device)
            # weight_mask = w != 0

            def quantize(w: torch.Tensor, basis: torch.Tensor, encodings: torch.Tensor):
                qlevels = encodings @ basis
                qlevel_idx = torch.argsort(qlevels)
                qlevels = qlevels[qlevel_idx]
                encodings = encodings[qlevel_idx]

                wq = torch.zeros_like(w, requires_grad=True)
                wb = torch.zeros(w.numel(), nbits, device=w.device)
                need_quantize = torch.ones_like(w, dtype=torch.bool)

                for i in range(2 ** nbits - 1):
                    thres = (qlevels[i] + qlevels[i+1]) / 2
                    mask = (w <= thres) & need_quantize
                    wq[mask] = qlevels[i]
                    wb[mask.view(-1)] = encodings[i]
                    need_quantize[mask] = False
                wq[need_quantize] = qlevels[-1]
                wb[need_quantize.view(-1)] = encodings[-1]

                return wq, wb

            if not training:
                wq, _ = quantize(w, basis, encodings)
            else:
                v = basis
                for _ in range(q_T):
                    wq, wb = quantize(w, v, encodings)
                    v = torch.linalg.solve(wb.T @ wb, wb.T @ w.view(-1))
                basis = q_alpha * basis + (1-q_alpha) * v

            # wq *= weight_mask
            return wq, basis

    class LearnedQuantization(torch.autograd.Function):
        forward = _forward

        @staticmethod
        def backward(ctx, g, _):
            return g, None, None

    return LearnedQuantization

NORM_PPF_0_75 = 0.6745

def weight_init_basis(nbits: int, n: int):
    base = NORM_PPF_0_75 * ((2. / n) ** 0.5) / (2 ** (nbits - 1))
    return torch.tensor([base * (2. ** i) for i in range(nbits)])

def activ_init_basis(nbits: int):
    return torch.tensor([(NORM_PPF_0_75 * 2 / (2 ** nbits - 1)) * (2. ** i) for i in range(nbits)])

class LQLinear(nn.Module):
    def __init__(self, in_features, out_features, nbits=None, q_T=1, q_alpha=0.9):
        super(LQLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

        # Initailization
        m = self.in_features
        n = self.out_features
        self.linear.weight.data.normal_(0, math.sqrt(2. / (m+n)))

        # Learned quantization
        if nbits is not None:
            self.register_buffer('basis', weight_init_basis(nbits, n))
        else:
            self.basis = None
        self.lq = learned_quantization(nbits, q_T, q_alpha)

    def forward(self, x):
        q_weight, new_basis = self.lq.apply(self.linear.weight, self.basis, self.training)
        self.basis = new_basis
        return F.linear(x, q_weight, self.linear.bias)

class LQConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, nbits=None, q_T=1, q_alpha=0.9):
        super(LQConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        # Initialization
        n = self.kernel_size * self.kernel_size * self.out_channels
        m = self.kernel_size * self.kernel_size * self.in_channels
        self.conv.weight.data.normal_(0, math.sqrt(2. / (n+m)))
        
        # Learned quantization
        if nbits is not None:
            self.register_buffer('basis', weight_init_basis(nbits, n))
        else:
            self.basis = None
        self.lq = learned_quantization(nbits, q_T, q_alpha)

    def forward(self, x):
        q_weight, new_basis = self.lq.apply(self.conv.weight, self.basis, self.training)
        self.basis = new_basis
        return F.conv2d(x, q_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
