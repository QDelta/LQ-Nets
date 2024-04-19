import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import re
import os
from lq_layers import LQLinear, LQConv, LQActiv
from resnet import ResNet

def parse_quantization_bits_from_filename(filename):
    match_w = re.search(r'_wq(\d+)', filename)
    match_a = re.search(r'_aq(\d+)', filename)
    
    w_nbits = int(match_w.group(1)) if match_w else None
    a_nbits = int(match_a.group(1)) if match_a else None

    print('weight bits:',w_nbits, 'activation bits:', a_nbits)
    
    return w_nbits, a_nbits

def load_model(model_path, num_layers, num_classes, w_nbits, a_nbits, device):
    model = ResNet(w_nbits=w_nbits, a_nbits=a_nbits)
    model.to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def collect_activations_and_weights(model, dataloader, device):
    activations = {}
    weights = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu().numpy()
        return hook

    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, LQConv)):
            layer.register_forward_hook(get_activation(name))
            if hasattr(layer, 'weight'):
                weights[name] = layer.weight.data.cpu().numpy()

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            _ = model(inputs)

    return activations, weights

def plot_distribution(activations, weights):
    for name in weights:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(weights[name].ravel(), bins=50, alpha=0.75)
        plt.title(f'Weight Distribution: {name}')
        plt.xlabel('Weight Values')
        plt.ylabel('Frequency')

        if name in activations:
            plt.subplot(1, 2, 2)
            plt.hist(activations[name].ravel(), bins=50, alpha=0.75)
            plt.title(f'Activation Distribution: {name}')
            plt.xlabel('Activation Values')
            plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

def analyze_model(model_filename, device='cuda'):
    w_nbits, a_nbits = parse_quantization_bits_from_filename(model_filename)
    
    # based on the known model configuration for CIFAR-10
    num_layers = 20
    num_classes = 10
    
    model = load_model(model_filename, num_layers, num_classes, w_nbits, a_nbits, device)

    test_loader = DataLoader(TEST_SET, batch_size=64, shuffle=False)

    activations, weights = collect_activations_and_weights(model, test_loader, device)
    plot_distribution(activations, weights)

TEST_SET = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

if __name__ == '__main__':
    model_file_path = 'resnet20_cifar.pt'
    analyze_model(model_file_path)