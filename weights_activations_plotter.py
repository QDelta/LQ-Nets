import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import re
import os
from lq_layers import LQConv, LQActiv
from liq_layers import LiQConv, LiQActiv
from resnet import ResNet

def parse_quantization_bits_from_filename(filename):
    match_w = re.search(r'_wq(\d+)', filename)
    match_a = re.search(r'_aq(\d+)', filename)

    w_nbits = int(match_w.group(1)) if match_w else None
    a_nbits = int(match_a.group(1)) if match_a else None
    base_filename = os.path.splitext(os.path.basename(filename))[0]

    contains_linear = 'linear' in base_filename

    print('weight bits:', w_nbits, 'activation bits:', a_nbits, 'is linear:', contains_linear, 'base filename:', base_filename)

    return w_nbits, a_nbits, contains_linear, base_filename

def load_model(model_path, num_layers, num_classes, w_nbits, a_nbits, device):
    model = ResNet(w_nbits=w_nbits, a_nbits=a_nbits)
    model.to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model

def collect_activations_and_quantized_weights(model, dataloader, device,linear = False):
    activations = {}
    quantized_weights = {}

    def get_quantized_weight(name, layer):
        def hook(model, input, output):
            q_weight = layer.lq.apply(layer.conv.weight, layer.basis, False)[0]
            quantized_weights[name] = q_weight.detach().cpu().numpy()
        return hook

    def get_activation(name, layer):
      def hook(model, input, output):
          activations[name] = output.detach().cpu().numpy()
      return hook

    for name, layer in model.named_modules():
        if linear:
            if isinstance(layer, LiQConv):
                layer.register_forward_hook(get_quantized_weight(name, layer))
            if isinstance(layer, LiQActiv):
                layer.register_forward_hook(get_activation(name, layer))
        else:
            if isinstance(layer, LQConv):
                layer.register_forward_hook(get_quantized_weight(name, layer))
            if isinstance(layer, LQActiv):
                layer.register_forward_hook(get_activation(name, layer))

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            _ = model(inputs)

    return activations, quantized_weights

def plot_distributions_side_by_side(activations, quantized_weights, base_filename):
    sorted_weight_keys = sorted(quantized_weights.keys())
    sorted_activation_keys = sorted(activations.keys())

    num_layers = max(len(sorted_weight_keys), len(sorted_activation_keys))
    fig, axes = plt.subplots(num_layers, 2, figsize=(12, 6 * num_layers))  

    output_directory = f"./output/{base_filename}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i, (weight_key, activation_key) in enumerate(zip(sorted_weight_keys, sorted_activation_keys)):
        ax_w = axes[i][0] if num_layers > 1 else axes[0]
        ax_a = axes[i][1] if num_layers > 1 else axes[1]
        ax_w.hist(quantized_weights[weight_key].ravel(), bins=50, alpha=0.75)
        ax_w.set_title(f'Quantized Weight Distribution: {weight_key}')
        ax_w.set_xlabel('Weight Values')
        ax_w.set_ylabel('Frequency')

        ax_a.hist(activations[activation_key].ravel(), bins=50, alpha=0.75)
        ax_a.set_title(f'Activation Distribution: {activation_key}')
        ax_a.set_xlabel('Activation Values')
        ax_a.set_ylabel('Frequency')

        plot_filename = f"{output_directory}/{base_filename}_{weight_key}_and_{activation_key}.png"
        plt.savefig(plot_filename)
        print(f"Saved plot as {plot_filename}")

    plt.close(fig)

def analyze_model(model_filename, device='cuda'):
    w_nbits, a_nbits, linear, base_filename = parse_quantization_bits_from_filename(model_filename)
    num_layers = 20
    num_classes = 10

    model = load_model(model_filename, num_layers, num_classes, w_nbits, a_nbits, device)
    test_loader = DataLoader(TEST_SET, batch_size=64, shuffle=False)

    activations, weights = collect_activations_and_quantized_weights(model, test_loader, device, linear=linear)
    plot_distributions_side_by_side(activations, weights, base_filename)

    
TEST_SET = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

if __name__ == '__main__':
    model_file_path = 'resnet20_cifar.pt'
    analyze_model(model_file_path)