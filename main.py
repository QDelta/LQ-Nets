import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import (random_split, DataLoader)
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from resnet20 import ResNetCIFAR

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRANSFORM_TRAIN = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
TRANSFORM_TEST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRAIN_SET, VAL_SET = random_split(datasets.CIFAR10(root='./data', train=True, download=True), [0.9, 0.1])
TRAIN_SET.dataset.transform = TRANSFORM_TRAIN
VAL_SET.dataset.transform = TRANSFORM_TEST
TEST_SET = datasets.CIFAR10(root='./data', train=False, download=True, transform=TRANSFORM_TEST)

def test(model: nn.Module, loader: DataLoader):
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            test_correct += (outputs.argmax(1) == targets).sum().item()
            test_total += targets.size(0)

    test_loss /= len(loader)
    test_acc = test_correct / test_total
    return test_loss, test_acc

def train(nbits, finetune=False, load_ckpt=False, epochs=200, batch_size=128):
    print(f'\nQuantization: {nbits}, Using device: {DEVICE}')

    model = ResNetCIFAR(nbits=nbits).to(DEVICE)

    ckpt_filename = f'resnet20_cifar{"_q" + str(nbits) if nbits is not None else ""}{"_ft" if finetune else ""}.pt'
    if load_ckpt:
        model.load_state_dict(torch.load(ckpt_filename))
    elif finetune:
        model_state = model.state_dict()
        model_state.update(torch.load('resnet20_cifar_pretrained.pt'))
        model.load_state_dict(model_state)

    train_loader = DataLoader(TRAIN_SET, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(VAL_SET, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TEST_SET, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[82, 123], gamma=0.1)

    best_val_acc = 0.0

    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}')
        model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0
        for inputs, targets in tqdm(train_loader, desc='Train', unit='batch', ascii=True, dynamic_ncols=True):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == targets).sum().item()
            train_total += targets.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        print(f'LR: {optimizer.param_groups[0]["lr"]:.4e}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

        model.eval()
        val_loss, val_acc = test(model, val_loader)
        scheduler.step()
        print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print('Saving ...')
            torch.save(model.state_dict(), ckpt_filename)

        test_loss, test_acc = test(model, test_loader)
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nbits', type=int, default=2)
    parser.add_argument('--finetune', action='store_true')
    args = parser.parse_args()
    train(nbits=args.nbits, finetune=args.finetune)
