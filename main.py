import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from resnet import ResNet
from time import time
from contextlib import contextmanager
import os
from torch.cuda import max_memory_allocated

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

@contextmanager
def print_and_log(log_filename):
    log_file = open(log_filename, 'w')
    def log(message):
        print(message)
        log_file.write(message + '\n')
        log_file.flush()
    yield log
    log_file.close()

class ApplyTransform(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return self.transform(x), y

    def __len__(self):
        return len(self.dataset)

# normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

TRANSFORM_TRAIN = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    normalize,
])
TRANSFORM_TEST = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

TRAIN_SET = datasets.CIFAR10(root='./data', train=True, download=True, transform=TRANSFORM_TRAIN)
TEST_SET = datasets.CIFAR10(root='./data', train=False, download=True, transform=TRANSFORM_TEST)

def test(model: nn.Module, loader: DataLoader):
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    test_correct = 0
    test_total = 0
    start_time = time()
    torch.cuda.reset_max_memory_allocated(DEVICE)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            test_correct += (outputs.argmax(1) == targets).sum().item()
            test_total += targets.size(0)

    latency = time() - start_time
    max_memory_usage = max_memory_allocated(DEVICE) / (1024 ** 2)

    test_loss /= len(loader)
    test_acc = test_correct / test_total
    return test_loss, test_acc, latency, max_memory_usage

def train(w_nbits, a_nbits, lr=0.1, weight_decay=1e-3,
          optimizer_type='sgd', epochs=200, batch_size=128):

    # import resnet20
    # model = resnet20.ResNetCIFAR(w_nbits=w_nbits, a_nbits=a_nbits).to(DEVICE)
    torch.cuda.reset_max_memory_allocated(DEVICE)
    model = ResNet(w_nbits=w_nbits, a_nbits=a_nbits).to(DEVICE)

    ckpt_base_filename = (
        'resnet20_cifar'
        + ('' if w_nbits is None else f'_wq{w_nbits}')
        + ('' if a_nbits is None else f'_aq{a_nbits}')
        + '.pt'
    )
    log_filename = ckpt_base_filename.replace('.pt', '.txt')
    last_saved_filename = None

    train_loader = DataLoader(TRAIN_SET, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TEST_SET, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    if optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)

    best_test_acc = 0.0
    save_threshold_epoch = min(50, epochs // 3)


    with print_and_log(log_filename) as log:
        log(f'\nQuantization: weight={w_nbits} activation={a_nbits}, Using device: {DEVICE}')

        for epoch in range(epochs):
            log(f'\nEpoch {epoch+1}')
            model.train()

            train_loss = 0
            train_correct = 0
            train_total = 0

            start_time = time()

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

            train_latency = time() - start_time
            max_memory_usage_during_training = max_memory_allocated(DEVICE) / (1024 ** 2)

            log(f'LR: {optimizer.param_groups[0]["lr"]:.4e}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Latency: {train_latency:.2f}s, Memory Usage: {max_memory_usage_during_training:.2f}MB')

            model.eval()
            test_loss, test_acc, test_latency, test_memory_usage = test(model, test_loader)
            log(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Latency: {test_latency:.2f}s, Test Memory Usage: {test_memory_usage:.2f}MB')

            scheduler.step()

            if epoch >= save_threshold_epoch:
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    print('Saving best model ...')
                    if last_saved_filename is not None and last_saved_filename != ckpt_base_filename:
                        os.remove(last_saved_filename)
                    ckpt_filename = f'epoch_{epoch+1}_' + ckpt_base_filename
                    torch.save(model.state_dict(), ckpt_filename)
                    last_saved_filename = ckpt_filename

    print('Saving final epoch model ...')
    torch.save(model.state_dict(), ckpt_base_filename)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wq', type=int, default=None)
    parser.add_argument('--aq', type=int, default=None)
    args = parser.parse_args()
    train(w_nbits=args.wq, a_nbits=args.aq)
