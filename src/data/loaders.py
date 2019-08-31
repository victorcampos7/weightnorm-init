import os
import numpy as np

import torch

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from copy import copy
from data.regularizers import Cutout


DATASETS = ['mnist', 'cifar10', 'cifar100']
DATA_ROOT = '/tmp/datasets/'
MNIST_PATH = os.path.join(DATA_ROOT, 'mnist')
CIFAR10_PATH = os.path.join(DATA_ROOT, 'cifar10')
CIFAR100_PATH = os.path.join(DATA_ROOT, 'cifar100')


def load_data(args):
    assert args.dataset in DATASETS, "Supported datasets: " + str(DATASETS)
    if args.dataset == 'mnist':
        assert args.val_fraction > 0.
        return mnist(batch_size=args.mini_batch_size, val_fraction=args.val_fraction)
    elif args.dataset == 'cifar10':
        assert args.val_fraction > 0.
        return cifar10(batch_size=args.mini_batch_size, val_fraction=args.val_fraction, cutout=args.cutout)
    elif args.dataset == 'cifar100':
        assert args.val_fraction > 0.
        return cifar100(batch_size=args.mini_batch_size, val_fraction=args.val_fraction, cutout=args.cutout)


def _split_train_val(trainset, val_fraction):
    n_train, n_val = int((1. - val_fraction) * len(trainset)), int(val_fraction * len(trainset))
    train_subset, val_subset = torch.utils.data.random_split(trainset, (n_train, n_val))
    return train_subset, val_subset


def mnist(batch_size, val_fraction=0.1):
    num_classes = 10

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    fulltrainset = torchvision.datasets.MNIST(root=MNIST_PATH, train=True, transform=trans, download=True)
    trainset, valset = _split_train_val(fulltrainset, val_fraction=val_fraction)
    testset = torchvision.datasets.MNIST(root=MNIST_PATH, train=False, transform=trans)

    # Create data loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=0, pin_memory=True)
    validloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False,
                                              num_workers=0, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                             num_workers=0, pin_memory=True)

    return trainloader, validloader, testloader, num_classes


def cifar10(batch_size, val_fraction=0.1, cutout=False):
    num_classes = 10

    transform_train = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

    if cutout:
        transform_train += [Cutout(n_holes=1, length=16)]

    transform_train = transforms.Compose(transform_train)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load data
    fulltrainset = torchvision.datasets.CIFAR10(root=CIFAR10_PATH, train=True, download=True, transform=transform_train)
    trainset, valset = _split_train_val(fulltrainset, val_fraction=val_fraction)
    testset = torchvision.datasets.CIFAR10(root=CIFAR10_PATH, train=False, download=False, transform=transform_test)

    # Remove data augmentation from validation set
    valset.dataset = copy(fulltrainset)
    valset.dataset.transform = transform_test

    # Create data loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=0, pin_memory=True)
    validloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False,
                                              num_workers=0, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                             num_workers=0, pin_memory=True)

    return trainloader, validloader, testloader, num_classes


def cifar100(batch_size, val_fraction=0.1, cutout=False):
    num_classes = 100

    transform_train = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

    if cutout:
        transform_train += [Cutout(n_holes=1, length=16)]

    transform_train = transforms.Compose(transform_train)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load data
    fulltrainset = torchvision.datasets.CIFAR100(root=CIFAR100_PATH, train=True, download=True,
                                                 transform=transform_train)
    trainset, valset = _split_train_val(fulltrainset, val_fraction=val_fraction)
    testset = torchvision.datasets.CIFAR100(root=CIFAR100_PATH, train=False, download=False, transform=transform_test)

    # Remove data augmentation from validation set
    valset.dataset = copy(fulltrainset)
    valset.dataset.transform = transform_test

    # Create data loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=0, pin_memory=True)
    validloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False,
                                              num_workers=0, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                             num_workers=0, pin_memory=True)

    return trainloader, validloader, testloader, num_classes


if __name__ == '__main__':
    batch_size = 8
    trainloader, validloader, testloader, num_classes = cifar10(batch_size=batch_size, val_fraction=0.1, cutout=False)
    print("========== CIFAR-10 ==========")
    print("    Train samples:", len(trainloader) * batch_size)
    print("    Val samples:", len(validloader) * batch_size)
    print("    Test samples:", len(testloader) * batch_size)
    print()
    print("    Train transforms:", trainloader.dataset.dataset.transform)
    print("    Val transforms:", validloader.dataset.dataset.transform)
    print("    Test transforms:", testloader.dataset.transform)
    print()
