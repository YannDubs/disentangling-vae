"""
Dataloading module.

soem code has been taken from
https://github.com/Schlumberger/joint-vae/blob/master/utils/dataloaders.py#L1
"""
import sys
import os

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

DIR = os.path.abspath(os.path.dirname(__file__))


class MNISTAE(datasets.MNIST):
    """MNIST dataset for autoencoder training."""

    def __getitem__(self, index):
        img, target = super(MNISTAE, self).__getitem__(index)
        return img, img


class FashionMNISTAE(datasets.FashionMNIST):
    """MNIST dataset for autoencoder training."""

    def __getitem__(self, index):
        img, target = super(FashionMNISTAE, self).__getitem__(index)
        return img, img


def get_mnist_dataloaders(batch_size=128,
                          path_to_data=os.path.join(DIR, '../data/mnist'),
                          is_autoencode=True):
    """MNIST dataloader with (32, 32) images."""
    img_size = (1, 32, 32)
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    Mnist = MNISTAE if is_autoencode else datasets.MNIST
    train_data = Mnist(path_to_data, train=True, download=True,
                       transform=all_transforms)
    test_data = Mnist(path_to_data, train=False,
                      transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, img_size


def get_fashion_mnist_dataloaders(batch_size=128,
                                  path_to_data=os.path.join(DIR, '../data/fashion-mnist'),
                                  is_autoencode=True):
    """FashionMNIST dataloader with (32, 32) images."""
    img_size = (1, 32, 32)
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    FasionMnist = FashionMNISTAE if is_autoencode else datasets.FashionMNIST
    train_data = FasionMnist(path_to_data, train=True, download=True,
                             transform=all_transforms)
    test_data = FasionMnist(path_to_data, train=False,
                            transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, img_size
