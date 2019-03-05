import glob
import os

import numpy as np
from skimage.io import imread

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

DIR = os.path.abspath(os.path.dirname(__file__))


def get_img_size(dataset):
    """Return the correct image size."""
    if dataset in ['mnist', 'fashion_mnist']:
        img_size = (1, 32, 32)
    if dataset in ['chairs', 'dsprites']:
        img_size = (1, 64, 64)
    if dataset == 'celeba':
        img_size = (3, 64, 64)
    return img_size


def get_dataloaders(dataset, shuffle=True, pin_memory=True, batch_size=128, **kwargs):
    """A generic data loader"""
    dataset_options = {
        "mnist": get_mnist_dataloaders,
        "fashion_mnist": get_fashion_mnist_dataloaders,
        "dsprites": get_dsprites_dataloader,
        "chairs": get_chairs_dataloader,
        "celeba": get_celeba_dataloader
    }
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available
    if dataset in dataset_options:
        return dataset_options[dataset](batch_size=batch_size, shuffle=shuffle,
                                        pin_memory=pin_memory, **kwargs)
    else:
        raise Exception("{} is not valid. Please enter a valid dataset".format(dataset))


def get_mnist_dataloaders(path_to_data=os.path.join(DIR, '../data/mnist'), **kwargs):
    """MNIST dataloader with (32, 32) images."""
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = datasets.MNIST(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST(path_to_data, train=False,
                               transform=all_transforms)
    train_loader = DataLoader(train_data, **kwargs)
    test_loader = DataLoader(test_data, **kwargs)
    return train_loader, test_loader


def get_fashion_mnist_dataloaders(path_to_data=os.path.join(DIR, '../data/fashion/mnist'),
                                  **kwargs):
    """FashionMNIST dataloader with (32, 32) images."""
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = datasets.FashionMNIST(path_to_data, train=True, download=True,
                                       transform=all_transforms)
    test_data = datasets.FashionMNIST(path_to_data, train=False,
                                      transform=all_transforms)
    train_loader = DataLoader(train_data, **kwargs)
    test_loader = DataLoader(test_data, **kwargs)
    return train_loader, test_loader


def get_dsprites_dataloader(path_to_data=os.path.join(DIR, '../data/dsprites/dsprites.npy'),
                            **kwargs):
    """DSprites dataloader."""
    dsprites_data = DSpritesDataset(path_to_data,
                                    transform=transforms.ToTensor())
    dsprites_loader = DataLoader(dsprites_data, **kwargs)
    return dsprites_loader, None


def get_chairs_dataloader(path_to_data=os.path.join(DIR, '../data/chairs/rendered_chairs_64'),
                          **kwargs):
    """Chairs dataloader. Chairs are center cropped and resized to (64, 64)."""
    all_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    chairs_data = datasets.ImageFolder(root=path_to_data,
                                       transform=all_transforms)
    chairs_loader = DataLoader(chairs_data, **kwargs)
    return chairs_loader


def get_chairs_test_dataloader(path_to_data=os.path.join(DIR, '../data/chairs/rendered_chairs_64_test'),
                               **kwargs):
    """There are 62 pictures of each chair, so get batches of data containing
    one chair per batch."""
    all_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    chairs_data = datasets.ImageFolder(root=path_to_data,
                                       transform=all_transforms)
    chairs_loader = DataLoader(chairs_data, **kwargs)
    return chairs_loader


def get_celeba_dataloader(path_to_data=os.path.join(DIR, '../data/celeba_64'),
                          **kwargs):
    """CelebA dataloader with (64, 64) images."""
    celeba_data = CelebADataset(path_to_data,
                                transform=transforms.ToTensor())
    celeba_loader = DataLoader(celeba_data, **kwargs)
    return celeba_loader


class DSpritesDataset(Dataset):
    """D Sprites dataset."""

    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        self.imgs = np.load(path_to_data)['imgs'][::subsample]

        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Each image in the dataset has binary values so multiply by 255 to get
        # pixel values
        sample = self.imgs[idx] * 255
        # Add extra dimension to turn shape into (H, W) -> (H, W, C)
        sample = sample.reshape(sample.shape + (1,))

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0


class CelebADataset(Dataset):
    """CelebA dataset with 64 by 64 images."""

    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        self.img_paths = glob.glob(path_to_data + '/*')[::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        sample = imread(sample_path)

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0
