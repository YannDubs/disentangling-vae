import glob
import os

import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import subprocess
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as F

DIR = os.path.abspath(os.path.dirname(__file__))

from PIL import Image
import os
import sys
from resizeimage import resizeimage

def get_img_size(dataset):
    """Return the correct image size."""
    if dataset in ['mnist', 'fashion_mnist']:
        img_size = (1, 32, 32)
    if dataset in ['chairs', 'dsprites']:
        img_size = (1, 64, 64)
    if dataset == 'celeba':
        img_size = (3, 64, 64)
    return img_size

def get_dataloaders(batch_size, dataset, shuffle=False):
    """A generic data loader"""
    dataset_options = {
        "mnist": get_mnist_dataloaders,
        "fashion_mnist": get_fashion_mnist_dataloaders,
        "dsprites": get_dsprites_dataloader,
        "chairs": get_chairs_dataloader,
        "celeba": get_celeba_dataloader
    }
    if dataset in dataset_options:
        return dataset_options[dataset](batch_size=batch_size, shuffle=shuffle)
    else:
        raise Exception("{} is not valid. Please enter a valid dataset".format(dataset))

def get_mnist_dataloaders(batch_size=128, shuffle=True,
                          path_to_data=os.path.join(DIR, '../data/mnist')):
    """MNIST dataloader with (32, 32) images."""
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = datasets.MNIST(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST(path_to_data, train=False,
                               transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader


def get_fashion_mnist_dataloaders(batch_size=128, shuffle=True,
                                  path_to_data=os.path.join(DIR, '../data/fashion/mnist')):
    """FashionMNIST dataloader with (32, 32) images."""
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = datasets.FashionMNIST(path_to_data, train=True, download=True,
                                       transform=all_transforms)
    test_data = datasets.FashionMNIST(path_to_data, train=False,
                                      transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader


def get_dsprites_dataloader(batch_size=128,path_to_data=os.path.join(DIR, '../data/dsprites-dataset'), shuffle=True):
    """DSprites dataloader."""
    print('in loop  ')
    root = os.path.join(os.path.dirname(DIR), 'data', 'dsprites-dataset')
    if os.path.isdir(root)==False:
        subprocess.call(DIR + '/load_DSprites.sh')
        
    dsprites_data = DSpritesDataset(path_to_data=path_to_data+'/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
                                    transform=transforms.ToTensor())
    dsprites_loader = DataLoader(dsprites_data, batch_size=batch_size, shuffle=shuffle)
    return dsprites_loader, 0


def get_chairs_dataloader(batch_size=128, shuffle=True,
                          path_to_data=os.path.join(DIR, '../data/3DChairs_64/rendered_chairs_64')):
    """Chairs dataloader. Chairs are center cropped and resized to (64, 64)."""
    all_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    root = os.path.join(os.path.dirname(DIR), 'data', '3DChairs')
    if os.path.isdir(root)==False:
        subprocess.call(DIR + '/load_3DChairs.sh')
    root2 = os.path.join(os.path.dirname(DIR), 'data', '3DChairs_64','images_64')
    if not os.listdir(root2):  
        resize_images_chairs()

    path_sub = os.path.join(os.path.dirname(DIR), 'data', '3DChairs_64')
    chairs_data = datasets.ImageFolder(root=path_sub, transform=all_transforms)

    chairs_loader = DataLoader(chairs_data, batch_size=batch_size,
                               shuffle=shuffle)
    return chairs_loader,0

def get_chairs_test_dataloader(batch_size=62, shuffle=True,
                               path_to_data=os.path.join(DIR, '../data/chairs/rendered_chairs_64_test')):
    """There are 62 pictures of each chair, so get batches of data containing
    one chair per batch."""
    all_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    chairs_data = datasets.ImageFolder(root=path_to_data,
                                       transform=all_transforms)
    chairs_loader = DataLoader(chairs_data, batch_size=batch_size,
                               shuffle=shuffle)
    return chairs_loader


def get_celeba_dataloader(batch_size=128, shuffle=True,
                          path_to_data=os.path.join(DIR, '../data', 'celeba','img_align_celeba_64')):
    """CelebA dataloader with (64, 64) images."""
    root = os.path.join(os.path.dirname(DIR), 'data', 'celeba')
    if os.path.isdir(root)==False:
        subprocess.call(DIR + '/load_celebA.sh')

    root2 = os.path.join(os.path.dirname(DIR), 'data', 'celeba','img_align_celeba_64')
    if not os.listdir(root2):
        resize_images_celeba()

    celeba_data = CelebADataset(path_to_data=path_to_data,
                                transform=transforms.ToTensor())
    celeba_loader = DataLoader(celeba_data, batch_size=batch_size,
                               shuffle=shuffle)
    return celeba_loader,0


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
        #print(sample_path)
        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0

class ChairsDataset(Dataset):
    """CelebA dataset with 64 by 64 images."""

    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        #print(path_to_data)
        #self.img_paths = glob.glob(path_to_data + '/images_64/*')[::subsample]
        self.img_paths = glob.glob(path_to_data + '/*')[::subsample]
        self.transform = transform
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        sample = imread(sample_path)
        #sample = self.loader(path)
        #sample = F.to_pil_image(sample)
        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0

def resize_images_celeba():
    directory_in = os.path.join(os.path.dirname(DIR), 'data', 'celeba', 'img_align_celeba')
    directory_out = os.path.join(os.path.dirname(DIR), 'data', 'celeba', 'img_align_celeba_64')
    i=0
    for file_name in os.listdir(directory_in):
        if (i%100)==0:
            print("iteration nr: %d / %d" %(i,len(os.listdir(directory_in))))
        #print("Processing %s" % file_name)
        image_temp = Image.open(os.path.join(directory_in, file_name))
        image_temp = resizeimage.resize_crop(image_temp, [178, 178])
        output = image_temp.resize((64, 64), Image.ANTIALIAS)
    
        output_file_name = os.path.join(directory_out, "64_" + file_name)
        output.save(output_file_name, "JPEG", quality = 95)
        i=i+1

def resize_images_chairs():
    directory_in = os.path.join(os.path.dirname(DIR), 'data', '3DChairs', 'images')
    directory_out = os.path.join(os.path.dirname(DIR), 'data', '3DChairs_64', 'images_64')
    i=0
    for file_name in os.listdir(directory_in):
        if (i%100)==0:
            print("iteration nr: %d / %d" %(i,len(os.listdir(directory_in))))
        #print("Processing %s" % file_name)
        image = Image.open(os.path.join(directory_in, file_name))
        image = resizeimage.resize_crop(image, [400, 400])
        output = image.resize((64, 64), Image.ANTIALIAS)
        
        output_file_name = os.path.join(directory_out, "64_" + file_name[0:-4]+'.jpeg')
        output.save(output_file_name, "JPEG", quality = 95)
        i=i+1

