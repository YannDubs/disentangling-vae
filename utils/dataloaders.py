import glob
import os

import subprocess
import numpy as np
from skimage.io import imread
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as F

DIR = os.path.abspath(os.path.dirname(__file__))

class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img

class ChairsDataset(Dataset):
    """CelebA dataset with 64 by 64 images."""

    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        self.img_paths = glob.glob(path_to_data + '/images/*')[::subsample]
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

def get_chairs_dataloader(batch_size=128,path_to_data=os.path.join(DIR, '../data/3DChairs/images')):
    """Chairs dataloader. Chairs are center cropped and resized to (64, 64)."""
    root = os.path.join(os.path.dirname(DIR), 'data', '3DChairs')
    if os.path.isdir(root)==False:
        subprocess.call(DIR + '/load_3DChairs.sh')
#     image_size = get_img_size('chairs')
    
#     transform = transforms.Compose([
#         transforms.Resize((image_size, image_size)),
#         transforms.ToTensor(),])
#     train_kwargs = {'root':root, 'transform':transform}
    
    all_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()])
    chairs_data = datasets.ImageFolder(root=root, transform=all_transforms)
#     dset = CustomImageFolder
#     train_data = dset(**train_kwargs) 
    chairs_loader = DataLoader(chairs_data, batch_size=batch_size, shuffle=True)
#     data_loader = DataLoader(train_data,
#                              batch_size=batch_size,
#                              shuffle=True)

    return chairs_loader, chairs_loader
    
def get_img_size(dataset):
    """Return the correct image size."""
    if dataset in ['mnist', 'fashion_mnist']:
        img_size = (1, 32, 32)
    if dataset in ['chairs', 'dsprites']:
        img_size = (1, 64, 64)
    if dataset == 'celeba':
        img_size = (3, 64, 64)
    return img_size

def get_dataloaders(batch_size, dataset):
    """A generic data loader"""
    dataset_options = {
        "mnist": get_mnist_dataloaders,
        "fashion_mnist": get_fashion_mnist_dataloaders,
        "dsprites": get_dsprites_dataloader,
        "chairs": get_chairs_dataloader,
        "celeba": get_celeba_dataloader
    }
    if dataset in dataset_options:
        return dataset_options[dataset](batch_size=batch_size)
    else:
        raise Exception("{} is not valid. Please enter a valid dataset".format(dataset))

def get_mnist_dataloaders(batch_size=128,
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
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_fashion_mnist_dataloaders(batch_size=128,
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
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_dsprites_dataloader(batch_size=128,path_to_data=os.path.join(DIR, '../data/dsprites-dataset')):
    """DSprites dataloader."""
    root = os.path.join(os.path.dirname(DIR), 'data', 'dsprites-dataset')
    if os.path.isdir(root)==False:
        subprocess.call(DIR + '/load_DSprites.sh')
        
    data_path = os.path.join(os.path.dirname(DIR), 'data', 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    data = np.load(data_path)
    data = torch.from_numpy(data['imgs']).unsqueeze(1).float()#*255
    train_kwargs = {'data_tensor':data}
    dset = CustomTensorDataset
    
    train_data = dset(**train_kwargs)    
    
    data_loader = DataLoader(train_data,
                             batch_size=batch_size,
                             shuffle=True)

    return data_loader





def get_celeba_dataloader(batch_size=128,path_to_data=os.path.join(DIR, '../data/celeba')):
    """CelebA dataloader with (64, 64) images."""
    root = os.path.join(os.path.dirname(DIR), 'data', 'celeba')
    if os.path.isdir(root)==False:
        subprocess.call(DIR + '/load_celebA.sh')
        
    image_size=get_img_size('celeba')
    root = os.path.join(os.path.dirname(DIR), 'data', 'celeba/')
    transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),])
    train_kwargs = {'root':root, 'transform':transform}
    dset = CustomImageFolder
    print(dset)
    train_data = dset(**train_kwargs)    
    print(train_data)
    data_loader = DataLoader(train_data,
                             batch_size=batch_size,
                             shuffle=True)
    print('hoi')
    print(data_loader)
    return data_loader


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




