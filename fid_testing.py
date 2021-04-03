# libraries needed for FID
import torchvision.datasets as datasets
import torch.utils.data as data

import torch.nn as nn

from torchvision.transforms import ToTensor

import numpy as np
import torch
import torchvision.transforms as TF
from scipy import linalg

import torchvision.models as models
import torchvision.transforms as transforms

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler

from utils.datasets import get_dataloaders, get_img_size, DATASETS
from torch.utils.data import Dataset, TensorDataset, DataLoader

from tqdm import tqdm

import utils.inception

INCEPTION_V3 = utils.inception.get_inception_v3()

class CustomTensorDataset(Dataset):
# Tensor Dataset with support for Transforms
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)
    
class NoneTransform(object):   
    def __call__(self, image):       
        return image

# TODO: get cuda working
def _get_activations(dataloader, length, model, batch_size, dims, device='cpu' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    if batch_size > length:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = length

    pred_arr = np.empty((length))

    start_idx = 0

    for inputs, labels in (dataloader):
        batch = inputs.to(device)
        with torch.no_grad():
            pred = model(batch)[0]

        pred = torch.flatten(pred, start_dim=0)
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]
    
    return pred_arr

def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # Calculation of Frechet Distance.
    # Params:
    # mu1   : Numpy array containing the activations of a layer of the
    #         model for generated samples.
    # mu2   : The sample mean over activations, precalculated on an
    #         representative data set.
    # sigma1: The covariance matrix over activations for generated samples.
    # sigma2: The covariance matrix over activations, precalculated on an
    #         representative data set.
    # Returns:
    #       : The Frechet Distance.

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape,               'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape,         'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def _calculate_activation_statistics(dataloader, length, model, batch_size=128, dims=2048):
    # Calculation of the statistics used by the FID.
    # Params:
    # length      : Number of samples
    # model       : Instance of model
    # batch_size  : The images numpy array is split into batches with
    #               batch size batch_size. A reasonable batch size
    #               depends on the hardware.
    # dims        : Dimensionality of features returned by model
    # device      : Device to run calculations
    # Returns:
    # mu    : The mean over samples of the activations of the pool_3 layer of
    #         the model.
    # sigma : The covariance matrix of the activations of the pool_3 layer of
    #         the model.
        
    act = _get_activations(dataloader, length, model, batch_size, dims)
    print("Sucessfully got InceptionV3 activations")
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def get_fid_value(dataloader, vae_model, batch_size = 128):
    # Calculate FID value
    # Params:
    # dataloader : data to test on
    # vae_model : model to evaluate
    # batch_size : batch size when evaluating model

    length = len(dataloader.dataset)
    model = INCEPTION_V3

    # calculated reconstructed data using VAE
    vae_output = []
    vae_label = []
    vae_model.eval()
    
    original_input = []
    original_label = []
    
    print("Running VAE model.")
    for inputs, labels in dataloader:
        outputs = vae_model(inputs)[0]
        for i in range(outputs.shape[0]): #why do we have two separate loops for outputs and labels?
            vae_output.append(outputs[i])
            original_input.append(inputs[i])
        for i in range(labels.shape[0]):
            vae_label.append(labels[i])
            original_label.append(labels[i])
    original_input = torch.stack(original_input)
    original_label = torch.stack(original_label)
    vae_output = torch.stack(vae_output)
    vae_label = torch.stack(vae_label)
    print(vae_output.shape)
    
    print("Outputs calculated. Constructing dataloader.")
    
    Transform = transforms.Compose([transforms.Resize((299, 299)), transforms.Lambda(lambda x: x.repeat(3, 1, 1))  if vae_output.shape[1]==1  else NoneTransform()])
    
    dataset_reconstructed = CustomTensorDataset(tensors=(vae_output, vae_label), transform = Transform)
    dataloader_reconstructed = DataLoader(dataset_reconstructed, batch_size=batch_size)
    print("dataloader_reconstructed built")
    #print(dataset_reconstructed[1][0].shape)

    # get the model dimensions
    for inputs, labels in dataloader:
        pred = inputs
        size = pred.size()[1:] # discard batch size
        dims = 1
        for dim in size:
            dims *= dim
        break
    
    dataset_original = CustomTensorDataset(tensors=(original_input, original_label), transform = Transform)
    dataloader_original = DataLoader(dataset_original, batch_size=batch_size)
    print("dataloader_original built. Shape is ", dataset_original[1][0].shape)
    
    m1, s1 = _calculate_activation_statistics(dataloader_original, length, model, batch_size, dims)
    print("Calculated m1 and s1")
    m2, s2 = _calculate_activation_statistics(dataloader_reconstructed, length, model, batch_size, dims)
    print("Calculated m2 and s2")

    fid_value = _calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value

if __name__ == "__main__":
    import torch
    import random
    from disvae.utils.modelIO import load_model
    import argparse
    import logging
    import sys

    MODEL_PATH = "results/betaH_mnist"
    MODEL_NAME = "model.pt"
    GPU_AVAILABLE = True

    vae_model = load_model(directory=MODEL_PATH, is_gpu=GPU_AVAILABLE, filename=MODEL_NAME)
    device = torch.device("cpu")
    vae_model = vae_model.to(device)
    vae_model.eval()

    mode = sys.argv[1] # get the name of the dataset you want to measure FID for
    if mode == 'cifar10'  or mode == 'cifar100' or mode == 'mnist':
        # Get the dataset
        dataloader1 = get_dataloaders(mode, batch_size=128)[0]
    else:
        print("Entered wrong name for dataset") 
        sys.exit()

    fid_value = get_fid_value(dataloader1, vae_model)

    print("FID for ", mode, ": ", fid_value)
