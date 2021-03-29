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

def _get_activations(dataloader, length, model, batch_size, dims):
    model.eval()
    if batch_size > length:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = length

    pred_arr = np.empty((length, dims))

    for inputs, labels in (dataloader):
        count = 0
        batch = inputs
        with torch.no_grad():
            pred = model(batch)[0]

        pred = torch.flatten(pred, start_dim=1) # to get the dimensionality vector

        if count != 0: # check if the first batch, if not, then just append by concatenation
            pred_arr = np.concatenate((pred_arr, pred.cpu().detach().numpy()), axis=0)
        else:
            pred_arr[:pred.shape[0]] = pred.cpu().detach().numpy()
        count = count + 1
    
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

def _calculate_activation_statistics(dataloader, length, model, batch_size=32, dims=2048):
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
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def get_fid_value(dataloader, model, batch_size = 32):
    # Calculate FID value
    # Params:
    # dataloader : data to test on
    # model : model to evaluate
    # batch_size : batch size when evaluating model

    length = len(dataloader.dataset)

    # get the model dimensions
    for inputs, labels in dataloader:
        pred = inputs
        size = pred.size()[1:] # discard batch size
        dims = 1
        for dim in size:
            dims *= dim
        break
    
    m1, s1 = _calculate_activation_statistics(dataloader, length, model, batch_size, dims)

    # m2, s2 = preset from the dataset calculated below
    arr = np.empty((length, dims))

    for inputs, labels in (dataloader):
        count = 0
        pred = inputs
        pred = torch.flatten(pred, start_dim=1)
        if count != 0:
            arr = np.concatenate((arr, pred), axis=0)
        else:
            arr[:pred.shape[0]] = pred
        count = count + 1

    m2 = np.mean(arr, axis=0)
    s2 = np.cov(arr, rowvar=False)

    fid_value = _calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value

if __name__ == "__main__":
    import torch
    import random
    from disvae.utils.modelIO import load_model
    import argparse
    import logging
    import sys

    MODEL_PATH = "results/fid_testing"
    MODEL_NAME = "model.pt"
    GPU_AVAILABLE = True

    model = load_model(directory=MODEL_PATH, is_gpu=GPU_AVAILABLE, filename=MODEL_NAME)
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    mode = sys.argv[1] # get the name of the dataset you want to measure FID for
    if mode == 'cifar10'  or mode == 'cifar100' or mode == 'mnist':
        # Get the dataset
        dataloader1 = get_dataloaders(mode, batch_size=32)[0]
    else:
        print("Entered wrong name for dataset") 
        sys.exit()

    fid_value = get_fid_value(dataloader1, model)

    print("FID for ", mode, ": ", fid_value)
