import torch

from disvae.utils.modelIO import load_model

MODEL_PATH = "results/fid_testing"
MODEL_NAME = "model.pt"
GPU_AVAILABLE = True

model = load_model(directory=MODEL_PATH, is_gpu=GPU_AVAILABLE, filename=MODEL_NAME)
device = torch.device("cpu")
model = model.to(device)
model.eval()

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


def get_activations(dataloader, files, model, batch_size, dims):
    # Calculates the activations of the last layer for all images.
    # Params:
    # files       : List of image file indices
    # model       : Instance of model
    # batch_size  : Batch size of images for the model to process at once.
    #               Make sure that the number of samples is a multiple of
    #               the batch size, otherwise some samples are ignored. This
    #               behavior matches the original FID score implementation.
    # dims        : Dimensionality of features returned by model
    # Returns:
    # A numpy array of dimension (num images, dims) that contains the
    # activations of the given tensor when feeding the model with the
    # query tensor.
    
    model.eval()
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    pred_arr = np.empty((len(files), dims))

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

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
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


def calculate_activation_statistics(dataloader, files, model, batch_size=32, dims=2048):
    # Calculation of the statistics used by the FID.
    # Params:
    # files       : List of image files paths
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
        
    act = get_activations(dataloader, files, model, batch_size, dims)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

transform = transforms.Compose([transforms.Resize(size=(32, 32)), transforms.ToTensor()]) # apply the image transformations   

# Get the dataset
dataset = datasets.CIFAR10(
root="data",
train=True,
download=True,
transform=transform)

# Get a subset of the images from the dataset you want to compare FID scores for
subset_indices1 = list(range(96))

# Dataloader loading the first n images you wanted from the dataset
dataloader1 = torch.utils.data.DataLoader(dataset,
                                      batch_size=32,
                                      shuffle=False,
                                      sampler=SequentialSampler(subset_indices1), #SubsetRandomSampler(subset_indices1),
                                      drop_last=False,
                                      num_workers=0)

dims = 3072 # dimensionality of the feature vector
batch_size = dims

m1, s1 = calculate_activation_statistics(dataloader1, subset_indices1, model, batch_size, dims)

# m2, s2 = preset from the dataset calculated below
arr = np.empty((len(subset_indices1), dims))

for inputs, labels in (dataloader1):
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

fid_value = calculate_frechet_distance(m1, s1, m2, s2)

print('FID: ', fid_value)