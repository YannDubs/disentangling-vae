
import torch
import numpy as np

def log_density_normal(latent_sample, latent_dist, batch_size, return_matrix=False):
    """
        Calculates log density of a normal distribution with latent samples
        and latent distribution parameters.

        Parameters
        ----------
        latent_sample: torch.Tensor
            sample from the latent dimension using the reparameterisation trick
            shape : (batch_size, latent_dim).

        latent_dist: torch.Tensor
            parameters of the latent distribution: mean and logvar
            shape: (batch_size, latent_dim, 2)

        batch_size: int
            number of training images in the batch

        return_matrix: bool
            True returns size (batch_size, batch_size, latent_dim) - used for mws and mss
            False returns size (batch_size, latent_dim)
        """
    if return_matrix:
        latent_sample = latent_sample.view(batch_size, 1, latent_dist.size(1))
        latent_dist = latent_dist.view(1, batch_size, latent_dist.size(1), 2)

    mu = latent_dist.select(-1, 0)
    logvar = latent_dist.select(-1, 1)

    inv_var = torch.exp(-logvar)
    tmp = (latent_sample - mu)
    log_density = -0.5 * (torch.pow(tmp, 2) * inv_var + logvar + np.log(2 * np.pi))

    return log_density


def log_importance_weight_matrix(batch_size, dataset_size):
    """
        Calculates a log importance weight matrix

        Parameters
        ----------
        batch_size: int
            number of training images in the batch

        dataset_size: int
        number of training images in the dataset
    """
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M + 1] = 1 / N
    W.view(-1)[1::M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()
