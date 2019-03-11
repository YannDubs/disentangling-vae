import numpy as np
import torch
from scipy import stats


class LatentTraverser():
    def __init__(self, latent_dim, sample_prior=False):
        """
        LatentTraverser is used to generate traversals of the latent space.

        Parameters
        ----------
        latent_dim : int
            Dimensionality of latent output.

        sample_prior : bool
            If False fixes samples in untraversed latent dimensions. If True
            samples untraversed latent dimensions from prior.
        """
        self.latent_dim = latent_dim
        self.sample_prior = sample_prior

    def traverse_line(self, sample_latent_space=None, idx=None, size=5):
        """
        Returns a (size, latent_size) latent sample, corresponding to a traversal
        of a continuous latent variable indicated by idx.

        Parameters
        ----------
        idx : int or None
            Index of continuous dimension to traverse. If the continuous latent
            vector is 10 dimensional and idx = 7, then the 7th dimension
            will be traversed while all others will either be fixed or randomly
            sampled. If None, no latent is traversed and all latent
            dimensions are randomly sampled or kept fixed.

        sample_latent_space : torch.Tensor or None
            The latent space of a sample which has been processed by the encoder.
            The dimensions are (size, num_latent_dims)

        idx : int or None
            Indicates which line (latent dimension) to traverse

        size : int
            Number of samples to generate.
        """
        if self.sample_prior and sample_latent_space is None:
            samples = np.random.normal(size=(size, self.latent_dim))
        elif sample_latent_space is None:
            samples = np.zeros(shape=(size, self.latent_dim))
        else:
            samples = np.repeat(sample_latent_space.cpu().numpy(), size, axis=0)

        if idx is not None:
            # Sweep over linearly spaced coordinates transformed through the
            # inverse CDF (ppf) of a gaussian since the prior of the latent
            # space is gaussian
            cdf_traversal = np.linspace(0.05, 0.95, size)
            traversal = stats.norm.ppf(cdf_traversal)
            for i in range(size):
                samples[i, idx] = traversal[i]

        return torch.Tensor(samples)

    def traverse_grid(self, idx=None, axis=0, size=(5, 5)):
        """
        Returns a (size[0] * size[1], latent_size) latent sample, corresponding to a
        two dimensional traversal of the latent space.

        Parameters
        ----------
        idx : int or None
            Index of continuous dimension to traverse. If the continuous latent
            vector is 10 dimensional and idx = 7, then the 7th dimension
            will be traversed while all others will either be fixed or randomly
            sampled. If None, no latent is traversed and all latent
            dimensions are randomly sampled or kept fixed.

        axis : int
            Either 0 for traversal across the rows or 1 for traversal across
            the columns.

        size : tuple of ints
            Shape of grid to generate. E.g. (6, 4).
        """
        num_samples = size[0] * size[1]

        if self.sample_prior:
            samples = np.random.normal(size=(num_samples, self.latent_dim))
        else:
            samples = np.zeros(shape=(num_samples, self.latent_dim))

        if idx is not None:
            # Sweep over linearly spaced coordinates transformed through the
            # inverse CDF (ppf) of a gaussian since the prior of the latent
            # space is gaussian
            cdf_traversal = np.linspace(0.05, 0.95, size[axis])
            traversal = stats.norm.ppf(cdf_traversal)

            for i in range(size[0]):
                for j in range(size[1]):
                    if axis == 0:
                        samples[i * size[1] + j, idx] = traversal[i]
                    else:
                        samples[i * size[1] + j, idx] = traversal[j]

        return torch.Tensor(samples)
