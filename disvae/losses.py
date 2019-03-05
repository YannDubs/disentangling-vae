"""
Module containing all vae losses.
"""
import abc
import torch
from torch.nn import functional as F


def get_loss_f(name, is_color, capacity=None, beta=None, discriminator=None, optimizer_d=None, device=None):
    """Return the correct loss function."""
    if name == "betaH":
        return BetaHLoss(is_color, beta)
    elif name == "VAE":
        return BetaHLoss(is_color, beta=1)
    elif name == "betaB":
        return BetaBLoss(is_color,
                         C_min=capacity[0],
                         C_max=capacity[1],
                         C_n_interp=capacity[2],
                         gamma=capacity[3])
    elif name == "factorising":
        # Paper: Disentangling by Factorising
        return FactorKLoss(is_color,
                           discriminator,
                           optimizer_d,
                           device,
                           beta)
    elif name == "batchTC":
        raise ValueError("{} loss not yet implemented".format(name))
        # Paper : Isolating Sources of Disentanglement in VAEs
        # return BatchTCLoss(**kwargs)
    else:
        raise ValueError("Uknown loss : {}".format(name))


class BaseLoss(abc.ABC):
    """
    Base class for losses.

    Parameters
    ----------
    is_color : bool
        Whether the images are in color.
    """

    def __init__(self, is_color, record_loss_every=5):
        self.n_train_steps = 0
        self.is_color = is_color
        self.record_loss_every = record_loss_every

    @abc.abstractmethod
    def __call__(self, data, recon_data, latent_dist, is_train, storer):
        """
        Calculates loss for a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images). Shape : (batch_size, n_chan,
            height, width).

        recon_data : torch.Tensor
            Reconstructed data. Shape : (batch_size, n_chan, height, width).

        latent_dist : tuple of torch.tensor
            sufficient statistics of the latent dimension. E.g. for gaussian
            (mean, log_var) each of shape : (batch_size, latent_dim).

        storer : dict
            Dictionary in which to store important variables for vizualisation.
        """
        pass

    def _pre_call(self, is_train, storer):
        if is_train:
            self.n_train_steps += 1

        if is_train and self.n_train_steps % self.record_loss_every == 1:
            storer = storer
        else:
            storer = None

        return storer


class BetaHLoss(BaseLoss):
    """
    Compute the Beta-VAE loss as in [1]

    Parameters
    ----------
    is_color : bool
        Whether the image are in color.

    beta : float, optional
        Weight of the kl divergence.

    References:
        [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
        a constrained variational framework." (2016).
    """

    def __init__(self, is_color, beta=4):
        super().__init__(is_color)
        self.beta = beta

    def __call__(self, data, recon_data, latent_dist, is_train, storer):
        storer = self._pre_call(is_train, storer)

        rec_loss = _reconstruction_loss(data, recon_data, self.is_color)
        kl_loss = _kl_normal_loss(*latent_dist, storer)
        loss = rec_loss + self.beta * kl_loss

        if storer is not None:
            storer['recon_loss'].append(rec_loss.item())
            storer['kl_loss'].append(kl_loss.item())
            storer['loss'].append(loss.item())

        return loss


class BetaBLoss(BaseLoss):
    """
    Compute the Beta-VAE loss as in [1]

    Parameters
    ----------
    is_color : bool
        Whether the image are in color.

    C_min : float, optional
        Starting capacity C.

    C_max : float, optional
        Final capacity C.

    C_n_interp : float, optional
        Number of interpolating steps for C.

    gamma : float, optional
        Weight of the KL divergence term.

    References:
        [1] Burgess, Christopher P., et al. "Understanding disentangling in
        $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
    """

    def __init__(self, is_color, C_min=0., C_max=5., C_n_interp=25000, gamma=30.):
        super().__init__(is_color)
        self.gamma = gamma
        self.C_min = C_min
        self.C_max = C_max
        self.C_n_interp = C_n_interp

    def __call__(self, data, recon_data, latent_dist, is_train, storer):
        storer = self._pre_call(is_train, storer)

        rec_loss = _reconstruction_loss(data, recon_data, self.is_color)
        kl_loss = _kl_normal_loss(*latent_dist, storer)

        # linearly increasing C
        C_delta = (self.C_max - self.C_min)
        C = self.C_min + C_delta * self.n_train_steps / self.C_n_interp

        loss = rec_loss + self.gamma * (kl_loss - C).abs()

        return loss


class FactorKLoss(BaseLoss):
    """
        Compute the Factor-VAE loss as per Algorithm 2 of [1]

        Parameters
        ----------
        is_color : bool
            Whether the image are in color.

        discriminator : disvae.discriminator.Discriminator

        optimizer_d : torch.optim.adam.Adam

        device : torch.device

        beta : float, optional
            Weight of the TC loss term.

        References :
            [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
            arXiv preprint arXiv:1802.05983 (2018).
        """

    def __init__(self, is_color, discriminator, optimizer_d, device, beta=40.):
        super().__init__(is_color)
        self.beta = beta
        self.device = device
        self.discriminator = discriminator
        self.optimizer_d = optimizer_d

    def __call__(self, data, model, optimizer, is_train, storer):
        storer = self._pre_call(is_train, storer)

        # If factor-vae split data into two batches
        batch_size = data.size(dim=0)
        half_batch_size = batch_size // 2
        data = data.split(half_batch_size)
        data1 = data[0].to(self.device)
        data2 = data[1].to(self.device)

        # Initialise the targets for cross_entropy loss
        ones = torch.ones(batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # Get first sample of latent distribution
        recon_batch, latent_dist = model(data1)

        # Get reconstruction loss
        rec_loss = _reconstruction_loss(data1, recon_batch, self.is_color)

        # Get KL-Divergence (latent_dist[0] = mean, latent_dist[1] = log_var)
        kl_loss = _kl_normal_loss(*latent_dist, storer)

        # Run latent distribution through discriminator
        d_z = self.discriminator(latent_dist)

        # Calculate the total correlation (TC) loss term
        tc_loss = (d_z[:, :, :1] - d_z[:, :, 1:]).mean()

        # Factor VAE loss
        vae_loss = rec_loss + kl_loss + self.beta * tc_loss

        # Make loss independent of number of pixels
        vae_loss = vae_loss / model.num_pixels

        # Train loss for function return
        train_loss = vae_loss.item()

        # Run VAE optimizer
        optimizer.zero_grad()
        vae_loss.backward(retain_graph=True)
        optimizer.step()

        # Get second sample of latent distribution
        _, latent_dist = model(data2)

        # Create a permutation of the latent distribution
        z_perm = _permute_dims(latent_dist)

        # Run permeated latent distribution through discriminator
        d_z_perm = self.discriminator(z_perm)

        # Calculate total correlation loss
        d_tc_loss = 0.5 * (F.cross_entropy(d_z.reshape((batch_size, 2)), zeros)
                           + F.cross_entropy(d_z_perm.reshape((batch_size, 2)), ones))

        # Run discriminator optimizer
        self.optimizer_d.zero_grad()
        d_tc_loss.backward()
        self.optimizer_d.step()

        return train_loss


def _reconstruction_loss(data, recon_data, is_color):
    """
    Calculates the reconstruction loss for a batch of data.

    Notes
    -----
    Usually for color images we use a Gaussian distribution, corresponding to
    a MSE loss. I think binary cross entropy makes mroe sense as each channel
    is bounded by 255. I thus simply renormalize before using cross entropy.

    Parameters
    ----------
    data : torch.Tensor
        Input data (e.g. batch of images). Shape : (batch_size, n_chan,
        height, width).

    recon_data : torch.Tensor
        Reconstructed data. Shape : (batch_size, n_chan, height, width).

    is_color : bool
        Whether the image are in color.
    """
    batch_size, n_chan, _, _ = recon_data.size()

    if is_color:
        recon_data = recon_data / 255
        data = data / 255

    loss = F.binary_cross_entropy(recon_data, data,
                                  reduction="sum") / batch_size

    return loss


def _kl_normal_loss(mean, logvar, storer=None):
    """
    Calculates the KL divergence between a normal distribution with
    diagonal covariance and a unit normal distribution.

    Parameters
    ----------
    mean : torch.Tensor
        Mean of the normal distribution. Shape (batch_size, latent_dim) where
        D is dimension of distribution.

    logvar : torch.Tensor
        Diagonal log variance of the normal distribution. Shape (batch_size,
        latent_dim)

    storer : dict
        Dictionary in which to store important variables for vizualisation.
    """
    latent_dim = mean.size(1)
    # batch mean of kl for each latent dimension
    latent_kl = (-0.5 * (1 + logvar - mean.pow(2) - logvar.exp())).mean(dim=0)
    total_kl = latent_kl.sum()

    if storer is not None:
        storer['kl_loss'].append(total_kl.item())
        for i in range(latent_dim):
            storer['kl_loss_' + str(i)] += latent_kl[i].item()
    return total_kl


def _permute_dims(latent_dist):
    """
    Implementation of Algorithm 1 in ref [1]. Randomly permutes the sample from
    q(z) (latent_dist) across the batch for each of the latent dimensions (mean
    and log_var).

    Parameters
    ----------
    latent_dist : tuple of torch.tensor
        sufficient statistics of the latent dimension. E.g. for gaussian
        (mean, log_var) each of shape : (batch_size, latent_dim).

    References :
        [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
        arXiv preprint arXiv:1802.05983 (2018).

    """

    perm = torch.zeros(latent_dist.size())
    dim_j, batch_size, dim_z = latent_dist.size()

    for j in range(dim_j):
        for z in range(dim_z):
            pi = torch.randperm(batch_size)
            perm[j, :, z] = latent_dist[j, pi, z]

    return perm
