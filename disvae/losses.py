"""
Module containing all vae losses.
"""
import abc
import torch
from torch.nn import functional as F
from torch import optim
from disvae.discriminator import Discriminator
import numpy as np
import math

def get_loss_f(name, capacity=None, alpha=None, beta=None, gamma=None,
               latent_dim=None, data_size=None, mss=False, mutual_info=True, device=None):
    """Return the correct loss function."""
    if name == "betaH":
        return BetaHLoss(beta)
    elif name == "VAE":
        return BetaHLoss(beta=1)
    elif name == "betaB":
        return BetaBLoss(C_min=capacity[0],
                         C_max=capacity[1],
                         C_n_interp=capacity[2],
                         gamma=capacity[3])
    elif name == "factor":
        return FactorKLoss(device,
                           beta,
                           mutual_info)
    elif name == "batchTC":
        return BatchTCLoss(alpha,
                           beta,
                           gamma,
                           data_size,
                           latent_dim,
                           mss)
        # Paper : Isolating Sources of Disentanglement in VAEs
    else:
        raise ValueError("Uknown loss : {}".format(name))


class BaseLoss(abc.ABC):
    """
    Base class for losses.

    Parameters
    ----------
    record_loss_every : int
        Every how many steps to recorsd the loss.
    """

    def __init__(self, record_loss_every=50):
        self.n_train_steps = 0
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

    def _pre_call(self, is_train, storer):
        if is_train:
            self.n_train_steps += 1

        if not is_train or self.n_train_steps % self.record_loss_every == 1:
            storer = storer
        else:
            storer = None

        return storer


class BetaHLoss(BaseLoss):
    """
    Compute the Beta-VAE loss as in [1]

    Parameters
    ----------
    beta : float, optional
        Weight of the kl divergence.

    References:
        [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
        a constrained variational framework." (2016).
    """

    def __init__(self, beta=4):
        super().__init__()
        self.beta = beta

    def __call__(self, data, recon_data, latent_dist, is_train, storer):
        storer = self._pre_call(is_train, storer)

        rec_loss = _reconstruction_loss(data, recon_data, storer=storer)
        kl_loss = _kl_normal_loss(*latent_dist, storer)
        loss = rec_loss + self.beta * kl_loss

        if storer is not None:
            storer['loss'].append(loss.item())

        return loss


class BetaBLoss(BaseLoss):
    """
    Compute the Beta-VAE loss as in [1]

    Parameters
    ----------
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

    def __init__(self, C_min=0., C_max=5., C_n_interp=25000, gamma=30.):
        super().__init__()
        self.gamma = gamma
        self.C_min = C_min
        self.C_max = C_max
        self.C_n_interp = C_n_interp

    def __call__(self, data, recon_data, latent_dist, is_train, storer):
        storer = self._pre_call(is_train, storer)

        rec_loss = _reconstruction_loss(data, recon_data, storer=storer)
        kl_loss = _kl_normal_loss(*latent_dist, storer)

        if is_train:
            # linearly increasing C
            assert self.C_max > self.C_min
            C_delta = (self.C_max - self.C_min)
            C = min(self.C_min + C_delta * self.n_train_steps / self.C_n_interp, self.C_max)
        else:
            C = self.C_max

        loss = rec_loss + self.gamma * (kl_loss - C).abs()

        batch_size = data.size(0)
        if storer is not None:
            storer['loss'].append(loss.item())

        return loss


class FactorKLoss(BaseLoss):
    """
        Compute the Factor-VAE loss as per Algorithm 2 of [1]

        Parameters
        ----------
        device : torch.device

        beta : float, optional
            Weight of the TC loss term. `gamma` in the paper.

        mutual_info : bool
            True : includes the mutual information term in the loss
            False : removes mutual information

        discriminator : disvae.discriminator.Discriminator

        optimizer_d : torch.optim

        References :
            [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
            arXiv preprint arXiv:1802.05983 (2018).
        """

    def __init__(self, device, beta=40.,
                 mutual_info=True,
                 disc_kwargs=dict(neg_slope=0.2, latent_dim=10, hidden_units=1000),
                 optim_kwargs=dict(lr=5e-4, betas=(0.5, 0.9))):
        super().__init__()
        self.beta = beta
        self.device = device
        self.mutual_info = mutual_info

        self.discriminator = Discriminator(**disc_kwargs).to(self.device)

        self.optimizer_d = optim.Adam(self.discriminator.parameters(), **optim_kwargs)

    def __call__(self, data, model, optimizer, storer):
        storer = self._pre_call(model.training, storer)

        # factor-vae split data into two batches. In the paper they sample 2 batches
        batch_size = data.size(dim=0)
        half_batch_size = batch_size // 2
        data = data.split(half_batch_size)
        data1 = data[0]
        data2 = data[1]

        # Factor VAE Loss
        recon_batch, latent_dist, latent_sample1 = model(data1)
        rec_loss = _reconstruction_loss(data1, recon_batch, storer=storer)
        kl_loss = _kl_normal_loss(*latent_dist, storer)
        d_z = self.discriminator(latent_sample1)
        
        # clamping to 0 because TC cannot be negative : TESTTTTTTTT
        tc_loss = (F.logsigmoid(d_z) - F.logsigmoid(1 - d_z)).clamp(0).mean()

        if self.mutual_info:
            # return vae loss
            vae_loss = rec_loss + kl_loss + self.beta * tc_loss
        else:
            # return vae loss without mutual information term
            dw_kl_loss = _dimwise_kl_loss(*latent_dist, storer)
            vae_loss = rec_loss + self.beta * tc_loss + dw_kl_loss

        if storer is not None:
            storer['loss'].append(vae_loss.item())
            storer['tc_loss'].append(tc_loss.item())

        if not model.training:
            # don't backprop if evaluating
            return vae_loss

        # Run VAE optimizer
        optimizer.zero_grad()
        vae_loss.backward(retain_graph=True)
        optimizer.step()

        # Discriminator Loss
        # Get second sample of latent distribution
        latent_sample2 = model.sample_latent(data2)
        z_perm = _permute_dims(latent_sample2).detach()
        d_z_perm = self.discriminator(z_perm)
        # Calculate total correlation loss
        d_tc_loss = - (0.5 * (F.logsigmoid(d_z) + F.logsigmoid(1 - d_z_perm))).mean()

        # Run discriminator optimizer
        self.optimizer_d.zero_grad()
        d_tc_loss.backward()
        self.optimizer_d.step()

        if storer is not None:
            storer['discrim_loss'].append(d_tc_loss.item())

        return vae_loss

class BatchTCLoss(BaseLoss):
    """
        Compute the decomposed KL loss with either minibatch weighted sampling or
        minibatch stratified sampling according to [1]

        Parameters
        ----------
        alpha : float
            Weight of the mutual information term.

        beta : float
            Weight of the total correlation term.

        gamma : float
            Weight of the dimension-wise KL term.

        latent_dim: int
            Dimension of the latent variable

        mss : bool
            Selects either minibatch stratified sampling (True) or minibatch
            weighted  sampling (False)

        References :
           [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
           autoencoders." Advances in Neural Information Processing Systems. 2018.
    """

    def __init__(self, alpha, beta, gamma, data_size, latent_dim, mss=False):
        super().__init__()

        self.latent_dim = latent_dim
        self.dataset_size = data_size
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.mss = mss # minibatch stratified sampling

    def __call__(self, data, recon_batch, latent_dist, is_train, storer, latent_sample=None):
        storer = self._pre_call(is_train, storer)

        batch_size = data.size(0)
        # change latent dist to torch.tensor (could probably avoid this)
        latent_dist = torch.stack((latent_dist[0], latent_dist[1]), dim=2)

        # calculate log q(z|x) and _log q(z) matrix
        logqz_condx = self._log_density_normal(latent_sample, latent_dist, batch_size, return_matrix=False).sum(dim=1)
        _logqz = self._log_density_normal(latent_sample, latent_dist, batch_size, return_matrix=True)

        if not self.mss:
            # minibatch weighted sampling
            logqz_prodmarginals = (torch.logsumexp(_logqz, dim=1, keepdim=False)
                                   - math.log(batch_size * self.dataset_size)).sum(dim=1)
            logqz = torch.logsumexp(_logqz.sum(2), dim=1, keepdim=False) \
                    - math.log(batch_size * self.dataset_size)
        else:
            # minibatch stratified sampling
            logiw_matrix = self._log_importance_weight_matrix(batch_size, self.dataset_size)
            logqz = torch.logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            logqz_prodmarginals = torch.logsumexp(logiw_matrix.view(batch_size, batch_size, 1)
                                                  + _logqz, dim=1, keepdim=False).sum(1)

        # rec loss, mutual information, total correlation and dim-wise kl
        rec_loss = _reconstruction_loss(data, recon_batch, storer=storer)
        mi_loss = (logqz_condx - logqz).mean()
        tc_loss = (logqz - logqz_prodmarginals).mean()
        dw_kl_loss = _dimwise_kl_loss(latent_dist[::,0],latent_dist[::,1], storer=storer)

        # total loss
        loss = rec_loss + self.alpha * mi_loss + self.beta * tc_loss + self.gamma * dw_kl_loss

        if storer is not None:
            storer['loss'].append(loss.item())
            storer['mi_loss'].append(mi_loss.item())
            storer['tc_loss'].append(tc_loss.item())

            # TODO Remove this when visualisation fixed
            tc_loss_vec = (logqz - logqz_prodmarginals)
            for i in range(self.latent_dim):
                storer['kl_loss_' + str(i)].append(tc_loss_vec[i].item())

        return loss

    def _log_density_normal(self, latent_sample, latent_dist, batch_size, return_matrix=False):
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
            latent_sample = latent_sample.view(batch_size, 1, self.latent_dim)
            latent_dist = latent_dist.view(1, batch_size, self.latent_dim, 2)

        mu = latent_dist.select(-1, 0)
        logvar = latent_dist.select(-1, 1)

        inv_var = torch.exp(-logvar)
        tmp = (latent_sample - mu)
        log_density = -0.5 * (torch.pow(tmp,2) * inv_var + logvar + np.log(2*np.pi))

        return log_density

    def _log_importance_weight_matrix(self, batch_size, dataset_size):
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
        W.view(-1)[::M+1] = 1 / N
        W.view(-1)[1::M+1] = strat_weight
        W[M-1, 0] = strat_weight
        return W.log()

def _dimwise_kl_loss(mean, logvar, storer=None):
    """
        Calculates the dimension-wise KL divergence between posterior and prior for each
        latent dimension.

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
    dw_kl = (- 0.5 * logvar + 0.5 * (torch.exp(logvar + torch.pow(mean, 2)))
             - 0.5).sum(dim=1).mean()

    if storer is not None:
        storer['dw_kl_loss'].append(dw_kl.item())

    return dw_kl

def _reconstruction_loss(data, recon_data, distribution="bernoulli", storer=None):
    """
    Calculates the per image reconstruction loss for a batch of data.

    Parameters
    ----------
    data : torch.Tensor
        Input data (e.g. batch of images). Shape : (batch_size, n_chan,
        height, width).

    recon_data : torch.Tensor
        Reconstructed data. Shape : (batch_size, n_chan, height, width).

    distribution : {"bernoulli", "gaussian", "laplace"}
        Distribution of the likelihood on the each pixel. Implicitely defines the
        loss Bernoulli corresponds to a binary cross entropy (bse) loss and is the
        most commonly used. It has the issue that it doesn't penalize the same
        way (0.1,0.2) and (0.4,0.5), which might not be optimal. Gaussian
        distribution corresponds to MSE, and is sometimes used, but hard to train
        ecause it ends up focusing only a few pixels that are very wrong. Laplace
        distribution corresponds to L1 solves partially the issue of MSE.

    storer : dict
        Dictionary in which to store important variables for vizualisation.

    Returns
    -------
        loss : torch.Tensor
            Per image cross entropy (i.e. normalized per batch but not pixel and
            channel)
    """
    batch_size, n_chan, height, width = recon_data.size()
    is_colored = n_chan == 3

    if distribution == "bernoulli":
        loss = F.binary_cross_entropy(recon_data, data, reduction="sum")
    elif distribution == "gaussian":
        # loss in [0,255] space but normalized by 255 to not be too big
        loss = F.mse_loss(recon_data * 255, data * 255, reduction="sum") / 255
    elif distribution == "laplace":
        # loss in [0,255] space but normalized by 255 to not be too big but
        # multiply by 255 and divide 255, is the same as not doing anything for L1
        loss = F.l1_loss(recon_data, data, reduction="sum")
    else:
        raise ValueError("Unkown distribution: {}".format(distribution))

    loss = loss / batch_size

    if storer is not None:
        storer['recon_loss'].append(loss.item())

    return loss


def _kl_normal_loss(mean, logvar, storer=None):
    """
    Calculates the KL divergence between a normal distribution
    with diagonal covariance and a unit normal distribution.

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
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
    total_kl = latent_kl.sum()

    if storer is not None:
        storer['kl_loss'].append(total_kl.item())
        for i in range(latent_dim):
            storer['kl_loss_' + str(i)].append(latent_kl[i].item())

    return total_kl


def _permute_dims(latent_sample):
    """
    Implementation of Algorithm 1 in ref [1]. Randomly permutes the sample from
    q(z) (latent_dist) across the batch for each of the latent dimensions (mean
    and log_var).

    Parameters
    ----------
    latent_sample: torch.Tensor
        sample from the latent dimension using the reparameterisation trick
        shape : (batch_size, latent_dim).

    References
    ----------
        [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
        arXiv preprint arXiv:1802.05983 (2018).

    """
    perm = torch.zeros_like(latent_sample)
    batch_size, dim_z = perm.size()

    for z in range(dim_z):
        pi = torch.randperm(batch_size).to(latent_sample.device)
        perm[:, z] = latent_sample[pi, z]

    return perm
