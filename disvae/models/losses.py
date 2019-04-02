"""
Module containing all vae losses.
"""
import abc
import math

import torch
from torch.nn import functional as F
from torch import optim

from .discriminator import Discriminator
from disvae.utils.math import log_density_normal, log_importance_weight_matrix


# TO-DO: clean data_size and device
def get_loss_f(name, kwargs_parse={}):
    """Return the correct loss function given the argparse arguments."""
    kwargs_all = dict(rec_dist=kwargs_parse["rec_dist"], steps_anneal=kwargs_parse["reg_anneal"])
    if name == "betaH":
        return BetaHLoss(beta=kwargs_parse["betaH_B"], **kwargs_all)
    elif name == "VAE":
        return BetaHLoss(beta=1, **kwargs_all)
    elif name == "betaB":
        return BetaBLoss(C_init=kwargs_parse["betaB_initC"],
                         C_fin=kwargs_parse["betaB_finC"],
                         C_n_interp=kwargs_parse["betaB_stepsC"],
                         gamma=kwargs_parse["betaB_G"],
                         **kwargs_all)
    elif name == "factor":
        return FactorKLoss(kwargs_parse["device"],
                           kwargs_parse["data_size"],
                           gamma=kwargs_parse["factor_G"],
                           is_mutual_info=not kwargs_parse["no_mutual_info"],
                           is_mss=not kwargs_parse["no_mss"],
                           **kwargs_all)
    elif name == "batchTC":
        return BatchTCLoss(kwargs_parse["device"],
                           kwargs_parse["data_size"],
                           alpha=kwargs_parse["batchTC_A"],
                           beta=kwargs_parse["batchTC_B"],
                           gamma=kwargs_parse["batchTC_G"],
                           is_mss=not kwargs_parse["no_mss"],
                           **kwargs_all)
    else:
        raise ValueError("Uknown loss : {}".format(name))


class BaseLoss(abc.ABC):
    """
    Base class for losses.

    Parameters
    ----------
    record_loss_every: int, optional
        Every how many steps to recorsd the loss.

    rec_dist: {"bernoulli", "gaussian", "laplace"}, optional
        Reconstruction distribution istribution of the likelihood on the each pixel.
        Implicitely defines the reconstruction loss. Bernoulli corresponds to a
        binary cross entropy (bse), Gaussian corresponds to MSE, Laplace
        corresponds to L1.

    steps_anneal: nool, optional
        Number of annealing steps where gradually adding the regularisation.
    """

    def __init__(self, record_loss_every=50, rec_dist="bernoulli", steps_anneal=0):
        self.n_train_steps = 0
        self.record_loss_every = record_loss_every
        self.rec_dist = rec_dist
        self.steps_anneal = steps_anneal

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

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.
    """

    def __init__(self, beta=4, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def __call__(self, data, recon_data, latent_dist, is_train, storer):
        storer = self._pre_call(is_train, storer)

        rec_loss = _reconstruction_loss(data, recon_data,
                                        storer=storer, distribution=self.rec_dist)
        kl_loss = _kl_normal_loss(*latent_dist, storer)
        anneal_rec = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if is_train else 1)
        loss = rec_loss + anneal_rec * (self.beta * kl_loss)

        if storer is not None:
            storer['loss'].append(loss.item())

        return loss


class BetaBLoss(BaseLoss):
    """
    Compute the Beta-VAE loss as in [1]

    Parameters
    ----------
    C_init : float, optional
        Starting annealed capacity C.

    C_fin : float, optional
        Final annealed capacity C.

    C_n_interp : float, optional
        Number of training iterations for interpolating C.

    gamma : float, optional
        Weight of the KL divergence term.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
        [1] Burgess, Christopher P., et al. "Understanding disentangling in
        $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
    """

    def __init__(self, C_init=0., C_fin=5., C_n_interp=25000, gamma=30., **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.C_init = C_init
        self.C_fin = C_fin
        self.C_n_interp = C_n_interp

    def __call__(self, data, recon_data, latent_dist, is_train, storer):
        storer = self._pre_call(is_train, storer)

        rec_loss = _reconstruction_loss(data, recon_data,
                                        storer=storer, distribution=self.rec_dist)
        kl_loss = _kl_normal_loss(*latent_dist, storer)

        C = (linear_annealing(self.C_init, self.C_fin, self.n_train_steps, self.C_n_interp)
             if is_train else self.C_fin)

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

        is_mutual_info : bool
            True : includes the mutual information term in the loss
            False : removes mutual information

        discriminator : disvae.discriminator.Discriminator

        optimizer_d : torch.optim

        kwargs:
            Additional arguments for `BaseLoss`, e.g. rec_dist`.

        References
        ----------
            [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
            arXiv preprint arXiv:1802.05983 (2018).
        """

    def __init__(self, device, data_size, gamma=40., is_mutual_info=True, is_mss=False,
                 disc_kwargs=dict(neg_slope=0.2, latent_dim=10, hidden_units=1000),
                 optim_kwargs=dict(lr=5e-4, betas=(0.5, 0.9)),
                 **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.data_size = data_size
        self.device = device
        self.is_mutual_info = is_mutual_info
        self.is_mss = is_mss

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
        rec_loss = _reconstruction_loss(data1, recon_batch,
                                        storer=storer, distribution=self.rec_dist)
        # TODO: remove this kl_loss term once viz is sorted
        # https://github.com/YannDubs/disentangling-vae/pull/25#issuecomment-473535863
        kl_loss = _kl_normal_loss(*latent_dist, storer)
        d_z = self.discriminator(latent_sample1)

        # clamping to 0 because TC cannot be negative : TEST
        tc_loss = (F.logsigmoid(d_z) - F.logsigmoid(1 - d_z)).clamp(0).mean()

        anneal_rec = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if model.training else 1)

        # TODO replace this code with the following commented out code after viz is fixed
        # https://github.com/YannDubs/disentangling-vae/pull/25#issuecomment-473535863
        if self.is_mutual_info:
            # return vae loss
            vae_loss = rec_loss + anneal_rec * (kl_loss + self.gamma * tc_loss)
        else:
            # return vae loss without mutual information term
            # change latent dist to torch.tensor (could probably avoid this)
            latent_dist = torch.stack((latent_dist[0], latent_dist[1]), dim=2)
            # calculate log p(z)
            prior_params = torch.zeros(half_batch_size, latent_dist.size(1), 2).to(self.device)
            logpz = log_density_normal(latent_sample1, prior_params, half_batch_size,
                                       return_matrix=False).view(half_batch_size, -1).sum(1)

            if not self.is_mss:
                # minibatch weighted sampling
                _, logqz_prodmarginals = _minibatch_weighted_sampling(latent_dist, latent_sample1,
                                                                      self.data_size)
            else:
                # minibatch stratified sampling
                _, logqz_prodmarginals = _minibatch_stratified_sampling(latent_dist, latent_sample1,
                                                                        self.data_size)

            gamma = self.gamma + 1

            dw_kl_loss = (logqz_prodmarginals - logpz).mean()
            vae_loss = rec_loss + anneal_rec * (gamma * tc_loss + dw_kl_loss)

        # if self.is_mutual_info:
        #     beta = self.beta
        #     kl_loss = _kl_normal_loss(*latent_dist, storer)
        # else:
        #     # beta has to be increased by one for correct comparaison
        #     # as the TC term is included in `_kl_normal_loss`
        #     beta = self.beta + 1
        #     kl_loss = _dimwise_kl_loss(*latent_dist, storer)
        #
        # vae_loss = rec_loss + kl_loss + beta * tc_loss

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
        data_size: int
            Size of the dataset

        alpha : float
            Weight of the mutual information term.

        beta : float
            Weight of the total correlation term.

        gamma : float
            Weight of the dimension-wise KL term.

        latent_dim: int
            Dimension of the latent variable

        is_mss : bool
            Selects either minibatch stratified sampling (True) or minibatch
            weighted  sampling (False)

        kwargs:
            Additional arguments for `BaseLoss`, e.g. rec_dist`.

        References
        ----------
           [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
           autoencoders." Advances in Neural Information Processing Systems. 2018.
    """

    def __init__(self, device, data_size, alpha=1., beta=6., gamma=1., is_mss=False, **kwargs):
        super().__init__(**kwargs)
        # beta values: dsprites: 6, celeba: 15
        self.device = device
        self.dataset_size = data_size
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.is_mss = is_mss  # minibatch stratified sampling

    def __call__(self, data, recon_batch, latent_dist, is_train, storer, latent_sample=None):
        storer = self._pre_call(is_train, storer)

        batch_size = data.size(0)
        # change latent dist to torch.tensor (could probably avoid this)
        latent_dist = torch.stack((latent_dist[0], latent_dist[1]), dim=2)

        # calculate log q(z|x) and _log q(z) matrix
        logqz_condx = log_density_normal(latent_sample, latent_dist, batch_size,
                                         return_matrix=False).sum(dim=1)

        # calculate log p(z)
        prior_params = torch.zeros(batch_size, latent_dist.size(1), 2).to(self.device)
        logpz = log_density_normal(latent_sample, prior_params, batch_size,
                                   return_matrix=False).view(batch_size, -1).sum(1)

        if not self.is_mss:
            # minibatch weighted sampling
            logqz, logqz_prodmarginals = _minibatch_weighted_sampling(latent_dist, latent_sample,
                                                                      self.dataset_size)

        else:
            # minibatch stratified sampling
            logqz, logqz_prodmarginals = _minibatch_stratified_sampling(latent_dist, latent_sample,
                                                                        self.dataset_size)

        # rec loss, mutual information, total correlation and dim-wise kl
        rec_loss = _reconstruction_loss(data, recon_batch,
                                        storer=storer, distribution=self.rec_dist)
        mi_loss = (logqz_condx - logqz).mean()
        tc_loss = (logqz - logqz_prodmarginals).mean()
        dw_kl_loss = (logqz_prodmarginals - logpz).mean()

        anneal_rec = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if is_train else 1)

        # total loss
        loss = rec_loss + anneal_rec * (self.alpha * mi_loss + self.beta * tc_loss + self.gamma * dw_kl_loss)

        if storer is not None:
            storer['loss'].append(loss.item())
            storer['mi_loss'].append(mi_loss.item())
            storer['tc_loss'].append(tc_loss.item())
            storer['dw_kl_loss'].append(dw_kl_loss.item())

            dw_kl_loss_vec = (logqz_prodmarginals - logpz)
            for i in range(latent_dist.size(1)):
                storer['dw_kl_loss_' + str(i)].append(dw_kl_loss_vec[i].item())

        return loss


def _minibatch_weighted_sampling(latent_dist, latent_sample, data_size):
    """
        Estimates log q(z) and the log (product of marginals of q(z_j)) with minibatch
        weighted sampling.

        Parameters
        ----------
        latent_dist : torch.Tensor
            Mean and logvar of the normal distribution. Shape (batch_size, latent_dim, 2)

        latent_sample: torch.Tensor
            sample from the latent dimension using the reparameterisation trick
            shape : (batch_size, latent_dim).

        data_size : int
            Number of data in the training set

        References :
           [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
           autoencoders." Advances in Neural Information Processing Systems. 2018.
        """
    batch_size = latent_dist.size(0)

    _logqz = log_density_normal(latent_sample, latent_dist,
                                batch_size, return_matrix=True)
    logqz_prodmarginals = (torch.logsumexp(_logqz, dim=1, keepdim=False) -
                           math.log(batch_size * data_size)).sum(dim=1)
    logqz = torch.logsumexp(_logqz.sum(2), dim=1, keepdim=False) \
        - math.log(batch_size * data_size)

    return logqz, logqz_prodmarginals


def _minibatch_stratified_sampling(latent_dist, latent_sample, data_size):
    """
        Estimates log q(z) and the log (product of marginals of q(z_j)) with minibatch
        stratified sampling.

        Parameters
        ----------
        latent_dist : torch.Tensor
            Mean and logvar of the normal distribution. Shape (batch_size, latent_dim, 2)

        latent_sample: torch.Tensor
            sample from the latent dimension using the reparameterisation trick
            shape : (batch_size, latent_dim).

        data_size : int
            Number of data in the training set

        References :
           [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
           autoencoders." Advances in Neural Information Processing Systems. 2018.
        """
    batch_size = latent_dist.size(0)

    _logqz = log_density_normal(latent_sample, latent_dist,
                                batch_size, return_matrix=True)
    logiw_matrix = log_importance_weight_matrix(batch_size, data_size).to(latent_dist.device)
    logqz = torch.logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
    logqz_prodmarginals = torch.logsumexp(logiw_matrix.view(batch_size, batch_size, 1) +
                                          _logqz, dim=1, keepdim=False).sum(1)

    return logqz, logqz_prodmarginals


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


def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed
