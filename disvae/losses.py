"""
Module containing all vae losses.
"""
import abc

from torch.nn import functional as F


def get_loss_f(name, **kwargs):
    """Return the correct loss function."""
    if name == "betaH":
        return BetaHLoss(**kwargs)
    elif name == "betaB":
        return BetaBLoss(**kwargs)
    elif name == "factorising":
        raise ValueError("{} loss not yet implemented".format(name))
        # return FactorLoss(**kwargs)
    elif name == "batchTC":
        raise ValueError("{} loss not yet implemented".format(name))
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

        latent_dist : tuple
            sufficient statistics of the latent dimension. E.g. for gaussian
            (mean, logvar) both of shape : (batch_size, latent_dim).

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
        mean, logvar = latent_dist
        kl_loss = _kl_normal_loss(mean, logvar, storer)
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

    Refernces:
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
        mean, logvar = latent_dist
        kl_loss = _kl_normal_loss(mean, logvar, storer)

        # linearly increasing C
        C_delta = (self.C_max - self.C_min)
        C = self.C_min + C_delta * self.n_train_steps / self.C_n_interp

        loss = rec_loss + self.gamma * (kl_loss - C).abs()

        return loss


def _reconstruction_loss(self, data, recon_data, is_color):
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


def _kl_normal_loss(self, mean, logvar, storer=None):
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
            storer['kl_loss_' + str(i)].append(latent_kl[i].item())

    return total_kl
