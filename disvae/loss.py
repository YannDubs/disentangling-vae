import torch
from torch import nn
import torch.nn.functional as F


def get_loss(loss, **kwargs):
    loss = loss.lower()
    if loss == "vae":
        return lambda y_pred, y_true: vae_loss(y_pred[0], y_pred[1], y_pred[2],
                                               y_true)
    elif loss == "b-vae":
        return lambda y_pred, y_true: vae_loss(y_pred[0], y_pred[1], y_pred[2],
                                               y_true, **kwargs)


def reconstrunction_loss(x_hat, x, distribution="bernoulli"):
    batch_size = x_hat.size(0)
    if distribution == "bernoulli":
        # black or white image => use sigmoid for each pixel
        rec_loss = F.binary_cross_entropy_with_logits(x_hat, x,
                                                      reduction='sum').div(batch_size)
    else:
        raise ValueError("Distribution {} not implemented.".format(distribution))

    return rec_loss


def kl_divergence(mu, log_var):
    batch_size = mu.size(0)
    # closed form solution for gaussian prior and posterior
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()
                              ).div(batch_size)
    return kl_div


def vae_loss(x_hat, mu, log_var, x, beta=1, distribution="bernoulli"):
    """Compute the ELBO loss"""
    rec_loss = reconstrunction_loss(x_hat, x, distribution=distribution)
    kl_div = kl_divergence(mu, log_var)
    vae_loss = rec_loss + beta * kl_div
    return vae_loss
