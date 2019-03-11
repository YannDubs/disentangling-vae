import os
import argparse
import logging
import math
from timeit import default_timer

from tqdm import trange
import numpy as np
import torch

from utils.modelIO import load_model, load_metadata
from utils.datasets import get_dataloaders

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def mutual_information_gap(model, dataset, is_progress_bar=True):
    """Compute the mutual information gap as in [1].

    # TO DOC

    """
    device = get_device(model)
    dataloader = get_dataloaders(dataset, batch_size=1000, shuffle=False)
    lat_sizes = dataloader.dataset.lat_sizes
    lat_names = dataloader.dataset.lat_names

    logger.info("Computing the empirical distribution q(z|x).")
    samples_zCx, params_zCx = compute_q_zCx(model, dataloader)
    len_dataset, latent_dim = samples_zCx.shape

    logger.info("Estimating the marginal entropy.")
    # marginal entropy H(z_j)
    H_z = estimate_entropies(samples_zCx, params_zCx, is_progress_bar=is_progress_bar)

    samples_zCx = samples_zCx.view(*lat_sizes, latent_dim)
    params_zCx = tuple(p.view(*lat_sizes, latent_dim) for p in params_zCx)

    # conditional entropy H(z|v)
    H_zCv = torch.zeros(len(lat_sizes), latent_dim, device=device)
    for i_fac_var, (lat_size, lat_name) in enumerate(zip(lat_sizes, lat_names)):
        idcs = [slice(None)] * len(lat_sizes)
        for i in range(lat_size):
            logger.info("Estimating conditional entropies for the {}th value of {}.".format(i, lat_name))
            idcs[i_fac_var] = i
            # samples from q(z,x|v)
            samples_zxCv = samples_zCx[idcs].contiguous().view(len_dataset // lat_size, latent_dim)
            params_zxCv = tuple(p[idcs].contiguous().view(len_dataset // lat_size, latent_dim)
                                for p in params_zCx)

            H_zCv[i_fac_var] += estimate_entropies(samples_zxCv, params_zxCv) / lat_size

    H_z = H_z.cpu()
    H_zCv = H_zCv.cpu()

    # I[z_j;v_k] = E[log \sum_x q(z_j|x)p(x|v_k)] + H[z_j] = - H[z_j|v_k] + H[z_j]
    mut_info = - H_zCv + H_z
    mut_info = torch.sort(mut_info, dim=1, descending=True)[0].clamp(min=0)
    # difference between the largest and second largest mutual info
    delta_mut_info = mut_info[:, 0] - mut_info[:, 1]
    # NOTE: currently only works if balanced dataset for every factor of variation
    # then H(v_k) = - |V_k|/|V_k| log(1/|V_k|) = log(|V_k|)
    H_v = torch.from_numpy(lat_sizes).float().log()
    metric_per_k = delta_mut_info / H_v

    logger.info("Metric per factor variation: {}.".format(list(metric_per_k)))
    metric = metric_per_k.mean()  # mean over factor of variations

    return metric, H_z, H_zCv


def log_gaussian(x, mu, logvar):
    """Compute the log gaussian density."""
    normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
    inv_sigma = torch.exp(-0.5 * logvar)
    out = normalization - 0.5 * ((x - mu) * inv_sigma)**2
    return out


def estimate_entropies(samples_zCx, params_zCX, n_samples=10000, is_progress_bar=True):
    r"""Estimate :math:`H(z_j) = E_{q(z_j)} [-log q(z_j)] = E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]`
    using the emperical distribution of :math:`p(x)`.

    Note
    ----
    - the expectation over the emperical distributio is: :math:`q(z) = 1/N sum_{n=1}^N q(z|x_n)`.
    - we assume that q(z|x) is factorial i.e. :math:`q(z|x) = \prod_j q(z_j|x)`.
    - computes numerically stable NLL: :math:`- log q(z) = log N - logsumexp_n=1^N log q(z|x_n)`.

    Parameters
    ----------
    samples_zCx: torch.tensor
        Tensor of shape (len_dataset, latent_dim) containing a sample of
        q(z|x) for every x in the dataset.

    params_zCX: tuple of torch.Tensor
        Sufficient statistics q(z|x) for each training example. E.g. for
        gaussian (mean, log_var) each of shape : (len_dataset, latent_dim).

    n_samples: int
        Number of samples to use to estimate the entropies.

    is_progress_bar: bool
        Whether to show a progress bar.

    Return
    ------
    H_z: torch.Tensor
        Tensor of shape (latent_dim) containing the marginal entropies H(z_j)
    """
    len_dataset, latent_dim = samples_zCx.shape
    device = samples_zCx.device
    H_z = torch.zeros(latent_dim, device=device)

    # sample from p(x)
    samples_x = torch.randperm(len_dataset, device=device)[:n_samples]
    # sample from p(z|x)
    samples_zCx = samples_zCx.index_select(0, samples_x).view(latent_dim, n_samples)

    mini_batch_size = 10
    samples_zCx = samples_zCx.expand(len_dataset, latent_dim, n_samples)
    mean = params_zCX[0].unsqueeze(-1).expand(len_dataset, latent_dim, n_samples)
    log_var = params_zCX[1].unsqueeze(-1).expand(len_dataset, latent_dim, n_samples)
    log_N = math.log(len_dataset)
    with trange(n_samples, leave=False, disable=not is_progress_bar) as t:
        for k in range(0, n_samples, mini_batch_size):
            # log q(z_j|x) for n_samples
            idcs = slice(k, k + mini_batch_size)
            log_q_zCx = log_gaussian(samples_zCx[..., idcs], mean[..., idcs], log_var[..., idcs])
            # numerically stable log q(z_j) for n_samples:
            # log q(z_j) = -log N + logsumexp_{n=1}^N log q(z_j|x_n)
            # As we don't know q(z) we appoximate it with the monte carlo
            # expectation of q(z_j|x_n) over x. => fix a single z and look at
            # proba for every x to generate it. n_samples is not used here !
            log_q_z = -log_N + torch.logsumexp(log_q_zCx, dim=0, keepdim=False)
            # H(z_j) = E_{z_j}[- log q(z_j)]
            # mean over n_samples over (i.e. dimesnion 1 because already summed over 0).
            H_z += (-log_q_z).sum(1)

            t.update(mini_batch_size)

    H_z /= n_samples

    return H_z


def compute_q_zCx(model, dataloader):
    """Compute the empiricall disitribution of q(z|x).

    Parameter
    ---------
    model: disvae.vae.VAE

    dataloader: torch.utils.data.DataLoader
        Batch data iterator.

    Return
    ------
    samples_zCx: torch.tensor
        Tensor of shape (len_dataset, latent_dim) containing a sample of
        q(z|x) for every x in the dataset.

    params_zCX: tuple of torch.Tensor
        Sufficient statistics q(z|x) for each training example. E.g. for
        gaussian (mean, log_var) each of shape : (len_dataset, latent_dim).
    """
    len_dataset = len(dataloader.dataset)
    device = get_device(model)
    batch_size = dataloader.batch_size
    latent_dim = model.latent_dim
    n_suff_stat = 2
    is_training = model.training
    model.eval()

    q_zCx = torch.zeros(len_dataset, latent_dim, n_suff_stat, device=device)

    with torch.no_grad():
        for i, (x, label) in enumerate(dataloader):
            idcs = slice(i * batch_size, (i + 1) * batch_size)
            q_zCx[idcs, :, 0], q_zCx[idcs, :, 1] = model.encoder(x.to(device))

    params_zCX = q_zCx.unbind(-1)
    samples_zCx = model.reparameterize(*params_zCX)

    if is_training:
        model.train()

    return samples_zCx, params_zCX


# SHOULD BE IN HELPERS.PY
def get_device(model):
    """Return the device on which a model is."""
    return next(model.parameters()).device


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch metrics for disentangling.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dir', help="Directory where the model is and where to save the metrics.")
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--no-progress-bar', action='store_true', default=False,
                        help='Disables progress bar.')
    log_levels = ['critical', 'error', 'warning', 'info', 'debug']
    parser.add_argument('-L', '--log-level', help="Logging levels.", default="info",
                        choices=log_levels)
    # add choices : progress bar / logging
    args = parser.parse_args()

    logger.setLevel(args.log_level.upper())
    model = load_model(args.dir, is_gpu=not args.no_cuda)
    metadata = load_metadata(args.dir)

    metric, H_z, H_zCv = mutual_information_gap(model, metadata["dataset"], is_progress_bar=not args.no_progress_bar)

    torch.save({'metric': metric, 'marginal_entropies': H_z, 'cond_entropies': H_zCv},
               os.path.join(args.dir, 'disentanglement_metric.pth'))

    logger.info('MIG: {:.3f}'.format(metric))
