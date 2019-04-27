import os
import logging
import math
from functools import reduce
from collections import defaultdict
import json
from timeit import default_timer

from tqdm import trange, tqdm
import numpy as np
import torch

from disvae.models.losses import get_loss_f
from disvae.utils.math import log_density_gaussian
from disvae.utils.modelIO import save_metadata

TEST_LOSSES_FILE = "test_losses.log"
METRICS_FILENAME = "metrics.log"
METRIC_HELPERS_FILE = "metric_helpers.pth"


class Evaluator:
    """
    Class to handle training of model.

    Parameters
    ----------
    model: disvae.vae.VAE

    loss_f: disvae.models.BaseLoss
        Loss function.

    device: torch.device, optional
        Device on which to run the code.

    logger: logging.Logger, optional
        Logger.

    save_dir : str, optional
        Directory for saving logs.

    is_progress_bar: bool, optional
        Whether to use a progress bar for training.
    """

    def __init__(self, model, loss_f,
                 device=torch.device("cpu"),
                 logger=logging.getLogger(__name__),
                 save_dir="results",
                 is_progress_bar=True):

        self.device = device
        self.loss_f = loss_f
        self.model = model.to(self.device)
        self.logger = logger
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger.info("Testing Device: {}".format(self.device))

    def __call__(self, data_loader, is_metrics=False, is_losses=True):
        """Compute all test losses.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        is_metrics: bool, optional
            Whether to compute and store the disentangling metrics.

        is_losses: bool, optional
            Whether to compute and store the test losses.
        """
        start = default_timer()
        is_still_training = self.model.training
        self.model.eval()

        metric, losses = None, None
        if is_metrics:
            self.logger.info('Computing metrics...')
            metrics = self.compute_metrics(data_loader)
            self.logger.info('Losses: {}'.format(metrics))
            save_metadata(metrics, self.save_dir, filename=METRICS_FILENAME)

        if is_losses:
            self.logger.info('Computing losses...')
            losses = self.compute_losses(data_loader)
            self.logger.info('Losses: {}'.format(losses))
            save_metadata(losses, self.save_dir, filename=TEST_LOSSES_FILE)

        if is_still_training:
            self.model.train()

        self.logger.info('Finished evaluating after {:.1f} min.'.format((default_timer() - start) / 60))

        return metric, losses

    def compute_losses(self, dataloader):
        """Compute all test losses.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        """
        storer = defaultdict(list)
        for data, _ in tqdm(dataloader, leave=False, disable=not self.is_progress_bar):
            data = data.to(self.device)

            try:
                recon_batch, latent_dist, latent_sample = self.model(data)
                _ = self.loss_f(data, recon_batch, latent_dist, self.model.training,
                                storer, latent_sample=latent_sample)
            except ValueError:
                # for losses that use multiple optimizers (e.g. Factor)
                _ = self.loss_f.call_optimize(data, self.model, None, storer)

            losses = {k: sum(v) / len(dataloader) for k, v in storer.items()}
            return losses

    def compute_metrics(self, dataloader):
        """Compute all the metrics.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        """
        try:
            lat_sizes = dataloader.dataset.lat_sizes
            lat_names = dataloader.dataset.lat_names
        except AttributeError:
            raise ValueError("Dataset needs to have known true factors of variations to compute the metric. This does not seem to be the case for {}".format(type(dataloader.__dict__["dataset"]).__name__))

        self.logger.info("Computing the empirical distribution q(z|x).")
        samples_zCx, params_zCx = self._compute_q_zCx(dataloader)
        len_dataset, latent_dim = samples_zCx.shape

        self.logger.info("Estimating the marginal entropy.")
        # marginal entropy H(z_j)
        H_z = self._estimate_latent_entropies(samples_zCx, params_zCx)

        # conditional entropy H(z|v)
        samples_zCx = samples_zCx.view(*lat_sizes, latent_dim)
        params_zCx = tuple(p.view(*lat_sizes, latent_dim) for p in params_zCx)
        H_zCv = self._estimate_H_zCv(samples_zCx, params_zCx, lat_sizes, lat_names)

        H_z = H_z.cpu()
        H_zCv = H_zCv.cpu()

        # I[z_j;v_k] = E[log \sum_x q(z_j|x)p(x|v_k)] + H[z_j] = - H[z_j|v_k] + H[z_j]
        mut_info = - H_zCv + H_z
        sorted_mut_info = torch.sort(mut_info, dim=1, descending=True)[0].clamp(min=0)

        metric_helpers = {'marginal_entropies': H_z, 'cond_entropies': H_zCv}
        mig = self._mutual_information_gap(sorted_mut_info, lat_sizes, storer=metric_helpers)
        aam = self._axis_aligned_metric(sorted_mut_info, storer=metric_helpers)

        metrics = {'MIG': mig.item(), 'AAM': aam.item()}
        torch.save(metric_helpers, os.path.join(self.save_dir, METRIC_HELPERS_FILE))

        return metrics

    def _mutual_information_gap(self, sorted_mut_info, lat_sizes, storer=None):
        """Compute the mutual information gap as in [1].

        References
        ----------
           [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
           autoencoders." Advances in Neural Information Processing Systems. 2018.
        """
        # difference between the largest and second largest mutual info
        delta_mut_info = sorted_mut_info[:, 0] - sorted_mut_info[:, 1]
        # NOTE: currently only works if balanced dataset for every factor of variation
        # then H(v_k) = - |V_k|/|V_k| log(1/|V_k|) = log(|V_k|)
        H_v = torch.from_numpy(lat_sizes).float().log()
        mig_k = delta_mut_info / H_v
        mig = mig_k.mean()  # mean over factor of variations

        if storer is not None:
            storer["mig_k"] = mig_k
            storer["mig"] = mig

        return mig

    def _axis_aligned_metric(self, sorted_mut_info, storer=None):
        """Compute the proposed axis aligned metrics."""
        numerator = (sorted_mut_info[:, 0] - sorted_mut_info[:, 1:].sum(dim=1)).clamp(min=0)
        aam_k = numerator / sorted_mut_info[:, 0]
        aam_k[torch.isnan(aam_k)] = 0
        aam = aam_k.mean()  # mean over factor of variations

        if storer is not None:
            storer["aam_k"] = aam_k
            storer["aam"] = aam

        return aam

    def _compute_q_zCx(self, dataloader):
        """Compute the empiricall disitribution of q(z|x).

        Parameter
        ---------
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
        latent_dim = self.model.latent_dim
        n_suff_stat = 2

        q_zCx = torch.zeros(len_dataset, latent_dim, n_suff_stat, device=self.device)

        n = 0
        with torch.no_grad():
            for x, label in dataloader:
                batch_size = x.size(0)
                idcs = slice(n, n + batch_size)
                q_zCx[idcs, :, 0], q_zCx[idcs, :, 1] = self.model.encoder(x.to(self.device))
                n += batch_size

        params_zCX = q_zCx.unbind(-1)
        samples_zCx = self.model.reparameterize(*params_zCX)

        return samples_zCx, params_zCX

    def _estimate_latent_entropies(self, samples_zCx, params_zCX,
                                   n_samples=10000):
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

        n_samples: int, optional
            Number of samples to use to estimate the entropies.

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
        with trange(n_samples, leave=False, disable=self.is_progress_bar) as t:
            for k in range(0, n_samples, mini_batch_size):
                # log q(z_j|x) for n_samples
                idcs = slice(k, k + mini_batch_size)
                log_q_zCx = log_density_gaussian(samples_zCx[..., idcs],
                                                 mean[..., idcs],
                                                 log_var[..., idcs])
                # numerically stable log q(z_j) for n_samples:
                # log q(z_j) = -log N + logsumexp_{n=1}^N log q(z_j|x_n)
                # As we don't know q(z) we appoximate it with the monte carlo
                # expectation of q(z_j|x_n) over x. => fix a single z and look at
                # proba for every x to generate it. n_samples is not used here !
                log_q_z = -log_N + torch.logsumexp(log_q_zCx, dim=0, keepdim=False)
                # H(z_j) = E_{z_j}[- log q(z_j)]
                # mean over n_samples (i.e. dimesnion 1 because already summed over 0).
                H_z += (-log_q_z).sum(1)

                t.update(mini_batch_size)

        H_z /= n_samples

        return H_z

    def _estimate_H_zCv(self, samples_zCx, params_zCx, lat_sizes, lat_names):
        """Estimate conditional entropies :math:`H[z|v]`."""
        latent_dim = samples_zCx.size(-1)
        len_dataset = reduce((lambda x, y: x * y), lat_sizes)
        H_zCv = torch.zeros(len(lat_sizes), latent_dim, device=self.device)
        for i_fac_var, (lat_size, lat_name) in enumerate(zip(lat_sizes, lat_names)):
            idcs = [slice(None)] * len(lat_sizes)
            for i in range(lat_size):
                self.logger.info("Estimating conditional entropies for the {}th value of {}.".format(i, lat_name))
                idcs[i_fac_var] = i
                # samples from q(z,x|v)
                samples_zxCv = samples_zCx[idcs].contiguous().view(len_dataset // lat_size,
                                                                   latent_dim)
                params_zxCv = tuple(p[idcs].contiguous().view(len_dataset // lat_size, latent_dim)
                                    for p in params_zCx)

                H_zCv[i_fac_var] += self._estimate_latent_entropies(samples_zxCv, params_zxCv
                                                                    ) / lat_size
        return H_zCv
