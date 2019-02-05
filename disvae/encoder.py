import numpy as np

import torch
from torch import nn


class EncoderBetaB(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10,
                 device=torch.device("cpu")):
        r"""Encoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        device : torch.device
            Device on which to run the code.

        Refernces:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(EncoderBetaB, self).__init__()

        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels * 2, kernel_size, kernel_size)
        n_chan = self.img_size[0]

        self.img_size = img_size

        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels * 2, kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(hid_channels * 2, hid_channels * 2, kernel_size, **cnn_kwargs)
        if self.img_size[1:] == (64, 64):
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)

        self.mu_gen = nn.Linear(hidden_dim, latent_dim)
        self.log_var_gen = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        batch_size = x.size(0)

        x = torch.relu(self.conv1(x))
        if self.img_size[1:] == (64, 64):
            x = torch.relu(self.conv_64(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        x = x.view((batch_size, -1))
        x = torch.relu(self.lin1(x))

        mu = self.mu_gen(x)
        log_var = self.log_var_gen(x)

        return mu, log_var
