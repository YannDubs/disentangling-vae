import numpy as np

import torch
from torch import nn


class DecoderBetaB(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10,
                 device=torch.device("cpu")):
        r"""Decoder of the model proposed in [1].

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
        super(DecoderBetaB, self).__init__()

        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels * 2, kernel_size, kernel_size)
        n_chan = self.img_size[0]

        self.img_size = img_size

        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, np.product(self.reshape))

        cnn_kwargs = dict(stride=2, padding=1)
        if self.img_size[1:] == (64, 64):
            self.convT_64 = nn.Conv2d(hid_channels * 2, hid_channels * 2, kernel_size,
                                      **cnn_kwargs)

        cnn_kwargs = dict(stride=2, padding=1)
        self.convT1 = nn.ConvTranspose2d(hid_channels * 2, hid_channels, kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs)

    def forward(self, z):
        batch_size = z.size(0)

        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))
        x = x.view(batch_size, *self.reshape)

        if self.img_size[1:] == (64, 64):
            x = torch.relu(self.convT_64(x))
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        x = torch.sigmoid(self.convT3(x))

        return x
