import torch
from torch import nn
import torch.nn.functional as F


class DecoderMini(nn.Module):
    def __init__(self, img_size, h_dim=100, z_dim=10):
        super(DecoderMini, self).__init__()

        _, h, w = img_size
        self.lin_dim = h * w

        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, self.lin_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        # black and white MNIST => sigmoid for each pixel
        x = torch.sigmoid(x)
        return x


class DecoderBetaB(nn.Module):
    def __init__(self,
                 img_size,
                 n_chan=3,
                 z_dim=10,
                 hid_channels=32,
                 kernel_dim=4):
        r"""Encoder of the model proposed in [1].

        Args:
            x_dim (int): dimensionality of input.
            z_dim (int): dimensionality of latent output.

        Refernces:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(DecoderBetaB, self).__init__()

        n_chan, h, w = img_size
        assert h == w and w in [32, 64], "only tested with 32*32 or 64*64 images"

        self.z_dim = z_dim
        self.hid_channels = hid_channels
        self.cnn_n_out_chan = (w // (2 ** 4))
        self.h_dim = self.hid_channels * self.cnn_n_out_chan**2

        self.lin1 = nn.Linear(self.z_dim, self.h_dim // 2)
        self.lin2 = nn.Linear(self.h_dim // 2, self.h_dim // 2)
        self.lin3 = nn.Linear(self.h_dim // 2, self.h_dim)

        cnn_args = [hid_channels, hid_channels, kernel_dim]
        cnn_kwargs = dict(stride=2, padding=1)
        self.convT1 = nn.ConvTranspose2d(*cnn_args, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(*cnn_args, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(*cnn_args, **cnn_kwargs)
        self.convT4 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_dim, **cnn_kwargs)

    def forward(self, x):
        batch_size = x.size(0)

        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view((batch_size, self.hid_channels, self.cnn_n_out_chan, self.cnn_n_out_chan))

        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        x = torch.relu(self.convT3(x))
        x = self.convT4(x)

        return x
