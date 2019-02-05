import torch
from torch import nn
import torch.nn.functional as F


class EncoderMini(nn.Module):
    def __init__(self, img_size, h_dim=100, z_dim=10):
        """Test Encoder

        Args:
            x_dim (int): dimensionality of input.
            h_dim (int): dimensionality of hidden.
            z_dim (int): dimensionality of latent output.
        """
        super(EncoderMini, self).__init__()

        _, h, w = img_size
        self.lin_dim = h * w

        self.fc1 = nn.Linear(self.lin_dim, h_dim)
        self.mu_gen = nn.Linear(h_dim, z_dim)
        # make the output to be the logarithm
        # i.e will have to take the exponent
        # which forces variance to be positive
        # not that this is the diagonal of the covariance
        self.log_var_gen = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        x = x.view(-1, self.lin_dim)
        x = F.relu(self.fc1(x))
        mu = self.mu_gen(x)
        log_var = self.log_var_gen(x)
        return mu, log_var


class EncoderBetaB(nn.Module):
    def __init__(self,
                 img_size,
                 n_chan=3,
                 z_dim=10,
                 hid_channels=32,
                 kernel_size=4):
        r"""Encoder of the model proposed in [1].

        Args:
            x_dim (int): dimensionality of input.
            z_dim (int): dimensionality of latent output.

        Refernces:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(EncoderBetaB, self).__init__()

        n_chan, h, w = img_size

        assert h == w and w in [32, 64], "only tested with 32*32 or 64*64 images"

        self.z_dim = z_dim
        self.hid_channels = hid_channels
        self.cnn_n_out_chan = (w // (2 ** 4))
        self.h_dim = self.hid_channels * self.cnn_n_out_chan**2

        cnn_args = [hid_channels, hid_channels, kernel_size]
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(*cnn_args, **cnn_kwargs)
        self.conv3 = nn.Conv2d(*cnn_args, **cnn_kwargs)
        self.conv4 = nn.Conv2d(*cnn_args, **cnn_kwargs)

        self.lin1 = nn.Linear(self.h_dim, self.h_dim // 2)
        self.lin2 = nn.Linear(self.h_dim // 2, self.h_dim // 2)

        self.mu_gen = nn.Linear(self.h_dim // 2, self.z_dim)
        self.log_var_gen = nn.Linear(self.h_dim // 2, self.z_dim)

    def forward(self, x):
        batch_size = x.size(0)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))

        x = x.view((batch_size, -1))
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))

        mu = self.mu_gen(x)
        log_var = self.log_var_gen(x)

        return mu, log_var
