import torch
from torch import nn, optim
from torch.nn import functional as F

from disvae.initialization import weights_init


class VAE(nn.Module):
    def __init__(self, img_size, encoder, decoder, latent_dim,
                 device=torch.device("cpu")):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        device : torch.device
            Device on which to run the code.
        """
        super(VAE, self).__init__()

        if img_size[1:] not in [(32, 32), (64, 64)]:
            raise RuntimeError("{} sized images not supported. Only (None, 32, 32) and (None, 64, 64) supported. Build your own architecture or reshape images!".format(img_size))

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.device = device
        self.is_color = self.img_size[0] > 1
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder = encoder(img_size, self.latent_dim, self.device)
        self.decoder = decoder(img_size, self.latent_dim, self.device)

        self.reset_parameters()

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct = self.decoder(latent_sample)
        if self.is_color:
            reconstruct = reconstruct * 255
        return reconstruct, latent_dist

    def reset_parameters(self):
        self.apply(weights_init)
