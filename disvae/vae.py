import torch
from torch import nn

from disvae.initialization import weights_init


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.reset_parameters()

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # square root in exponent => std
        eps = torch.randn_like(std)
        z = std * eps + mu
        return z

    def forward(self, x):
        # make image linear (i.e vector form)
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z).view(x.size())
        return x_hat, mu, log_var

    def reset_parameters(self):
        self.apply(weights_init)
