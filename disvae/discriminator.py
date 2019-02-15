import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self,
                 latent_dim=10,
                 device=torch.device("cpu")):
        r"""Discriminator proposed in [1].

        Parameters
        ----------
        latent_dim : int
            Dimensionality of latent variables.

        device : torch.device
            Device on which to run the code.

        Model Architecture
        ------------
        - 6 layer multi-layer perceptron, each with 1000 hidden units
        - Leaky ReLu activations
        - Output 2 logits

        References:
            [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
            arXiv preprint arXiv:1802.05983 (2018).

        """
        super(Discriminator, self).__init__()

        # Activation parameters
        neg_slope = 0.2
        self.leaky_relu = nn.LeakyReLU(neg_slope, True)

        # Layer parameters
        self.z_dim = latent_dim
        hidden_units = 1000
        out_units = 2

        # Fully connected layers
        self.lin1 = nn.Linear(self.z_dim, hidden_units)
        self.lin2 = nn.Linear(hidden_units, hidden_units)
        self.lin3 = nn.Linear(hidden_units, hidden_units)
        self.lin4 = nn.Linear(hidden_units, hidden_units)
        self.lin5 = nn.Linear(hidden_units, hidden_units)
        self.lin6 = nn.Linear(hidden_units, out_units)

    def forward(self, z):

        # Fully connected layers with leaky ReLu activations
        z = self.leaky_relu(self.lin1(z))
        z = self.leaky_relu(self.lin2(z))
        z = self.leaky_relu(self.lin3(z))
        z = self.leaky_relu(self.lin4(z))
        z = self.leaky_relu(self.lin5(z))
        z = self.lin6(z)

        return z
