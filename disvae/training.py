import imageio
import logging

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.utils import make_grid

from losses import get_loss_f

logger = logging.getLogger(__name__)


class Trainer():
    def __init__(self, model, optimizer,
                 latent_dim=10,
                 capacity=None,
                 print_loss_every=50,
                 record_loss_every=5,
                 device=torch.device("cpu"),
                 log_level=None,
                 loss):
        """
        Class to handle training of model.

        Parameters
        ----------
        model : disvae.vae.VAE

        optimizer : torch.optim.Optimizer

        latent_dim : int
            Dimensionality of latent output.

        capacity : tuple (float, float, int, float)
            Tuple containing (min_capacity, max_capacity, num_iters, gamma_z).
            Parameters to control the capacity of the continuous latent
            channels.

        print_loss_every : int
            Frequency with which loss is printed during training.

        record_loss_every : int
            Frequency with which loss is recorded during training.

        device : torch.device
            Device on which to run the code.

        log_level : {'critical', 'error', 'warning', 'info', 'debug'}
            Logging levels.

        loss : {"betaH", "betaB", "factorising", "factorising", "batchTC"}
            Type of VAE loss to use.
        """
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.print_loss_every = print_loss_every
        self.record_loss_every = record_loss_every
        self.capacity = capacity
        self.dec_dist = "bernoulli" if self.img_size[0] == 1 else "gaussian"
        self.loss_f = get_loss_f(name)

        # Initialize attributes
        self.num_steps = 0
        self.batch_size = None
        self.losses = {'loss': [],
                       'recon_loss': [],
                       'kl_loss': []}

        # For every dimension of continuous latent variables
        for i in range(latent_dim):
            self.losses['kl_loss_' + str(i)] = []

        self.logger = logger
        if log_level is not None:
            self.logger.setLevel(log_level.upper())

    def train(self, data_loader, epochs=10, save_training_gif=None):
        """
        Trains the model.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader

        epochs : int
            Number of epochs to train the model for.

        save_training_gif : None or tuple (string, Visualizer instance)
            If not None, will use visualizer object to create image of samples
            after every epoch and will save gif of these at location specified
            by string. Note that string should end with '.gif'.
        """
        if save_training_gif is not None:
            training_progress_images = []

        self.batch_size = data_loader.batch_size
        self.model.train()
        for epoch in range(epochs):
            mean_epoch_loss = self._train_epoch(data_loader)
            avg_loss = self.batch_size * self.model.num_pixels * mean_epoch_loss
            self.logger.info('Epoch: {} Average loss: {:.2f}'.format(epoch + 1,
                                                                     avg_loss))

            if save_training_gif is not None:
                # Generate batch of images and convert to grid
                viz = save_training_gif[1]
                viz.save_images = False
                img_grid = viz.all_latent_traversals(size=10)
                # Convert to numpy and transpose axes to fit imageio convention
                # i.e. (width, height, channels)
                img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
                # Add image grid to training progress
                training_progress_images.append(img_grid)

        if save_training_gif is not None:
            imageio.mimsave(save_training_gif[0], training_progress_images,
                            fps=24)

    def _train_epoch(self, data_loader):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
        """
        epoch_loss = 0.
        print_every_loss = 0.  # Keeps track of loss to print every
        for batch_idx, (data, label) in enumerate(data_loader):
            iter_loss = self._train_iteration(data)
            epoch_loss += iter_loss
            print_every_loss += iter_loss
            # Print loss info every self.print_loss_every iteration
            if batch_idx % self.print_loss_every == 0:
                if batch_idx == 0:
                    mean_loss = print_every_loss
                else:
                    mean_loss = print_every_loss / self.print_loss_every
                self.logger.info('{}/{}\tLoss: {:.3f}'.format(batch_idx * len(data),
                                                              len(data_loader.dataset),
                                                              self.model.num_pixels * mean_loss))
                print_every_loss = 0.
        # Return mean epoch loss
        return epoch_loss / len(data_loader.dataset)

    def _train_iteration(self, data):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).
        """
        self.num_steps += 1

        self.optimizer.zero_grad()
        data = data.to(self.device)
        recon_batch, latent_dist = self.model(data)
        loss = self._loss_function(data, recon_batch, latent_dist)
        # make loss independent of number of pixels
        loss = loss / self.model.num_pixels
        loss.backward()
        self.optimizer.step()

        train_loss = loss.item()
        return train_loss
