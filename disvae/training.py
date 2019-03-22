import imageio
import logging
import os
from timeit import default_timer
from collections import defaultdict

from tqdm import trange
import torch
from torch.nn import functional as F

import sys
sys.path.append("..")

from utils.modelIO import save_model
from disvae.losses import get_loss_f
from viz.visualize import Visualizer
from utils.helpers import mean


TRAIN_FILE = "train_losses.log"


class Trainer():
    def __init__(self, model, optimizer,
                 loss_type="betaB",
                 latent_dim=10,
                 loss_kwargs={},
                 device=torch.device("cpu"),
                 logger=logging.getLogger(__name__),
                 save_dir="experiments",
                 is_viz_gif=True,
                 is_progress_bar=True,
                 checkpoint_every=10,
                 dataset="mnist"):
        """
        Class to handle training of model.

        Parameters
        ----------
        model : disvae.vae.VAE

        optimizer : torch.optim.Optimizer

        latent_dim : int
            Dimensionality of latent output.

        loss_kwargs : dict.
            Additional arguments to the loss function. FOrmat need to be the
            loss specific argparse arguments.

        device : torch.device
            Device on which to run the code.

        log_level : {'critical', 'error', 'warning', 'info', 'debug'}
            Logging levels.

        loss_type : {"VAE", "betaH", "betaB", "factorising", "batchTC"}
            Type of VAE loss to use.

        save_dir : str
            Directory for saving logs.

        is_viz_gif : bool
            Whether to store a gif of samples after every epoch.

        is_progress_bar: bool
            Whether to use a progress bar for training.

        checkpoint_every: int
            Save a checkpoint of the trained model every n epoch.
        """
        self.device = device
        self.loss_type = loss_type
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.num_latent_dim = latent_dim
        loss_kwargs["device"] = device
        self.loss_f = get_loss_f(self.loss_type, kwargs_parse=loss_kwargs)
        self.save_dir = save_dir
        self.is_viz_gif = is_viz_gif
        self.is_progress_bar = is_progress_bar
        self.checkpoint_every = checkpoint_every

        self.logger = logger

        self.losses_logger = LossesLogger(os.path.join(self.save_dir, TRAIN_FILE))
        if self.is_viz_gif:
            self.vizualizer = Visualizer(model, dataset,
                                         model_dir=self.save_dir,
                                         save_images=False)

        self.logger.info("Training Device: {}".format(self.device))

    def __call__(self, data_loader, epochs=10, visualizer=None):
        """
        Trains the model.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader

        epochs : int
            Number of epochs to train the model for.
        """
        start = default_timer()

        if self.is_viz_gif:
            training_progress_images = []

        self.model.train()
        for epoch in range(epochs):
            storer = defaultdict(list)
            mean_epoch_loss = self._train_epoch(data_loader, storer, epoch)
            self.logger.info('Epoch: {} Average loss per image: {:.2f}'.format(epoch + 1,
                                                                               mean_epoch_loss))
            self.losses_logger.log(epoch, storer)

            if self.is_viz_gif:
                img_grid = self.vizualizer.all_latent_traversals(size=10)
                training_progress_images.append(img_grid)

            if epoch % self.checkpoint_every == 0:
                save_model(self.model, self.save_dir,
                           filename="model-{}.pt".format(epoch))

        if self.is_viz_gif:
            imageio.mimsave(os.path.join(self.save_dir, "training.gif"),
                            training_progress_images,
                            fps=24)

        self.model.eval()

        self.logger.info('Finished training after {:.1f} min.'.format((default_timer() - start) / 60))

    def _train_epoch(self, data_loader, storer, epoch):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader

        storer : dict
            Dictionary in which to store important variables for vizualisation.

        epoch: int
            Epoch number
        """
        epoch_loss = 0.
        kwargs = dict(desc="Epoch {}".format(epoch), leave=False,
                      disable=not self.is_progress_bar)
        with trange(len(data_loader), **kwargs) as t:
            for batch_idx, (data, label) in enumerate(data_loader):
                iter_loss = self._train_iteration(data, storer)
                epoch_loss += iter_loss

                t.set_postfix(loss=iter_loss)
                t.update()

        mean_epoch_loss = epoch_loss / len(data_loader)
        return mean_epoch_loss

    def _train_iteration(self, data, storer):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).

        storer : dict
            Dictionary in which to store important variables for vizualisation.
        """
        batch_size, channel, height, width = data.size()
        data = data.to(self.device)

        # TO-DO: clean all these if statements
        if self.loss_type == 'factor':
            loss = self.loss_f(data, self.model, self.optimizer, storer)
        else:
            recon_batch, latent_dist, latent_sample = self.model(data)
            loss_kwargs = dict()
            if self.loss_type == 'batchTC':
                loss_kwargs["latent_sample"] = latent_sample
            loss = self.loss_f(data, recon_batch, latent_dist, self.model.training, storer, **loss_kwargs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()


class LossesLogger(object):
    """ Class definition for objects to write data to log files in a
        form which is then easy to be plotted.
    """

    def __init__(self, file_path_name):
        """ Create a logger to store information for plotting. """
        if os.path.isfile(file_path_name):
            os.remove(file_path_name)

        self.logger = logging.getLogger("losses_logger")
        self.logger.setLevel(0)  # always store
        file_handler = logging.FileHandler(file_path_name)
        file_handler.setLevel(0)
        self.logger.addHandler(file_handler)

        header = ",".join(["Epoch", "Loss", "Value"])
        self.logger.debug(header)

    def log(self, epoch, losses_storer):
        """Write to the log file """
        for k, v in losses_storer.items():
            log_string = ",".join(str(item) for item in [epoch, k, mean(v)])
            self.logger.debug(log_string)
