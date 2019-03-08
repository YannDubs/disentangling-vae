import torch
from viz.latent_traversals import LatentTraverser
from scipy import stats
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
from torchvision import transforms
import numpy as np
import csv
import os
import json
from PIL import Image, ImageFont, ImageDraw
from utils.datasets import get_background


class Visualizer():
    def __init__(self, model, dataset, model_dir=None, save_images=True):
        """
        Visualizer is used to generate images of samples, reconstructions,
        latent traversals and so on of the trained model.

        Parameters
        ----------
        model : disvae.vae.VAE

        model_dir : str
            The directory that the model is saved to.

        save_images : bool
            Whether to save images or return a tensor.

        dataset : str
            Name of the dataset.
        """
        self.model = model
        self.device = next(self.model.parameters()).device
        self.latent_traverser = LatentTraverser(self.model.latent_dim)
        self.save_images = save_images
        self.model_dir = model_dir
        self.dataset = dataset

    def show_disentanglement_fig2(self, latent_sweep_data, heat_map_data):
        """ Reproduce Figure 2 from Burgess https://arxiv.org/pdf/1804.03599.pdf
            TODO: STILL TO BE IMPLEMENTED
        """
        pass
        # avg_kl_list_descending = self.recon_and_traverse_all(data=latent_sweep_data)
        # self.generate_heat_maps(data=heat_map_data, latent_order=avg_kl_list_descending)

    def generate_heat_maps(self, data, latent_order=None, heat_map_size=(32, 32), filename='imgs/heatmap.png'):
        """
        Generates heat maps of the mean of each latent dimension in the model. The spites are
        assumed to be in order, therefore no information about (x,y) positions is required.

        Parameters
        ----------
        data : torch.Tensor
            Data to be used to generate the heat maps. Shape (N, C, H, W)

        heat_map_size : tuple of ints
            Size of grid on which heat map will be plotted.

        filename : String
            The name of the file you want the heat maps to be saved as.
            Note that a suffix of -* will be used to denote the latent dimension number.
        """
        # Plot reconstructions in test mode, i.e. without sampling from latent
        self.model.eval()
        # Pass data through VAE to obtain reconstruction
        with torch.no_grad():
            input_data = data.to(self.device)
            _, latent_dist, _ = self.model(input_data)
            means = latent_dist[1]

            heat_map_height = heat_map_size[0]
            heat_map_width = heat_map_size[1]
            num_latent_dims = means.shape[1]

            heat_map = torch.zeros([num_latent_dims, 1, heat_map_height, heat_map_width], dtype=torch.int32)

            for latent_dim in range(num_latent_dims):
                for y_posn in range(heat_map_width):
                    for x_posn in range(heat_map_height):
                        heat_map[latent_dim, 0, x_posn, y_posn] = means[heat_map_width * y_posn + x_posn, latent_dim]

            if latent_order is not None:
                # Reorder latent samples by average KL
                heat_map = [
                    latent_sample for _, latent_sample in sorted(zip(latent_order, heat_map), reverse=True)
                ]
                heat_map = torch.stack(heat_map)

            if self.save_images:
                save_image(heat_map.data, filename=filename, nrow=1, pad_value=(1 - get_background(self.dataset)))
            else:
                return make_grid(heat_map.data, nrow=latent_dim, pad_value=(1 - get_background(self.dataset)))

    def recon_and_traverse_all(self, data, filename='imgs/recon_and_traverse.png'):
        """
        Take 8 sample images, run them through the decoder, obtain the mean latent
        space vector. With this as the initialisation, traverse each dimension one
        by one to observe what each latent dimension encodes.

        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)

        filename : string
            Name of file in which results are stored.
        """
        # Plot reconstructions in test mode, i.e. without sampling from latent
        self.model.eval()
        # Pass data through VAE to obtain reconstruction
        with torch.no_grad():
            input_data = data.to(self.device)
            _, latent_dist, _ = self.model(input_data)
        means = latent_dist[1]
        return self.all_latent_traversals(sample_latent_space=means, filename=filename)

    def reconstructions(self, data, size=(8, 8), filename='imgs/recon.png'):
        """
        Generates reconstructions of data through the model.

        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)

        size : tuple of ints
            Size of grid on which reconstructions will be plotted. The number
            of rows should be even, so that upper half contains true data and
            bottom half contains reconstructions

        filename : string
            Name of file in which results are stored.
        """
        # Plot reconstructions in test mode, i.e. without sampling from latent
        self.model.eval()
        # Pass data through VAE to obtain reconstruction
        with torch.no_grad():
            input_data = data.to(self.device)
            recon_data, _, _ = self.model(input_data)
        self.model.train()

        # Upper half of plot will contain data, bottom half will contain
        # reconstructions of the original image
        num_images = size[0]
        originals = input_data.cpu()
        reconstructions = recon_data.view(-1, *self.model.img_size).cpu()
        # If there are fewer examples given than spaces available in grid,
        # augment with blank images
        num_examples = originals.size()[0]
        if num_images > num_examples:
            blank_images = torch.zeros((num_images - num_examples,) + originals.size()[1:])
            originals = torch.cat([originals, blank_images])
            reconstructions = torch.cat([reconstructions, blank_images])

        # Concatenate images and reconstructions
        comparison = torch.cat([originals, reconstructions])

        if self.save_images:
            save_image(comparison.data, filename, nrow=size[0], pad_value=(1 - get_background(self.dataset)))
        else:
            return make_grid(comparison.data, nrow=size[0], pad_value=(1 - get_background(self.dataset)))

    def samples(self, size=(8, 8), filename='imgs/samples.png'):
        """
        Generates samples from learned distribution by sampling prior and
        decoding.

        size : tuple of ints
        """
        # Get prior samples from latent distribution
        cached_sample_prior = self.latent_traverser.sample_prior
        self.latent_traverser.sample_prior = True
        prior_samples = self.latent_traverser.traverse_grid(size=size)
        self.latent_traverser.sample_prior = cached_sample_prior

        # Map samples through decoder
        generated = self._decode_latents(prior_samples)

        if self.save_images:
            save_image(generated.data, filename, nrow=size[1], pad_value=(1 - get_background(self.dataset)))
        else:
            return make_grid(generated.data, nrow=size[1], pad_value=(1 - get_background(self.dataset)))

    def latent_traversal_line(self, idx=None, size=8,
                              filename='imgs/traversal_line.png'):
        """
        Generates an image traversal through a latent dimension.

        Parameters
        ----------
        See viz.latent_traversals.LatentTraverser.traverse_line for parameter
        documentation.
        """
        # Generate latent traversal
        latent_samples = self.latent_traverser.traverse_line(idx=idx,
                                                             size=size)

        # Map samples through decoder
        generated = self._decode_latents(latent_samples)

        if self.save_images:
            save_image(generated.data, filename, nrow=size, pad_value=(1 - get_background(self.dataset)))
        else:
            return make_grid(generated.data, nrow=size, pad_value=(1 - get_background(self.dataset)))

    def latent_traversal_grid(self, idx=None, axis=None, size=(5, 5),
                              filename='imgs/traversal_grid.png'):
        """
        Generates a grid of image traversals through two latent dimensions.

        Parameters
        ----------
        See viz.latent_traversals.LatentTraverser.traverse_grid for parameter
        documentation.
        """
        # Generate latent traversal
        latent_samples = self.latent_traverser.traverse_grid(idx=idx,
                                                             axis=axis,
                                                             size=size)

        # Map samples through decoder
        generated = self._decode_latents(latent_samples)

        if self.save_images:
            save_image(generated.data, filename, nrow=size[1], pad_value=(1 - get_background(self.dataset)))
        else:
            return make_grid(generated.data, nrow=size[1], pad_value=(1 - get_background(self.dataset)))

    def all_latent_traversals(self, sample_latent_space=None, size=8, filename='imgs/all_traversals.png'):
        """
        Traverses all latent dimensions one by one and plots a grid of images
        where each row corresponds to a latent traversal of one latent
        dimension.

        Parameters
        ----------
        sample_latent_space : torch.Tensor or None
            The latent space of a sample which has been processed by the encoder.
            The dimensions are (size, num_latent_dims)

        size : int
            Number of samples for each latent traversal.
        """
        latent_samples = []
        avg_kl_list = read_avg_kl_from_file(os.path.join(self.model_dir, 'kl_data.log'))
        # Perform line traversal of every latent
        for idx in range(self.model.latent_dim):
            latent_samples.append(self.latent_traverser.traverse_line(idx=idx,
                                                                      size=size,
                                                                      sample_latent_space=sample_latent_space))
        latent_samples = [
            latent_sample for _, latent_sample in sorted(zip(avg_kl_list, latent_samples), reverse=True)
        ]
        sorted_avg_kl_list = [
            round(float(avg_kl_sample), 3) for avg_kl_sample, _ in sorted(zip(avg_kl_list, latent_samples), reverse=True)
        ]

        # Decode samples
        generated = self._decode_latents(torch.cat(latent_samples, dim=0))
        traversal_images_with_text = add_kl_labels(generated, size, sorted_avg_kl_list, self.dataset)

        if self.save_images:
            traversal_images_with_text.save(filename)
            return avg_kl_list
        else:
            return make_grid(generated.data, nrow=size, pad_value=(1 - get_background(self.dataset)))

    def _decode_latents(self, latent_samples):
        """
        Decodes latent samples into images.

        Parameters
        ----------
        latent_samples : torch.autograd.Variable
            Samples from latent distribution. Shape (N, L) where L is dimension
            of latent distribution.
        """
        latent_samples = latent_samples.to(self.device)
        return self.model.decoder(latent_samples).cpu()


def add_kl_labels(generated, size, sorted_avg_kl_list, dataset):
    """ Adds the average KL per latent dimension as a label next to the relevant row as in
        figure 2 of Burgress et al.
    """
    # Convert tensor to PIL Image
    tensor = make_grid(generated.data, nrow=size, pad_value=(1 - get_background(dataset)))
    all_traversal_im = transforms.ToPILImage()(tensor)
    # Resize image
    new_width = int(1.3 * all_traversal_im.width)
    new_size = (all_traversal_im.height, new_width)
    traversal_images_with_text = Image.new("RGB", new_size, color='white')
    traversal_images_with_text.paste(all_traversal_im, (0, 0))
    # Add KL text alongside each row
    draw = ImageDraw.Draw(traversal_images_with_text)
    for latent_idx, latent_dim in enumerate(sorted_avg_kl_list):
        draw.text(xy=(int(0.825 * traversal_images_with_text.width),
                        int((latent_idx / len(sorted_avg_kl_list) + \
                            1 / (2 * len(sorted_avg_kl_list))) * all_traversal_im.height)),
                    text="KL = {}".format(latent_dim),
                    fill=(0,0,0))
    return traversal_images_with_text

def reorder_img(orig_img, reorder, by_row=True, img_size=(3, 32, 32), padding=2):
    """
    Reorders rows or columns of an image grid.

    Parameters
    ----------
    orig_img : torch.Tensor
        Original image. Shape (channels, width, height)

    reorder : list of ints
        List corresponding to desired permutation of rows or columns

    by_row : bool
        If True reorders rows, otherwise reorders columns

    img_size : tuple of ints
        Image size following pytorch convention

    padding : int
        Number of pixels used to pad in torchvision.utils.make_grid
    """
    reordered_img = torch.zeros(orig_img.size())
    _, height, width = img_size

    for new_idx, old_idx in enumerate(reorder):
        if by_row:
            start_pix_new = new_idx * (padding + height) + padding
            start_pix_old = old_idx * (padding + height) + padding
            reordered_img[:, start_pix_new:start_pix_new + height, :] = orig_img[:, start_pix_old:start_pix_old + height, :]
        else:
            start_pix_new = new_idx * (padding + width) + padding
            start_pix_old = old_idx * (padding + width) + padding
            reordered_img[:, :, start_pix_new:start_pix_new + width] = orig_img[:, :, start_pix_old:start_pix_old + width]

    return reordered_img


def read_avg_kl_from_file(log_file_path):
    """ Read the average KL per latent dimension at the final stage of training from the log file.
    """
    with open(log_file_path, 'r') as f:
        last_line = list(csv.reader(f))[-1:][0]
        return last_line[1:]
