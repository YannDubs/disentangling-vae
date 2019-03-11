import torch
from scipy import stats
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from PIL import Image, ImageDraw

from utils.datasets import get_background
from viz.latent_traversals import LatentTraverser
from viz.viz_helpers import reorder_img, read_avg_kl_from_file, add_labels, upsample

import numpy as np
import csv
import os
import json


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

    def show_disentanglement_fig2(self, reconstruction_data, latent_sweep_data, heat_map_data, latent_order=None, heat_map_size = (32,32), filename='imgs/show_disentanglement.png',size=8, sample_latent_space=None):
        """ Reproduce Figure 2 from Burgess https://arxiv.org/pdf/1804.03599.pdf
            TODO: STILL TO BE IMPLEMENTED
        """
        # Plot reconstructions in test mode, i.e. without sampling from latent
        self.model.eval()
        # Pass data through VAE to obtain reconstruction
        with torch.no_grad():
            input_data = reconstruction_data.to(self.device)
            sample_recon, _, _ = self.model(input_data)
        self.model.train()

        # Upper half of plot will contain data, bottom half will contain
        # reconstructions of the original image
        num_images = 9
        originals = input_data.cpu()
        reconstructions = sample_recon.view(-1, *self.model.img_size).cpu()
        # Upper half of plot will contain data, bottom half will contain
        # reconstructions of the original image
        num_images = 9
        originals = input_data.cpu()
        reconstructions = sample_recon.view(-1, *self.model.img_size).cpu()
        # If there are fewer num_exampleses given than spaces available in grid,
        # augment with blank images
        num_examples = originals.size()[0]
        if num_images > num_examples:
            blank_images = torch.zeros((num_images - num_examples,) + originals.size()[1:])
            originals = torch.cat([originals, blank_images])
            reconstructions = torch.cat([reconstructions, blank_images])
        comparison = torch.cat([originals, reconstructions])

        # Function: generateheatmap

        # Plot reconstructions in test mode, i.e. without sampling from latent
        self.model.eval()
        # Pass data through VAE to obtain reconstruction
        with torch.no_grad():
            input_data = heat_map_data.to(self.device)
            sample_heat_map = self.model.sample_latent(input_data)
            # means = latent_dist[1]

            heat_map_height = heat_map_size[0]
            heat_map_width = heat_map_size[1]
            num_latent_dims = sample_heat_map.shape[1]

            heat_map = torch.zeros([num_latent_dims, 1, heat_map_height, heat_map_width])

            for latent_dim in range(num_latent_dims):
                for y_posn in range(heat_map_width):
                    for x_posn in range(heat_map_height):
                        heat_map[latent_dim, 0, x_posn, y_posn] = sample_heat_map[heat_map_width * y_posn + x_posn, latent_dim]

            if latent_order is not None:
                # Reorder latent samples by average KL
                heat_map = [
                    latent_sample for _, latent_sample in sorted(zip(latent_order, heat_map), reverse=True)
                ]
            heat_map_np = np.array(heat_map)
            upsampled_heat_map = torch.tensor(upsample(input_data=heat_map_np, scale_factor=2))   


        # Function: all_latent_traversals
        # Plot reconstructions in test mode, i.e. without sampling from latent
        self.model.eval()
        # Pass data through VAE to obtain reconstruction
        with torch.no_grad():
            input_data = latent_sweep_data.to(self.device)
            sample = self.model.sample_latent(input_data)
        # means = latent_dist[1]

        avg_kl_list = read_avg_kl_from_file(os.path.join(self.model_dir, 'losses.log'),self.model.latent_dim)

        # Perform line traversal of every latent
        latent_samples = []
        for idx in range(self.model.latent_dim):
            latent_samples.append(self.latent_traverser.traverse_line(idx=idx,
                                                                      size=size,
                                                                      sample_latent_space=None))
        latent_samples = [
            latent_sample for _, latent_sample in sorted(zip(avg_kl_list, latent_samples), reverse=True)
        ]
        
        sorted_avg_kl_list = [
            round(float(avg_kl_sample), 3) for avg_kl_sample, _ in sorted(zip(avg_kl_list, latent_samples), reverse=True)
        ]

        # Decode samples
        generated = self._decode_latents(torch.cat(latent_samples, dim=0))

        list_heat_map  = []
        for i in range(0, self.model.latent_dim):
            list_heat_map.append(upsampled_heat_map[i,0,:,:])
        
        heat_map_sorted = [
            list_heat_map for _, list_heat_map in sorted(zip(avg_kl_list, list_heat_map), reverse=True)
        ]
        
        new_torch = torch.tensor(np.zeros((self.model.latent_dim,1,64,64)))
        for i in range(0, self.model.latent_dim):
            new_torch[i,0,:,:]=heat_map_sorted[i]
        nr_imgs_per_latent = size

        # combine all images in the right order
        combined_torch = comparison
        for i in range(0,self.model.latent_dim):
            combined_torch = torch.cat((combined_torch, generated[nr_imgs_per_latent*i:nr_imgs_per_latent*(i+1),:,:,:].float()))
            combined_torch = torch.cat((combined_torch, new_torch[i:i+1,:,:,:].float()))
 
        traversal_images_with_text = add_labels('KL',combined_torch, size+1, sorted_avg_kl_list, self.dataset)

        if self.save_images:
            traversal_images_with_text.save(filename)
            return avg_kl_list
        else:
            return make_grid(combined_torch.data, nrow=size, pad_value=(1 - get_background(self.dataset)))

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
            sample = self.model.sample_latent(input_data)

            heat_map_height = heat_map_size[0]
            heat_map_width = heat_map_size[1]
            num_latent_dims = sample.shape[1]

            heat_map = torch.zeros([num_latent_dims, 1, heat_map_height, heat_map_width])

            for latent_dim in range(num_latent_dims):
                for y_posn in range(heat_map_width):
                    for x_posn in range(heat_map_height):
                        heat_map[latent_dim, 0, x_posn, y_posn] = sample[heat_map_width * y_posn + x_posn, latent_dim]

            if latent_order is not None:
                # Reorder latent samples by average KL
                heat_map = [
                    latent_sample for _, latent_sample in sorted(zip(latent_order, heat_map), reverse=True)
                ]
                heat_map = torch.stack(heat_map)
            # Normalise between 0 and 1
            heat_map = (heat_map - torch.min(heat_map)) / (torch.max(heat_map) - torch.min(heat_map))

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
            sample = self.model.sample_latent(input_data)
        return self.all_latent_traversals(sample_latent_space=sample, filename=filename)

    def reconstruction_comparisons(self, data, size=(8, 8), filename='imgs/recon_comp.png', exclude_original=False, exclude_recon=False):
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
        if exclude_original and exclude_recon:
            raise Exception('exclude_original and exclude_recon cannot both be True')
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

        if exclude_original:
            comparison = reconstructions
        if exclude_recon:
            comparison = originals
        if not exclude_original and not exclude_recon:
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

        avg_kl_list = read_avg_kl_from_file(os.path.join(self.model_dir, 'losses.log'),self.model.latent_dim)
        
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

        # Only keep the top 5 KL dimensions
        latent_samples = latent_samples[:5]

        # Decode samples
        generated = self._decode_latents(torch.cat(latent_samples, dim=0))
        # traversal_images_with_text = add_labels(
        #     label_name='KL',
        #     tensor=generated,
        #     num_rows=size,
        #     sorted_list=sorted_avg_kl_list,
        #     dataset=self.dataset)

        if self.save_images:
            # traversal_images_with_text.save(filename)
            # return avg_kl_list
            save_image(generated.data, filename, nrow=size, pad_value=(1 - get_background(self.dataset)))
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

