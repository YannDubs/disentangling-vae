import numpy as np
import os

import torch
from scipy import stats
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from utils.datasets import get_background
from viz.latent_traversals import LatentTraverser
from viz.viz_helpers import (reorder_img, read_loss_from_file, add_labels,
                             upsample, make_grid_img)

import PIL
from PIL import Image
import os
from math import ceil, floor

TRAIN_FILE = "train_losses.log"
DECIMAL_POINTS = 3


class Visualizer():
    def __init__(self, model, dataset, model_dir=None, save_images=True,
                 loss_of_interest='kl_loss_', display_loss_per_dim=True):
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

        loss_of_interest : str
            The loss type (as saved in the log file) to order the latent dimensions by and display (optionally)

        display_loss_per_dim : bool
            if the loss should be included as text next to the corresponding latent dimension images.
        """
        self.model = model
        self.device = next(self.model.parameters()).device
        self.latent_traverser = LatentTraverser(self.model.latent_dim)
        self.save_images = save_images
        self.model_dir = model_dir
        self.dataset = dataset
        self.loss_of_interest = loss_of_interest

    def tensor_gray_scale_to_color(self, input_tensor):
        # input_tensor, consists of gray_scale values and has dimension: latent dim, gray_scale value, x_position, y_position

        size_output = list(input_tensor.size())
        size_output[1] = 3

        black_rgb = [0,0,0]
        white_rgb = [1,1,1]
        min_color = black_rgb # the color that the lowest gray scale value has 
        max_color = white_rgb # the color that the highest gray scale color has
        output = torch.zeros(size_output)
        for latent_dim in range(size_output[0]):
            for y_posn in range(size_output[2]):
                for x_posn in range(size_output[3]):
                    scale_id = input_tensor[latent_dim, 0, x_posn, y_posn]
                    output[latent_dim, 0, x_posn, y_posn] = min_color[0] + (max_color[0]-min_color[0])*scale_id
                    output[latent_dim, 1, x_posn, y_posn] = min_color[1] + (max_color[1]-min_color[1])*scale_id
                    output[latent_dim, 2, x_posn, y_posn] = min_color[2] + (max_color[2]-min_color[2])*scale_id
        return output

    def show_disentanglement_fig2(self, reconstruction_data, latent_sweep_data, heat_map_data,
                                  latent_order=None, heat_map_size=(32, 32), file_name='show-disentanglement.png',
                                  size=8, sample_latent_space=None, base_directory = '',select_prior=False,show_text=False):
        """ Reproduce Figure 2 from Burgess https://arxiv.org/pdf/1804.03599.pdf
            TODO: STILL TO BE IMPLEMENTED
        """
        image_file_name_list = [
            os.path.join(base_directory, 'recon_comp.png'),
            os.path.join(base_directory, 'heatmap.png'),
            ''
            ]
        # === get reconstruction === #
        self.reconstruction_comparisons(reconstruction_data, size=(9, 8), exclude_original=False, exclude_recon=False, color_flag = True, file_name=os.path.join(base_directory, 'recon_comp.png'))
        self.generate_heat_maps(heat_map_data, reorder=True, heat_map_size=(32, 32),file_name=os.path.join(base_directory, 'heatmap.png'))

        if select_prior == True:
            self.prior_traversal(reorder_latent_dims=True, file_name=os.path.join(base_directory, 'prior_traversal.png'))
            image_file_name_list[2] = os.path.join(base_directory, 'prior_traversal.png')
        else:
            self.traverse_posterior(data = latent_sweep_data, reorder_latent_dims=True, file_name=os.path.join(base_directory, 'posterior_traversal.png'))
            image_file_name_list[2] = os.path.join(base_directory, 'posterior_traversal.png')
        
        image_random_reconstruction = Image.open(image_file_name_list[0])
        image_heat_maps = Image.open(image_file_name_list[1])
        image_traversal = Image.open(image_file_name_list[2])

        width_new_image = image_random_reconstruction.size[0]
        height_new_image = image_random_reconstruction.size[1] + image_traversal.size[1]

        new_image = PIL.Image.new("RGB", (width_new_image, height_new_image))
        new_image.paste(image_random_reconstruction, (0,0))
        new_image.paste(image_heat_maps, (image_traversal.size[0],image_random_reconstruction.size[1]))
        new_image.paste(image_traversal, (0, image_random_reconstruction.size[1]))

        if show_text == True:
            loss_list = read_loss_from_file(os.path.join(self.model_dir, TRAIN_FILE), loss_to_fetch=self.loss_of_interest)
            new_image = add_labels('KL', new_image, 8, sorted(loss_list, reverse=True), self.dataset)

        new_image.save(file_name)

        return new_image
        

    def generate_heat_maps(self, data, reorder=False, heat_map_size=(32, 32), file_name='heatmap.png'):
        """
        Generates heat maps of the mean of each latent dimension in the model. The spites are
        assumed to be in order, therefore no information about (x,y) positions is required.

        Parameters
        ----------
        data : torch.Tensor
            Data to be used to generate the heat maps. Shape (N, C, H, W)

        reorder : bool
            If the heat maps should be reordered by descending loss

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

            heat_map_gray_scale = torch.zeros([num_latent_dims, 1, heat_map_height, heat_map_width])
            for latent_dim in range(num_latent_dims):
                for y_posn in range(heat_map_width):
                    for x_posn in range(heat_map_height):
                        heat_map_gray_scale[latent_dim, 0, x_posn, y_posn] = sample[heat_map_width * y_posn + x_posn, latent_dim]
            

            red_rgb = [1,0,0]
            blue_rgb = [0,0,1]
            min_gray = torch.min(heat_map_gray_scale)
            max_gray = torch.max(heat_map_gray_scale)

            heat_map_color = torch.zeros([num_latent_dims, 3, heat_map_height, heat_map_width])
            for latent_dim in range(num_latent_dims):
                for y_posn in range(heat_map_width):
                    for x_posn in range(heat_map_height):
                        scale_id = (heat_map_gray_scale[latent_dim, 0, x_posn, y_posn]-min_gray)/(max_gray-min_gray)
                        heat_map_color[latent_dim, 0, x_posn, y_posn] = red_rgb[0] + (blue_rgb[0]-red_rgb[0])*scale_id
                        heat_map_color[latent_dim, 1, x_posn, y_posn] = red_rgb[1] + (blue_rgb[1]-red_rgb[1])*scale_id
                        heat_map_color[latent_dim, 2, x_posn, y_posn] = red_rgb[2] + (blue_rgb[2]-red_rgb[2])*scale_id
            heat_map_np = np.array(heat_map_color)
            heat_map_color = torch.tensor(upsample(input_data=heat_map_np, scale_factor=2,colour_flag=True))

            if reorder:
                num_latent_dims = heat_map_color.shape[0]
                heat_map_list = [heat_map_color[i,:,:,:] for i in range(num_latent_dims)]

                loss_list = read_loss_from_file(os.path.join(self.model_dir, TRAIN_FILE), loss_to_fetch=self.loss_of_interest)
                heat_map_color = self.reorder(list_to_reorder=heat_map_list, reorder_by_list=loss_list)
            # Normalise between 0 and 1
            # heat_map = (heat_map - torch.min(heat_map)) / (torch.max(heat_map) - torch.min(heat_map))

            if self.save_images:
                save_image(heat_map_color.data, filename=file_name, nrow=1, pad_value=(1 - get_background(self.dataset)))

            else:
                return make_grid(heat_map_color.data, nrow=latent_dim, pad_value=(1 - get_background(self.dataset))), heat_map_color

    def traverse_posterior(self, data, num_increments=8, reorder_latent_dims=True,
                           display_loss_per_dim=False, file_name='posterior_traversal.png'):
        """
        Take 8 sample images, run them through the decoder, obtain the mean latent
        space vector. With this as the initialisation, traverse each dimension one
        by one to observe what each latent dimension encodes.

        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)

        num_increments : int
            Number of incremental steps to take in the traversal

        reorder_latent_dims : bool
            If the latent dimensions should be reordered or not

        display_loss_per_dim : bool
            If the loss should be included as text next to the corresponding latent dimension images.

        filename : string
            Name of file in which results are stored.
        """
        # Plot reconstructions in test mode, i.e. without sampling from latent
        self.model.eval()
        # Pass data through VAE to obtain reconstruction
        with torch.no_grad():
            input_data = data.to(self.device)
            sample = self.model.sample_latent(input_data)
        decoded_samples = self.all_latent_traversals(sample_latent_space=sample, size=num_increments)
        reorder_latent_dims=True
        if reorder_latent_dims:
            # Reshape into the appropriate form
            (num_images, num_channels, image_width, image_height) = decoded_samples.size()
            num_rows = int(num_images / num_increments)
            decoded_samples = torch.reshape(decoded_samples, (num_rows, num_increments, num_channels, image_width, image_height))
            decoded_list = [decoded_samples[i,:,:,:] for i in range(0,list(decoded_samples.size())[0])]

            loss_list = read_loss_from_file(os.path.join(self.model_dir, TRAIN_FILE), loss_to_fetch=self.loss_of_interest)
            decoded_samples = self.reorder(list_to_reorder=decoded_list, reorder_by_list=loss_list)
            decoded_samples = torch.reshape(decoded_samples, (num_images, num_channels, image_width, image_height))

        if display_loss_per_dim:
            sorted_loss_list = [
                    round(float(loss_sample), DECIMAL_POINTS) for loss_sample, _ in sorted(zip(loss_list, decoded_samples), reverse=True)
            ]

            traversal_images_with_text = add_labels(
                            label_name='KL',
                            tensor=decoded_samples,
                            num_rows=num_increments,
                            sorted_list=sorted_loss_list,
                            dataset=self.dataset
                        )
            traversal_images_with_text.save(file_name)




        if self.save_images and not display_loss_per_dim:
            save_image(
                tensor=decoded_samples.data,
                filename=file_name,
                nrow=num_increments,
                pad_value=(1 - get_background(self.dataset))
            )
        else:
            return make_grid_img(
                tensor=decoded_samples.data,
                nrow=num_increments,
                pad_value=(1 - get_background(self.dataset))
            )

    def reorder(self, list_to_reorder, reorder_by_list):
        """ Reorder the latent dimensions which are being traversed according to the reorder_by_list parameter.

        Parameters
        ----------
        list_to_reorder : list
            The list to reorder

        reorder_by_list : list
            A list with which to determine the reordering of list_to_reorder
        """

        latent_samples = [
            latent_sample[None,:,:,:] for _, latent_sample in sorted(zip(reorder_by_list, list_to_reorder), reverse=True)
        ]
        return torch.cat(latent_samples, dim=0)

    def visualise_data_set(self, data, size=(8, 8), file_name='visualise_data_set.png'):
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
        self.model.eval()
        # Pass data through VAE to obtain reconstruction
        with torch.no_grad():
            input_data = data.to(self.device)

        num_images = size[0]
        originals = input_data.cpu()

        # originals = torch.cat([originals, blank_images])

        if self.save_images:
            save_image(input_data.data,
                       filename=file_name,
                       nrow=size[0],
                       pad_value=(1 - get_background(self.dataset)))
        else:
            return comparison
            

    def reconstruction_comparisons(self, data, size=(8, 8), file_name='recon_comp.png',
                                   exclude_original=False, exclude_recon=False, color_flag = False):
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
            save_image(comparison.data,
                       filename=file_name,
                       nrow=size[0],
                       pad_value=(1 - get_background(self.dataset)))
        else:
            return comparison


    def generate_samples(self, size=(8, 8), file_name='samples.png'):
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
            save_image(generated.data, file_name, nrow=size[1], pad_value=(1 - get_background(self.dataset)))
        else:
            return make_grid_img(generated.data,
                                 nrow=size[1],
                                 pad_value=(1 - get_background(self.dataset)))

    def latent_traversal_line(self, idx=None, size=8,
                              file_name='traversal_line.png'):
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
            save_image(generated.data, 
                       filename=file_name,
                       nrow=size,
                       pad_value=(1 - get_background(self.dataset)))
        else:
            make_grid_img(generated.data,
                          nrow=size,
                          pad_value=(1 - get_background(self.dataset)))

    def latent_traversal_grid(self, idx=None, axis=None, size=(5, 5),
                              file_name='traversal_grid.png'):
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
            save_image(generated.data, file_name, nrow=size[1], pad_value=(1 - get_background(self.dataset)))
        else:
            return make_grid_img(generated.data,
                                 nrow=size[1],
                                 pad_value=(1 - get_background(self.dataset)))

    def prior_traversal(self, sample_latent_space=None, reorder_latent_dims=False, num_increments=8, file_name='prior_traversal.png'):
        """ Traverse the latent prior.
            Parameters
            ----------
            sample_latent_space : torch.Tensor or None
                The latent space of a sample which has been processed by the encoder.
                The dimensions are (size, num_latent_dims)
            num_increments : int
                The number of points to include in the traversal of a latent dimension.
            file_name : str
                The name of the output file.
            reorder_latent_dims : bool
                If the latent dimensions should be reordered or not
        """
        decoded_traversal = self.all_latent_traversals(
            sample_latent_space=sample_latent_space,
            size=num_increments
            )

        if reorder_latent_dims:
            # Reshape into the appropriate form
            (num_images, num_channels, image_width, image_height) = decoded_traversal.size()
            num_rows = int(num_images / num_increments)
            decoded_traversal = torch.reshape(decoded_traversal, (num_rows, num_increments, num_channels, image_width, image_height))
            decoded_list = [decoded_traversal[i,:,:,:] for i in range(0,list(decoded_traversal.size())[0])]

            loss_list = read_loss_from_file(os.path.join(self.model_dir, TRAIN_FILE), loss_to_fetch=self.loss_of_interest)
            decoded_traversal = self.reorder(list_to_reorder=decoded_list, reorder_by_list=loss_list)
            decoded_traversal = torch.reshape(decoded_traversal, (num_images, num_channels, image_width, image_height))

        if self.save_images:
            save_image(
                tensor=decoded_traversal.data,
                filename=file_name,
                nrow=num_increments,
                pad_value=(1 - get_background(self.dataset))
            )
        else:
            return make_grid_img(
                tensor=decoded_traversal.data,
                filename=file_name,
                nrow=num_increments,
                pad_value=(1 - get_background(self.dataset))
                )       


    def all_latent_traversals(self, sample_latent_space=None, size=8):
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
        # Perform line traversal of every latent
        for idx in range(self.model.latent_dim):
            latent_samples.append(self.latent_traverser.traverse_line(idx=idx,
                                                                      size=size,
                                                                      sample_latent_space=sample_latent_space))
        # Decode samples
        decoded_samples = self._decode_latents(torch.cat(latent_samples, dim=0))
        if self.save_images:
            return decoded_samples
        else:
            return make_grid_img(decoded_samples.data,
                                 nrow=size, pad_value=(1 - get_background(self.dataset)))

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
