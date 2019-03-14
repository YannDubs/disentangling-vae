import csv
import torch

from torchvision.utils import make_grid
from torchvision import transforms
from utils.datasets import get_background
from PIL import Image, ImageDraw
import numpy as np


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


def read_avg_kl_from_file(log_file_path, nr_latent_variables):
    """ Read the average KL per latent dimension at the final stage of training from the log file.
    """
    with open(log_file_path, 'r') as f:
        total_list = list(csv.reader(f))
        avg_kl = [0] * nr_latent_variables
        for i in range(1, nr_latent_variables + 1):
            avg_kl[i - 1] = total_list[-(2 + nr_latent_variables) + i][2]

    return avg_kl


def add_labels(label_name, tensor, num_rows, sorted_list, dataset):
    """ Adds the average KL per latent dimension as a label next to the relevant row as in
        figure 2 of Burgress et al.
    """
    # Convert tensor to PIL Image
    tensor = make_grid(tensor.data, nrow=num_rows, pad_value=(1 - get_background(dataset)))
    all_traversal_im = transforms.ToPILImage()(tensor)
    # Resize image
    if num_rows == 8:
        mult_x = 1.3
    elif num_rows == 9:
        mult_x = 1.2
    # Resize image
    new_width = int(mult_x * all_traversal_im.width)
    new_size = (new_width, all_traversal_im.height)
    traversal_images_with_text = Image.new("RGB", new_size, color='white')
    traversal_images_with_text.paste(all_traversal_im, (0, 0))
    # Add KL text alongside each row
    fraction_x = 1 / mult_x + 0.050
    text_list = ['orig', 'recon']
    draw = ImageDraw.Draw(traversal_images_with_text)
    for i in range(0, 2):
        draw.text(xy=(int(fraction_x * traversal_images_with_text.width),
                      int((i / (len(sorted_list) + 2) +
                           1 / (2 * (len(sorted_list) + 2))) * all_traversal_im.height)),
                  text=text_list[i],
                  fill=(0, 0, 0))

    for latent_idx, latent_dim in enumerate(sorted_list):
        draw.text(xy=(int(fraction_x * traversal_images_with_text.width),
                      int(((latent_idx + 2) / (len(sorted_list) + 2) +
                           1 / (2 * (len(sorted_list) + 2))) * all_traversal_im.height)),
                  text="KL = {}".format(latent_dim),
                  fill=(0, 0, 0))
    return traversal_images_with_text


def upsample(input_data, scale_factor):
    # dubplicate
    new_array = np.zeros((input_data.shape[0], input_data.shape[1], input_data.shape[2] * scale_factor, input_data.shape[3] * scale_factor))
    for latent_dim in range(0, input_data.shape[0]):
        for x in range(0, input_data.shape[2]):
            for y in range(0, input_data.shape[2]):
                new_array[latent_dim, 0, x * scale_factor:x * scale_factor + scale_factor - 1, y * scale_factor:y * scale_factor + scale_factor - 1] = input_data[latent_dim, 0, x, y]
    return new_array


def make_grid_img(tensor, **kwargs):
    """Converts a tensor to a grid of images that can be read by imageio.

    Notes
    -----
    * from in https://github.com/pytorch/vision/blob/master/torchvision/utils.py

    Parameters
    ----------
    tensor (torch.Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
        or a list of images all of the same size.

    kwargs:
        Additional arguments to `make_grid_img`.
    """
    grid = make_grid(tensor, **kwargs)
    img_grid = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    img_grid = img_grid.to('cpu', torch.uint8).numpy()
    return img_grid
