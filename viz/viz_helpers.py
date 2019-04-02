import csv
import numpy as np
import pandas as pd
import torch

from torchvision.utils import make_grid
from torchvision import transforms
from utils.datasets import get_background
from PIL import Image, ImageDraw, ImageFont


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


def read_loss_from_file(log_file_path, loss_to_fetch="kl_loss_"):
    """ Read the average KL per latent dimension at the final stage of training from the log file.
        Parameters
        ----------
        log_file_path : str
            Full path and file name for the log file. For example 'experiments/custom/losses.log'.

        loss_to_fetch : str
            The loss type to search for in the log file and return. This must be in the exact form as stored.
    """
    EPOCH = "Epoch"
    LOSS = "Loss"

    logs = pd.read_csv(log_file_path)
    df_last_epoch_loss = logs[logs.loc[:, EPOCH] == logs.loc[:, EPOCH].max()]
    df_last_epoch_loss = df_last_epoch_loss.loc[df_last_epoch_loss.loc[:, LOSS].str.startswith(loss_to_fetch), :]
    df_last_epoch_loss.loc[:, LOSS] = df_last_epoch_loss.loc[:, LOSS].str.replace(loss_to_fetch,"").astype(int)
    df_last_epoch_loss = df_last_epoch_loss.sort_values(LOSS).loc[:, "Value"]
    return list(df_last_epoch_loss)


# def add_labels(label_name, tensor, num_rows, sorted_list, dataset):
def add_labels(label_name, input_image, num_rows, sorted_list, dataset):
    """ Adds the label next to the relevant row as in an image. This is used to reproduce
        figure 2 of Burgress et al.

        Parameters
        ----------
        label_name : str
            The name of the labels to add, for sample 'KL' or 'C'.
        input_image : image
            The image to which to add the labels
        num_rows : int
            The number of rows of images to display
        sorted_list : list
            The list of sorted objects.
        dataset : str
            The dataset name.
    """
    all_traversal_im = input_image
    # Resize image
    if num_rows == 7:
        mult_x = 1.5
    elif num_rows == 8:
        mult_x = 1.3
    elif num_rows == 9:
        mult_x = 1.2
    new_width = int(mult_x * all_traversal_im.width)
    new_size = (new_width, all_traversal_im.height)
    traversal_images_with_text = Image.new("RGB", new_size, color='white')
    traversal_images_with_text.paste(all_traversal_im, (0, 0))

    # Add KL text alongside each row
    fraction_x = 1 / mult_x + 0.005
    text_list = ['orig', 'recon']
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
    draw = ImageDraw.Draw(traversal_images_with_text)
    for i in range(0, 2):
        draw.text(xy=(int(fraction_x * traversal_images_with_text.width),
                      int((i / (len(sorted_list) + 2) + \
                            1 / (2 * (len(sorted_list) + 2))) * all_traversal_im.height)),
                    text=text_list[i],
                    fill=(0,0,0),
                    font=fnt)

    for latent_idx, latent_dim in enumerate(sorted_list):
        draw.text(xy=(int(fraction_x * traversal_images_with_text.width),
                      int(((latent_idx+2) / (len(sorted_list)+2) + \
                            1 / (2 * (len(sorted_list)+2))) * all_traversal_im.height)),
                    text=label_name + " = %7.4f"%(latent_dim),
                    fill=(0,0,0),
                    font=fnt)
    return traversal_images_with_text   


def upsample(input_data, scale_factor, is_torch_input=False, colour_flag=False):
    """ TODO: add Docstring
    """
    is_torch_input = False
    if isinstance(input_data, torch.Tensor):
        input_data = input_data.detach().numpy()
        is_torch_input = True

    new_array = np.zeros((input_data.shape[0], input_data.shape[1], input_data.shape[2] * scale_factor, input_data.shape[3] * scale_factor))
    for latent_dim in range(input_data.shape[0]):
        for x in range(input_data.shape[2]):
            for y in range(input_data.shape[3]):
                if colour_flag == False:
                    # new_array[latent_dim, 0, x * scale_factor:x * scale_factor + scale_factor, y * scale_factor:y * scale_factor + scale_factor] = input_data[latent_dim, 0, x, y]
                    new_array[latent_dim, 0, x * scale_factor:x * scale_factor + scale_factor - 1, y * scale_factor:y * scale_factor + scale_factor - 1] = input_data[latent_dim, 0, x, y]
                else:
                    new_array[latent_dim, 0, x * scale_factor:x * scale_factor + scale_factor, y * scale_factor:y * scale_factor + scale_factor] = input_data[latent_dim, 0, x, y]
                    new_array[latent_dim, 1, x * scale_factor:x * scale_factor + scale_factor, y * scale_factor:y * scale_factor + scale_factor] = input_data[latent_dim, 1, x, y]
                    new_array[latent_dim, 2, x * scale_factor:x * scale_factor + scale_factor, y * scale_factor:y * scale_factor + scale_factor] = input_data[latent_dim, 2, x, y]
    # new_array = np.zeros((input_data.shape[0], input_data.shape[1], input_data.shape[2] * scale_factor, input_data.shape[3] * scale_factor))
    # for latent_dim in range(0, input_data.shape[0]):
    #     for x in range(0, input_data.shape[2]):
    #         for y in range(0, input_data.shape[2]):
    #             new_array[latent_dim, 0, x * scale_factor:x * scale_factor + scale_factor - 1, y * scale_factor:y * scale_factor + scale_factor - 1] = input_data[latent_dim, 0, x, y]

    if is_torch_input:
        return torch.from_numpy(new_array)
    else:
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

def get_image_list(image_file_name_list):
    image_list = []
    for file_name in image_file_name_list:
        image_list.append(Image.open(file_name))
    return image_list
