import csv
import torch

from torchvision.utils import make_grid
from torchvision import transforms
from utils.datasets import get_background
from PIL import Image, ImageDraw


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
        avg_kl = [0]*nr_latent_variables
        for i in range(1, nr_latent_variables+1):
            avg_kl[i-1] = total_list[-(2+nr_latent_variables)+i][2]

    return avg_kl


def add_labels(label_name, tensor, num_rows, sorted_list, dataset):
    """ Adds the average KL per latent dimension as a label next to the relevant row as in
        figure 2 of Burgress et al.
    """
    # Convert tensor to PIL Image
    tensor = make_grid(tensor.data, nrow=num_rows, pad_value=(1 - get_background(dataset)))
    all_traversal_im = transforms.ToPILImage()(tensor)
    # Resize image
    new_width = int(1.3 * all_traversal_im.width)
    new_size = (all_traversal_im.height, new_width)
    traversal_images_with_text = Image.new("RGB", new_size, color='white')
    traversal_images_with_text.paste(all_traversal_im, (0, 0))
    # Add KL text alongside each row
    draw = ImageDraw.Draw(traversal_images_with_text)
    for idx, elem in enumerate(sorted_list):
        draw.text(xy=(int(0.825 * traversal_images_with_text.width),
                        int((idx / len(sorted_list) + \
                            1 / (2 * len(sorted_list))) * all_traversal_im.height)),
                    text="{} = {}".format(label_name, elem),
                    fill=(0,0,0))
    return traversal_images_with_text
