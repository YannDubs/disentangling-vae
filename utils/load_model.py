import json
import os

import torch

from disvae.vae import VAE
from disvae.encoder import get_Encoder
from disvae.decoder import get_Decoder
from utils.dataloaders import (get_mnist_dataloaders, get_dsprites_dataloader,
                               get_chairs_dataloader, get_fashion_mnist_dataloaders,
                               get_img_size)


def load_model(path, is_gpu=False):
    """
    Loads a trained model.

    Parameters
    ----------
    path : string
        Path to folder where model is saved. For example './experiments/mnist'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and is_gpu
                          else "cpu")

    path_to_specs = os.path.join(path, 'specs.json')
    path_to_model = os.path.join(path, 'model.pt')

    # Open specs file
    with open(path_to_specs) as specs_file:
        specs = json.load(specs_file)

    dataset = specs["dataset"]
    latent_dim = specs["latent_dim"]
    model_type = specs["model_type"]
    img_size = get_img_size(dataset)

    # Get model
    encoder = get_Encoder(model_type)
    decoder = get_Decoder(model_type)
    model = VAE(img_size, encoder, decoder, latent_dim).to(device)
    # works with state_dict to make it independent of the file structure
    model.load_state_dict(torch.load(path_to_model))
    model.eval()

    return model
