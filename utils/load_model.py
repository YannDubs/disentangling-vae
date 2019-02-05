import json
import torch
from disvae.vae import VAE
from disvae.encoder import EncoderBetaB
from disvae.decoder import DecoderBetaB
from utils.dataloaders import (get_mnist_dataloaders, get_dsprites_dataloader,
                               get_chairs_dataloader, get_fashion_mnist_dataloaders,
                               get_img_size)


def load(path):
    """
    Loads a trained model.

    Parameters
    ----------
    path : string
        Path to folder where model is saved. For example
        './trained_models/mnist/'. Note the path MUST end with a '/'
    """
    path_to_specs = path + 'specs.json'
    path_to_model = path + 'model.pt'

    # Open specs file
    with open(path_to_specs) as specs_file:
        specs = json.load(specs_file)

    dataset = specs["dataset"]
    latent_dim = specs["latent_dim"]
    model_type = specs["model_type"]

    img_size = get_img_size(dataset)

    # Get model
    if model_type == "Burgess":
        encoder = EncoderBetaB
        decoder = DecoderBetaB
    model = VAE(img_size, encoder, decoder, latent_dim=latent_dim)
    model.load_state_dict(torch.load(path_to_model,
                                     map_location=lambda storage, loc: storage))

    return model
