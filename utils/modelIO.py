import json
import os
import re

import torch

from disvae.vae import VAE
from disvae.encoder import get_Encoder
from disvae.decoder import get_Decoder
from utils.datasets import get_img_size


MODEL_FILENAME = "model"
META_FILENAME = "specs.json"  # CHANGE TO METADATA.json


def save_model(model, directory, metadata=None, filename=MODEL_FILENAME):
    """
    Save a model and corresponding metadata.

    Parameters
    ----------
    model : nn.Module
        Model.

    directory : str
        Path to the directory where to save the data.

    metadata : dict
        Metadata to save.
    """
    device = next(model.parameters()).device
    model.cpu()

    path_to_metadata = os.path.join(directory, META_FILENAME)
    path_to_model = os.path.join(directory, filename)

    torch.save(model.state_dict(), path_to_model)

    model.to(device)  # restore device

    if metadata is not None:
        with open(path_to_metadata, 'w') as f:
            json.dump(metadata, f, indent=4, sort_keys=True)


def load_metadata(directory):
    """Load the metadata of a training directory.

    Parameters
    ----------
    directory : string
        Path to folder where model is saved. For example './experiments/mnist'.

    """
    path_to_metadata = os.path.join(directory, META_FILENAME)

    with open(path_to_metadata) as metadata_file:
        metadata = json.load(metadata_file)

    return metadata


def load_model(directory, load_snapshots=False, is_gpu=True):
    """Load a trained model, or alternatively a list of trained models.

    Parameters
    ----------
    directory : string
        Path to folder where model is saved. For example './experiments/mnist'.

    load_snapshots : bool
        Indicates whether the models saved at different stages of training
        should be loaded and returned as a list.

    is_gpu : bool
        Whether to load on GPU is available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and is_gpu
                          else "cpu")

    path_to_model = os.path.join(directory, MODEL_FILENAME)
    metadata = load_metadata(directory)

    dataset = metadata["dataset"]
    latent_dim = metadata["latent_dim"]
    model_type = metadata["model_type"]
    img_size = get_img_size(dataset)

    if load_snapshots:
        model_list = []
        for root, _, names in os.walk(directory):
            for name in names:
                results = re.search(r'.*?-([0-9].*?).pt', name)
                if results is not None:
                    epoch_idx = int(results.group(1))

                    path_to_model = os.path.join(root, name)
                    model = _get_model(model_type, img_size, latent_dim, device, path_to_model)
                    model_list.append((epoch_idx, model))
        return model_list
    else:
        path_to_model = os.path.join(directory, MODEL_FILENAME + '.pt')
        model = _get_model(model_type, img_size, latent_dim, device, path_to_model)
        return model


def _get_model(model_type, img_size, latent_dim, device, path_to_model):
    """ Load a single model.

    Parameters
    ----------
    model_type : str
        The name of the model to load. For example Burgess.

    img_size : tuple
        Tuple of the number of pixels in the image width and height. 
        For example (32, 32) or (64, 64).

    latent_dim : int
        The number of latent dimensions in the bottleneck.
    
    device : str
        Either 'cuda' or 'cpu'

    path_to_device : str
        Full path to the saved model on the device.
    """
    encoder = get_Encoder(model_type)
    decoder = get_Decoder(model_type)
    model = VAE(img_size, encoder, decoder, latent_dim).to(device)
    # works with state_dict to make it independent of the file structure
    model.load_state_dict(torch.load(path_to_model))
    model.eval()

    return model
