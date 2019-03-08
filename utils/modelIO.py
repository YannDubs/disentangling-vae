import json
import os
import re

import torch

from disvae.vae import VAE
from disvae.encoder import get_Encoder
from disvae.decoder import get_Decoder
from utils.datasets import get_img_size

MODEL_FILENAME = "model"
SPECS_FILENAME = "specs.json"


def save_model(model, specs, directory, original_device=None, epoch=None):
    """
    Save a model and corresponding specs.

    Parameters
    ----------
    model : nn.Module
        Model.

    specs : dict
        Metadata to save.

    directory : str
        Path to the directory where to save the data.

    original_device : torch.device
        Original device on which the model runs. Include this parameter to
        return the model to this device after saving.
    """
    model.cpu()
    if epoch is None:
        path_to_model = os.path.join(directory, MODEL_FILENAME + '.pt')
    else:
        path_to_model = os.path.join(directory, MODEL_FILENAME + "-{}{}".format(epoch, '.pt'))

    torch.save(model.state_dict(), path_to_model)

    if specs is not None:
        path_to_specs = os.path.join(directory, SPECS_FILENAME)
        with open(path_to_specs, 'w') as f:
            json.dump(specs, f, indent=4, sort_keys=True)

    if original_device is not None:
        model.to(original_device)

def load_model(directory, load_snapshots=False, is_gpu=True):
    """
    Loads a trained model.

    Parameters
    ----------
    directory : string
        Path to folder where model is saved. For example './experiments/mnist'.

    is_gpu : bool
        Whether to load on GPU is available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and is_gpu
                          else "cpu")

    path_to_specs = os.path.join(directory, SPECS_FILENAME)

    # Open specs file
    with open(path_to_specs) as specs_file:
        specs = json.load(specs_file)

    dataset = specs["dataset"]
    latent_dim = specs["latent_dim"]
    model_type = specs["model_type"]
    img_size = get_img_size(dataset)


    if load_snapshots:
        model_list = []
        for root, _, names in os.walk(directory):
            for name in names:
                results = re.search(r'.*?-([0-9].*?).pt', name)
                if results.group(0) is not None:
                    print('results.group(0): ' + str(results.group(0)))
                    print('results.group(1): ' + str(results.group(1)))
                    epoch_idx = int(results.group(1))

                    path_to_model = os.path.join(root, name)
                    model = _get_model(model_type, img_size, latent_dim, device, path_to_model)
                    model_list.append([epoch_idx, model])
        return model_list
    else:
        path_to_model = os.path.join(directory, MODEL_FILENAME + '.pt')
        model = _get_model(model_type, img_size, latent_dim, device, path_to_model)
        return model

def _get_model(model_type, img_size, latent_dim, device, path_to_model):
    """ Get model """
    encoder = get_Encoder(model_type)
    decoder = get_Decoder(model_type)
    model = VAE(img_size, encoder, decoder, latent_dim).to(device)
    # works with state_dict to make it independent of the file structure
    model.load_state_dict(torch.load(path_to_model))
    model.eval()

    return model
    