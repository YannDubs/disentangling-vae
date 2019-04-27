import json
import os
import re

import numpy as np
import torch

from disvae import init_specific_model

MODEL_FILENAME = "model.pt"
META_FILENAME = "specs.json"


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

    if metadata is None:
        # save the minimum required for loading
        metadata = dict(img_size=model.img_size, latent_dim=model.latent_dim,
                        model_type=model.model_type)

    save_metadata(metadata, directory)

    path_to_model = os.path.join(directory, filename)
    torch.save(model.state_dict(), path_to_model)

    model.to(device)  # restore device


def load_metadata(directory, filename=META_FILENAME):
    """Load the metadata of a training directory.

    Parameters
    ----------
    directory : string
        Path to folder where model is saved. For example './experiments/mnist'.
    """
    path_to_metadata = os.path.join(directory, filename)

    with open(path_to_metadata) as metadata_file:
        metadata = json.load(metadata_file)

    return metadata


def save_metadata(metadata, directory, filename=META_FILENAME, **kwargs):
    """Load the metadata of a training directory.

    Parameters
    ----------
    metadata:
        Object to save

    directory: string
        Path to folder where to save model. For example './experiments/mnist'.

    kwargs:
        Additional arguments to `json.dump`
    """
    path_to_metadata = os.path.join(directory, filename)

    with open(path_to_metadata, 'w') as f:
        json.dump(metadata, f, indent=4, sort_keys=True, **kwargs)


def load_model(directory, is_gpu=True, filename=MODEL_FILENAME):
    """Load a trained model.

    Parameters
    ----------
    directory : string
        Path to folder where model is saved. For example './experiments/mnist'.

    is_gpu : bool
        Whether to load on GPU is available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and is_gpu
                          else "cpu")

    path_to_model = os.path.join(directory, MODEL_FILENAME)

    metadata = load_metadata(directory)
    img_size = metadata["img_size"]
    latent_dim = metadata["latent_dim"]
    model_type = metadata["model_type"]

    path_to_model = os.path.join(directory, filename)
    model = _get_model(model_type, img_size, latent_dim, device, path_to_model)
    return model


def load_checkpoints(directory, is_gpu=True):
    """Load all chechpointed models.

    Parameters
    ----------
    directory : string
        Path to folder where model is saved. For example './experiments/mnist'.

    is_gpu : bool
        Whether to load on GPU .
    """
    checkpoints = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            results = re.search(r'.*?-([0-9].*?).pt', filename)
            if results is not None:
                epoch_idx = int(results.group(1))
                model = load_model(root, is_gpu=is_gpu, filename=filename)
                checkpoints.append((epoch_idx, model))

    return checkpoints


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
    model = init_specific_model(model_type, img_size, latent_dim).to(device)
    # works with state_dict to make it independent of the file structure
    model.load_state_dict(torch.load(path_to_model), strict=False)
    model.eval()

    return model


def numpy_serialize(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


def save_np_arrays(arrays, directory, filename):
    """Save dictionary of arrays in json file."""
    save_metadata(arrays, directory, filename=filename, default=numpy_serialize)


def load_np_arrays(directory, filename):
    """Load dictionary of arrays from json file."""
    arrays = load_metadata(directory, filename=filename)
    return {k: np.array(v) for k, v in arrays.items()}
