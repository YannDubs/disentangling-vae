import os
import shutil
import numpy as np
import ast
import configparser

import torch


def create_safe_directory(directory, logger=None):
    """Create a directory and archive the previous one if already existed."""
    if os.path.exists(directory):
        if logger is not None:
            warn = "Directory {} already exists. Archiving it to {}.zip"
            logger.warning(warn.format(directory, directory))
        shutil.make_archive(directory, 'zip', directory)
        shutil.rmtree(directory)
    os.makedirs(directory)


def set_seed(seed):
    """Set all random seeds."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        # if want pure determinism could uncomment below: but slower
        # torch.backends.cudnn.deterministic = True


def get_device(is_gpu=True):
    """Return the correct device"""
    return torch.device("cuda" if torch.cuda.is_available() and is_gpu
                        else "cpu")


def get_model_device(model):
    """Return the device on which a model is."""
    return next(model.parameters()).device


def get_n_param(model):
    """Return the number of parameters."""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nParams = sum([np.prod(p.size()) for p in model_parameters])
    return nParams


def mean(l):
    """Compute the mean of a list"""
    return sum(l) / len(l)


def update_namespace_(namespace, dictionnary):
    """Update an argparse namespace in_place."""
    vars(namespace).update(dictionnary)


def get_config_section(filenames, section):
    """Return a dictionnary of the section of `.ini` config files. Every value
    int the `.ini` will be litterally evaluated, such that `l=[1,"as"]` actually
    returns a list.
    """
    parser = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    parser.optionxform = str
    files = parser.read(filenames)
    if len(files) == 0:
        raise ValueError("Config files not found: {}".format(filenames))
    dict_session = dict(parser[section])
    dict_session = {k: ast.literal_eval(v) for k, v in dict_session.items()}
    return dict_session
