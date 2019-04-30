import os
import shutil
import numpy as np
import ast
import configparser
import argparse
import random

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
        random.seed(seed)
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


def check_bounds(value, type=float, lb=-float("inf"), ub=float("inf"),
                 is_inclusive=True, name="value"):
    """Argparse bound checker"""
    value = type(value)
    is_in_bound = lb <= value <= ub if is_inclusive else lb < value < ub
    if not is_in_bound:
        raise argparse.ArgumentTypeError("{}={} outside of bounds ({},{})".format(name, value, lb, ub))
    return value


class FormatterNoDuplicate(argparse.ArgumentDefaultsHelpFormatter):
    """Formatter overriding `argparse.ArgumentDefaultsHelpFormatter` to show
    `-e, --epoch EPOCH` instead of `-e EPOCH, --epoch EPOCH`

    Note
    ----
    - code modified from cPython: https://github.com/python/cpython/blob/master/Lib/argparse.py
    """

    def _format_action_invocation(self, action):
        # no args given
        if not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            metavar, = self._metavar_formatter(action, default)(1)
            return metavar
        else:
            parts = []
            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                parts.extend(action.option_strings)
            # if the Optional takes a value, format is:
            #    -s ARGS, --long ARGS
            else:
                default = self._get_default_metavar_for_optional(action)
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    # don't store the DEFAULT
                    parts.append('%s' % (option_string))
                # store DEFAULT for the last one
                parts[-1] += ' %s' % args_string
            return ', '.join(parts)
