import argparse
import sys
import os
import logging
from timeit import default_timer

parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)

import torch
import numpy as np
from pytoune.framework import Model
from pytoune.framework.callbacks import TerminateOnNaN, ModelCheckpoint

from evaluate.dataloaders import get_mnist_dataloaders
from disvae.loss import get_loss
from disvae.vae import VAE
from disvae.encoder import EncoderBetaB
from disvae.decoder import DecoderBetaB


def default_config():
    return {'epochs': 3,
            'batch_size': 64,
            'no_cuda': False,
            'no_checkpoint': False,
            'seed': 1234,
            'log-level': "info",

            'model': 'Burgess',  # follows the paper by Burgess et al
            'dataset': 'mnist',
            'loss': 'beta-vae',
            'experiment': 'custom'
            }


def default_experiment(args):
    if args.experiment == 'custom':
        return args

    # TO FILL IN FOR ALL THE DEFAULT HYPERPARAMETERS OF TEH EXPERIMENTS WE WANT
    # TO REPLICATE
    # @aleco

    return args


def parse_arguments():
    defaultsConfig = default_config()
    parser = argparse.ArgumentParser(description="PyTorch implementation and evaluation of disentangled Variational AutoEncoders.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General options
    general = parser.add_argument_group('General options')
    log_levels = ['critical', 'error', 'warning', 'info', 'debug']
    general.add_argument('-L', '--log-level', help="Logging levels.",
                         default=defaultsConfig['log-level'],
                         choices=log_levels)
    general.add_argument('--no-checkpoint',
                         action='store_true', default=defaultsConfig['no_checkpoint'],
                         help='Disables model checkpoint. I.e saving best model based on validation loss.')

    # Dataset options
    data = parser.add_argument_group('Dataset options')
    datasets = ['mnist']
    data.add_argument('-d', '--dataset', help="Path to training data.",
                      default=defaultsConfig['dataset'], choices=datasets)

    # Predefined experiments
    experiment = parser.add_argument_group('Predefined experiments')
    experiments = ['custom']
    experiment.add_argument('-x', '--experiment',
                            default=defaultsConfig['experiment'], choices=experiments,
                            help='Predefined experiments to run. If not `custom` this will set the correct other arguments.')

    # Learning options
    learn = parser.add_argument_group('Learning options')
    learn.add_argument('-e', '--epochs',
                       type=int, default=defaultsConfig['epochs'],
                       help='Maximum number of epochs to run for.')
    learn.add_argument('-b', '--batch-size',
                       type=int, default=defaultsConfig['batch_size'],
                       help='Batch size for training.')
    learn.add_argument('-s', '--seed', type=int, default=defaultsConfig['seed'],
                       help='Random seed. Can be `None` for stochastic behavior.')
    learn.add_argument('--no-cuda', action='store_true',
                       default=defaultsConfig['no_cuda'],
                       help='Disables CUDA training, even when have one.')

    # Model Options
    model = parser.add_argument_group('Learning options')
    models = ['mnist']
    model.add_argument('-m', '--model',
                       default=defaultsConfig['model'], choices=models,
                       help='Type of encoder and decoder to use.')

    # Loss Options
    loss = parser.add_argument_group('Learning options')
    losses = ['beta-vae']
    loss.add_argument('-l', '--loss',
                      default=defaultsConfig['loss'], choices=losses,
                      help='Type of VAE loss.')

    args = parser.parse_args()

    return args


def main(args):
    start = default_timer()

    logging.basicConfig(format='%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                        datefmt="%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # if want pure determinism could uncomment below: but slower
        # torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda
                          else "cpu")

    # PREPARES DATA
    if args.dataset == "mnist":
        train, test, img_size = get_mnist_dataloaders(batch_size=args.batch_size)

    logger.info("Train {} with {} samples".format(args.dataset, len(train)))

    # PREPARES MODEL
    if args.model == "Burgess":
        encoder = EncoderBetaB(img_size)
        decoder = DecoderBetaB(img_size)
    vae = VAE(encoder, decoder)

    model_parameters = filter(lambda p: p.requires_grad, vae.parameters())
    nParams = sum([np.prod(p.size()) for p in model_parameters])
    logger.info('Num parameters in model: {}'.format(nParams))

    # COMPILES
    if args.loss == "beta-vae":
        loss = get_loss("b-vae", beta=4)

    model = Model(vae, "adam", loss)
    model.to(device)

    callbacks = [TerminateOnNaN()]
    if not args.no_checkpoint:
        modelDir = os.path.join(parentddir, 'results/models')
        filename = os.path.join(modelDir, "{}.pth.tar".format(args.dataset))
        callbacks.append(ModelCheckpoint(filename,
                                         save_best_only=True,
                                         monitor="train_loss"))

    # TRAINS
    model.fit_generator(train, epochs=args.epochs)

    # EVALUATES

    logger.info('Finished after {:.1f} min.'.format((default_timer() - start) / 60))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
