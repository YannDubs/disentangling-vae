import argparse
import os
import logging
import shutil
from timeit import default_timer

import torch
import numpy as np
from torch import optim

from disvae.vae import VAE
from disvae.encoder import get_Encoder
from disvae.decoder import get_Decoder
from disvae.discriminator import Discriminator
from disvae.training import Trainer
from utils.datasets import (get_dataloaders, get_img_size)
from utils.modelIO import save_model


def default_experiment():
    """Default arguments."""
    return {'epochs': 10,
            'batch_size': 64,
            'no_cuda': False,
            'seed': 1234,
            'log_level': "info",
            "lr": 1e-3,
            "capacity": [0.0, 5.0, 25000, 30.0],
            "beta": 4.,
            "loss": "batchTC",
            "print_every": 50,
            "record_every": 5,
            'model': 'Burgess',  # follows the paper by Burgess et al
            'dataset': 'mnist',
            'experiment': 'custom',
            "latent_dim": 10,
            'save_model_on_epochs': ()
            }


def set_experiment(default_config):
    """ Set up default experiments to replicate the results in the paper:
        "Understanding Disentanglement in Beta-VAE" (https://arxiv.org/pdf/1804.03599.pdf)
    """
    if default_config.experiment == 'custom':
        return default_config
    elif default_config.experiment == 'vae_blob_x_y':
        default_config.beta = 1
        default_config.dataset = 'dsprites'
        default_config.loss = "betaH"
    elif default_config.experiment == 'beta_vae_blob_x_y':
        default_config.beta = 150
        default_config.dataset = 'black_and_white_dsprite'
        default_config.loss = "betaH"
    elif default_config.experiment == 'beta_vae_dsprite':
        default_config.capacity = [0.0, 25.0, 100000, 1000.0]
        default_config.dataset = 'dsprites'
        default_config.loss = "betaB"
    elif default_config.experiment == 'beta_vae_celeba':
        default_config.capacity = [0.0, 50.0, 100000, 10000.]
        default_config.dataset = 'celeba'
        default_config.loss = "betaB"
    elif default_config.experiment == 'beta_vae_colour_dsprite':
        default_config.capacity = [0.0, 25.0, 100000, 1000.0]
        default_config.dataset = 'dsprites'
        default_config.loss = "betaB"
    elif default_config.experiment == 'beta_vae_chairs':
        default_config.beta = 1000
        default_config.dataset = 'chairs'
        default_config.loss = "betaH"

    return default_config


def parse_arguments():
    """Parse the command line arguments."""
    default_config = default_experiment()

    parser = argparse.ArgumentParser(description="PyTorch implementation and evaluation of disentangled Variational AutoEncoders.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General options
    general = parser.add_argument_group('General options')
    log_levels = ['critical', 'error', 'warning', 'info', 'debug']
    general.add_argument('-L', '--log-level', help="Logging levels.",
                         default=default_config['log_level'],
                         choices=log_levels)
    parser.add_argument("-P", '--print_every',
                        type=int, default=default_config['print_every'],
                        help='Every how many batches to print results')
    parser.add_argument("-R", '--record_every',
                        type=int, default=default_config['record_every'],
                        help='Every how many batches to save results')
    parser.add_argument("-S", '--save_model_on_epochs', nargs=2,
                        default=default_config['save_model_on_epochs'],
                        help='On which epochs the model will be saved (in addition to at the end of training)')

    # Dataset options
    data = parser.add_argument_group('Dataset options')
    datasets = ['mnist', "celeba", "chairs", "dsprites", "fashion"]
    data.add_argument('-d', '--dataset', help="Path to training data.",
                      default=default_config['dataset'], choices=datasets)

    # Predefined experiments
    experiment = parser.add_argument_group('Predefined experiments')
    experiments = ['custom', 'vae_blob_x_y', 'beta_vae_blob_x_y', 'beta_vae_dsprite',
                   'beta_vae_celeba', 'beta_vae_colour_dsprite', 'beta_vae_chairs']
    experiment.add_argument('-x', '--experiment',
                            default=default_config['experiment'], choices=experiments,
                            help='Predefined experiments to run. If not `custom` this will set the correct other arguments.')
    experiment.add_argument('-n', '--name', type=str, default=None,
                            help="Name for storing the experiment. If not given, uses `experiment`.")

    # Learning options
    learn = parser.add_argument_group('Learning options')
    learn.add_argument('-e', '--epochs',
                       type=int, default=default_config['epochs'],
                       help='Maximum number of epochs to run for.')
    learn.add_argument('-b', '--batch-size',
                       type=int, default=default_config['batch_size'],
                       help='Batch size for training.')
    learn.add_argument('-s', '--seed', type=int, default=default_config['seed'],
                       help='Random seed. Can be `None` for stochastic behavior.')
    learn.add_argument('--no-cuda', action='store_true',
                       default=default_config['no_cuda'],
                       help='Disables CUDA training, even when have one.')
    learn.add_argument('-a', '--lr',
                       type=float, default=default_config['lr'],
                       help='Learning rate.')

    # Model Options
    model = parser.add_argument_group('Learning options')
    models = ['Burgess']
    model.add_argument('-m', '--model-type',
                       default=default_config['model'], choices=models,
                       help='Type of encoder and decoder to use.')
    model.add_argument('-z', '--latent-dim',
                       default=default_config['latent_dim'], type=int,
                       help='Dimension of the latent variable.')
    model.add_argument('-c', '--capacity',
                       type=float, default=default_config['capacity'],
                       metavar=('MIN_C, MAX_C, C_N_INTERP, GAMMA'),
                       nargs='*',
                       help="Capacity of latent channel. Only used if `loss=betaB`")
    model.add_argument('-B', '--beta',
                       type=float, default=default_config['beta'],
                       help="Weight of the KL term. Only used if `loss=betaH`")
    losses = ["VAE", "betaH", "betaB", "factorising", "batchTC"]
    model.add_argument('-l', '--loss',
                       choices=losses, default=default_config['loss'],
                       help="type of VAE loss function to use.")

    args = parser.parse_args()
    args = set_experiment(args)
    if args.name is None:
        args.name = args.experiment

    return args


def main(args):
    start = default_timer()

    logging.basicConfig(format='%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                        datefmt="%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())

    # Experiments directory
    exp_dir = "experiments/{}".format(args.name)
    if os.path.exists(exp_dir):
        warn = "Directory {} already exists. Archiving it to {}.zip"
        logger.warning(warn.format(exp_dir, exp_dir))
        shutil.make_archive(exp_dir, 'zip', exp_dir)
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # if want pure determinism could uncomment below: but slower
        # torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda
                          else "cpu")

    # PREPARES DATA
    train_loader = get_dataloaders(args.dataset,
                                   batch_size=args.batch_size,
                                   pin_memory=not args.no_cuda)

    img_size = get_img_size(args.dataset)

    logger.info("Train {} with {} samples".format(args.dataset, len(train_loader.dataset)))

    # PREPARES MODEL
    encoder = get_Encoder(args.model_type)
    decoder = get_Decoder(args.model_type)
    model = VAE(img_size, encoder, decoder, args.latent_dim)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nParams = sum([np.prod(p.size()) for p in model_parameters])
    logger.info('Num parameters in model: {}'.format(nParams))

    # TRAINS
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_kwargs = dict(capacity=args.capacity, beta=args.beta, latent_dim=args.latent_dim, data_size=len(train_loader.dataset))
    trainer = Trainer(model, optimizer,
                      loss_type=args.loss,
                      latent_dim=args.latent_dim,
                      loss_kwargs=loss_kwargs,
                      print_loss_every=args.print_every,
                      record_loss_every=args.record_every,
                      device=device,
                      log_level=args.log_level,
                      save_dir=exp_dir,
                      save_epoch_list=args.save_model_on_epochs,
                      dataset=args.dataset)
    trainer.train(train_loader, epochs=args.epochs)

    # SAVE MODEL AND EXPERIMENT INFORMATION
    save_model(trainer.model, vars(args), exp_dir)

    logger.info('Finished after {:.1f} min.'.format((default_timer() - start) / 60))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
