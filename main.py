import argparse
import os
import logging
from timeit import default_timer
import json

import torch
import numpy as np
from torch import optim

from disvae.vae import VAE
from disvae.encoder import EncoderBetaB
from disvae.decoder import DecoderBetaB
from disvae.training import Trainer
from utils.dataloaders import (get_dataloaders, get_mnist_dataloaders, get_dsprites_dataloader,
                               get_chairs_dataloader, get_fashion_mnist_dataloaders,
                               get_img_size)
from viz.visualize import Visualizer


def default_experiment():
    return {'epochs': 100,
            'batch_size': 64,
            'no_cuda': False,
            'seed': 1234,
            'log-level': "info",
            "lr": 1e-3,
            "capacity": [0.0, 5.0, 25000, 30.0],
            "print_every": 50,
            "record_every": 5,
            'model': 'Burgess',  # follows the paper by Burgess et al
            'dataset': 'mnist',
            'experiment': 'custom',
            "latent_dim": 10
            }


def set_experiment(default_config):
    """ Set up default experiments to replicate the results in the paper:
        "Understanding Disentanglement in Beta-VAE" (https://arxiv.org/pdf/1804.03599.pdf)
    """
    if default_config.experiment == 'custom':
        return default_config
    elif default_config.experiment == 'vae_blob_x_y':
        default_config.capacity = 1
        default_config.dataset = 'dsprites'
    elif default_config.experiment == 'beta_vae_blob_x_y':
        default_config.capacity = 150
        default_config.dataset = 'black_and_white_dsprite'
    elif default_config.experiment == 'beta_vae_dsprite':
        default_config.capacity = [0.0, 25.0, 100000, 1000.0]
        default_config.dataset = 'dsprites'
    elif default_config.experiment == 'beta_vae_celeba':
        default_config.capacity = [0.0, 50.0, 100000, 10000.]
        default_config.dataset = 'celeba'
    elif default_config.experiment == 'beta_vae_colour_dsprite':
        default_config.capacity = [0.0, 25.0, 100000, 1000.0]
        default_config.dataset = 'dsprites'
    elif default_config.experiment == 'beta_vae_chairs':
        default_config.capacity = 1000
        default_config.dataset = 'chairs'

    return default_config


def parse_arguments():
    default_config = default_experiment()

    parser = argparse.ArgumentParser(description="PyTorch implementation and evaluation of disentangled Variational AutoEncoders.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General options
    general = parser.add_argument_group('General options')
    log_levels = ['critical', 'error', 'warning', 'info', 'debug']
    general.add_argument('-L', '--log-level', help="Logging levels.",
                         default=default_config['log-level'],
                         choices=log_levels)
    parser.add_argument("-P", '--print_every',
                        type=int, default=default_config['print_every'],
                        help='Every how many batches to print results')
    parser.add_argument("-R", '--record_every',
                        type=int, default=default_config['record_every'],
                        help='Every how many batches to save results')

    # Dataset options
    data = parser.add_argument_group('Dataset options')
    datasets = ['mnist', "celeba", "chairs", "dsprites", "fashion_mnist"]
    data.add_argument('-d', '--dataset', help="Path to training data.",
                      default=default_config['dataset'], choices=datasets)

    # Predefined experiments
    experiment = parser.add_argument_group('Predefined experiments')
    experiments = ['custom', 'vae_blob_x_y', 'beta_vae_blob_x_y', 'beta_vae_dsprite', 'beta_vae_celeba', 'beta_vae_colour_dsprite', 'beta_vae_chairs']
    experiment.add_argument('-x', '--experiment',
                            default=default_config['experiment'], choices=experiments,
                            help='Predefined experiments to run. If not `custom` this will set the correct other arguments.')

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
    learn.add_argument('-l', '--lr',
                       type=float, default=default_config['lr'],
                       help='Learning rate.')
    learn.add_argument('-c', '--capacity',
                       type=float, default=default_config['capacity'],
                       metavar=('MIN_CAPACITY, MAX_CAPACITY, NUM_ITERS, GAMMA_Z'),
                       nargs='*',
                       help="Capacity of latent channel.")

    # Model Options
    model = parser.add_argument_group('Learning options')
    models = ['Burgess']
    model.add_argument('-m', '--model',
                       default=default_config['model'], choices=models,
                       help='Type of encoder and decoder to use.')
    model.add_argument('-z', '--latent-dim',
                       default=default_config['latent_dim'], type=int,
                       help='Dimension of the latent variable.')

    args = parser.parse_args()
    
    print(args)

    experiment_config = set_experiment(args)

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
    train_loader, test = get_dataloaders(batch_size=args.batch_size, dataset=args.dataset)

    img_size = get_img_size(args.dataset)

    logger.info("Train {} with {} samples".format(args.dataset, len(train_loader)))

    # PREPARES MODEL
    if args.model == "Burgess":
        encoder = EncoderBetaB
        decoder = DecoderBetaB
    model = VAE(img_size, encoder, decoder, args.latent_dim, device=device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nParams = sum([np.prod(p.size()) for p in model_parameters])
    logger.info('Num parameters in model: {}'.format(nParams))

    # TRAINS
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(model, optimizer,
                      latent_dim=args.latent_dim,
                      capacity=args.capacity,
                      print_loss_every=args.print_every,
                      record_loss_every=args.record_every,
                      device=device,
                      log_level=args.log_level)
    viz = Visualizer(model)
    trainer.train(train_loader,
                  epochs=args.epochs,
                  save_training_gif=('./imgs/training.gif', viz))

    # SAVE MODEL AND EXPERIMENT INFORMATION
    model_dir = "trained_models/{}/".format(args.experiment)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save(trainer.model, os.path.join(model_dir, 'model.pt'))
    with open(os.path.join(model_dir, 'specs.json'), 'w') as f:
        specs = dict(dataset=args.dataset,
                     latent_dim=args.latent_dim,
                     model_type=args.model,
                     capacity=args.capacity,
                     experiment_name=args.experiment)
        json.dump(specs, f)

    logger.info('Finished after {:.1f} min.'.format((default_timer() - start) / 60))


if __name__ == '__main__':
    args = parse_arguments()
    args.experiment
    main(args)
