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
from utils.dataloaders import (get_mnist_dataloaders, get_dsprites_dataloader,
                               get_chairs_dataloader, get_fashion_mnist_dataloaders,
                               get_img_size)
from viz.visualize import Visualizer


def default_config():
    return {'epochs': 10,
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


def default_experiment(args):
    if args.experiment == 'custom':
        return args

    # TO FILL IN FOR ALL THE DEFAULT HYPERPARAMETERS OF THE EXPERIMENTS WE WANT
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
    parser.add_argument("-P", '--print_every',
                        type=int, default=defaultsConfig['print_every'],
                        help='Every how many batches to print results')
    parser.add_argument("-R", '--record_every',
                        type=int, default=defaultsConfig['record_every'],
                        help='Every how many batches to save results')

    # Dataset options
    data = parser.add_argument_group('Dataset options')
    datasets = ['mnist', "celeba", "chairs", "dsprites", "fashion"]
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
    learn.add_argument('-l', '--lr',
                       type=float, default=defaultsConfig['lr'],
                       help='Learning rate.')
    learn.add_argument('-c', '--capacity',
                       type=float, default=defaultsConfig['capacity'],
                       metavar=('MIN_CAPACITY, MAX_CAPACITY, NUM_ITERS, GAMMA_Z'),
                       nargs='*',
                       help="Capacity of latent channel.")

    # Model Options
    model = parser.add_argument_group('Learning options')
    models = ['Burgess']
    model.add_argument('-m', '--model',
                       default=defaultsConfig['model'], choices=models,
                       help='Type of encoder and decoder to use.')
    model.add_argument('-z', '--latent-dim',
                       default=defaultsConfig['latent_dim'], type=int,
                       help='Dimension of the latent variable.')

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
        train_loader, test = get_mnist_dataloaders(batch_size=args.batch_size)

    img_size = get_img_size(args.dataset)

    logger.info("Train {} with {} samples".format(args.dataset, len(train_loader)))

    # PREPARES MODEL
    if args.model == "Burgess":
        encoder = EncoderBetaB
        decoder = DecoderBetaB
    model = VAE(img_size, encoder, decoder, args.latent_dim)

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

    # EVALUATES / EXPERMINETS
    # TO DO @Aleco

    # SAVE
    model_dir = "trained_models/{}".format(args.dataset)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save(trainer.model.state_dict(), os.path.join(model_dir, 'model.pt'))
    with open(os.path.join(model_dir, 'specs.json'), 'w') as f:
        specs = dict(dataset=args.dataset,
                     latent_dim=args.latent_dim,
                     model_type=args.model)
        json.dump(specs, f)

    logger.info('Finished after {:.1f} min.'.format((default_timer() - start) / 60))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
