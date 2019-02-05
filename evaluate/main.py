import argparse
import sys
import os

parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)

from pytoune.framework import Model

from evaluate.dataloaders import get_mnist_dataloaders
from disvae.loss import get_loss
from disvae.vae import VAE
from disvae.encoder import EncoderBetaB
from disvae.decoder import DecoderBetaB


def default_config():
    return {'epochs': 3,
            'batch_size': 64,

            'model': 'Burgess',  # follows the paper by Burgess et al
            'dataset': 'mnist',
            'loss': 'beta-vae',
            }


def parse_arguments():
    defaultsConfig = default_config()
    parser = argparse.ArgumentParser(description="PyTorch implementation and evaluation of disentangled Variational AutoEncoders.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset options
    data = parser.add_argument_group('Dataset options')
    datasets = ['mnist']
    data.add_argument('-d', '--dataset', help="Path to training data.",
                      default=defaultsConfig['dataset'], choices=datasets)

    # Learning options
    learn = parser.add_argument_group('Learning options')
    learn.add_argument('-e', '--epochs',
                       type=int, default=defaultsConfig['epochs'],
                       help='Maximum number of epochs to run for.')
    learn.add_argument('-b', '--batch-size',
                       type=int, default=defaultsConfig['batch_size'],
                       help='Batch size for training.')

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
    if args.dataset == "mnist":
        train, test, img_size = get_mnist_dataloaders(batch_size=args.batch_size)

    if args.loss == "beta-vae":
        loss = get_loss("b-vae", beta=4)

    if args.model == "Burgess":
        encoder = EncoderBetaB(img_size)
        decoder = DecoderBetaB(img_size)
    vae = VAE(encoder, decoder)

    model = Model(vae, "adam", loss)
    model.fit_generator(train, epochs=args.epochs)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
