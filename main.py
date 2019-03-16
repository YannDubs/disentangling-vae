import argparse
import logging
from timeit import default_timer

from torch import optim

from disvae.vae import VAE
from disvae.encoder import get_Encoder
from disvae.decoder import get_Decoder
from disvae.discriminator import Discriminator
from disvae.training import Trainer
from utils.datasets import (get_dataloaders, get_img_size)
from utils.modelIO import save_model
from utils.helpers import create_safe_directory, get_device, set_seed, get_n_param


def default_experiment():
    """Default arguments."""
    return {'epochs': 1,
            'batch_size': 64,
            'no_cuda': False,
            'seed': 1234,
            'log_level': "info",
            "lr": 5e-4,
            "capacity": [0.0, 5.0, 25000, 30.0],
            "beta": 4.,
            "loss": "betaB",
            'model': 'Burgess',  # follows the paper by Burgess et al
            'dataset': 'mnist',
            'experiment': 'custom',
            "latent_dim": 10,
            "no_progress_bar": False,
            "checkpoint_every": 10,
            "lr_disc": 5e-4,
            "mss": True, # minibatch stratified sampling (batchTC)
            "alpha": 1.,
            "gamma": 1.,
            "mutual_info": False # Include mutual information term in factor-VAE
            }


def set_experiment(default_config):
    """ Set up default experiments to replicate the results in the paper:
        "Understanding Disentanglement in Beta-VAE" (https://arxiv.org/pdf/1804.03599.pdf)
    """
    # SHOULD BE A CONFIG I:I FILE
    if default_config.experiment == 'custom':
        return default_config
    elif default_config.experiment == 'beta_vae_chairs':
        default_config.beta = 1000
        default_config.dataset = 'chairs'
        default_config.loss = "betaH"
    elif default_config.experiment == 'factor_celeba':
        default_config.beta = 6.4
        default_config.dataset = 'celeba'
        default_config.loss = "factor"
        default_config.epochs = 300
        default_config.lr_disc = 1e-5
        default_config.lr = 1e-4
        default_config.batch_size = 64
        default_config.latent_dim = 10
    elif default_config.experiment == 'factor_chairs':
        default_config.beta = 3.2
        default_config.dataset = 'chairs'
        default_config.loss = "factor"
        default_config.epochs = 300
        default_config.lr_disc = 1e-5
        default_config.lr = 1e-4
        default_config.batch_size = 64
        default_config.latent_dim = 10
    elif default_config.experiment == 'factor_dsprites':
        default_config.beta = 36
        default_config.dataset = 'dsprites'
        default_config.loss = "factor"
        default_config.epochs = 50
        default_config.lr_disc = 1e-4
        default_config.lr = 1e-4
        default_config.batch_size = 64
        default_config.latent_dim = 10
    elif default_config.experiment == 'betaB_dsprites':
        default_config.capacity = [0.0, 25.0, 100000, 1000.0]
        default_config.dataset = 'dsprites'
        default_config.loss = "betaB"
        default_config.epochs = 50
        default_config.lr = 5e-4
        default_config.batch_size = 64
        default_config.latent_dim = 10
    elif default_config.experiment == 'betaB_celeba':
        default_config.capacity = [0.0, 50.0, 100000, 1000.0]
        default_config.dataset = 'celeba'
        default_config.loss = "betaB"
        default_config.epochs = 300
        default_config.lr = 5e-4
        default_config.batch_size = 64
        default_config.latent_dim = 10

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
    general.add_argument('--no-progress-bar', action='store_true',
                         default=default_config['no_progress_bar'],
                         help='Disables progress bar.')
    general.add_argument('--checkpoint-every',
                         type=int, default=default_config['checkpoint_every'],
                         help='Save a checkpoint of the trained model every n epoch.')

    # Dataset options
    data = parser.add_argument_group('Dataset options')
    datasets = ['mnist', "celeba", "chairs", "dsprites", "fashion"]
    data.add_argument('-d', '--dataset', help="Path to training data.",
                      default=default_config['dataset'], choices=datasets)

    # Predefined experiments
    experiment = parser.add_argument_group('Predefined experiments')
    experiments = ['custom', 'vae_blob_x_y', 'beta_vae_blob_x_y', 'beta_vae_dsprite', 'beta_vae_celeba', 'beta_vae_colour_dsprite', 'beta_vae_chairs', "betaB_dsprites",
                   "factor_dsprites", "factor_chairs", "factor_celeba", "betaB_celeba"]
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
    learn.add_argument('--lr',
                       type=float, default=default_config['lr'],
                       help='Learning rate.')
    learn.add_argument('--lr-disc',
                       type=float, default=default_config['lr_disc'],
                       help='Additional learning rate for a possible dirscriminator.')

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
    model.add_argument('-A', '--alpha',
                       type=float, default=default_config['alpha'],
                       help="Weight of the MI term. Only used if `loss=batchTC`")
    model.add_argument('-B', '--beta',
                       type=float, default=default_config['beta'],
                       help="Weight of the KL / TC term. Used if `loss=betaH` / `loss=batchTC`")
    model.add_argument('-G', '--gamma',
                       type=float, default=default_config['gamma'],
                       help="Weight of the dim-wise KL term. Only used if `loss=batchTC`")
    model.add_argument('-S', '--mss',
                       type=bool, default=default_config['mss'],
                       help="Weight of the MI term. Only used if `loss=batchTC`")
    model.add_argument('-MI', '--mi',
                       type=bool, default=default_config['mutual_info'],
                       help="Include mutual information in factor-VAE. Only used if `loss=factor`")
    losses = ["VAE", "betaH", "betaB", "factor", "batchTC"]
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

    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())
    stream = logging.StreamHandler()
    stream.setLevel(args.log_level.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if args.loss == "factor":
        logger.info("FactorVae needs 2 batches per iteration. To replicate this behavior while being consistent, we double the batch size and the the number of epochs.")
        args.batch_size *= 2
        args.epochs *= 2

    exp_dir = "experiments/{}".format(args.name)
    logger.info("Saving experiments to {}".format(exp_dir))
    create_safe_directory(exp_dir, logger=logger)
    set_seed(args.seed)
    device = get_device(is_gpu=not args.no_cuda)

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
    logger.info('Num parameters in model: {}'.format(get_n_param(model)))

    # TRAINS
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_kwargs = dict(capacity=args.capacity, beta=args.beta, latent_dim=args.latent_dim,
                       data_size=len(train_loader.dataset), mss=args.mss, alpha=args.alpha,
                       gamma=args.gamma)
    trainer = Trainer(model, optimizer,
                      loss_type=args.loss,
                      latent_dim=args.latent_dim,
                      loss_kwargs=loss_kwargs,
                      device=device,
                      log_level=args.log_level,
                      save_dir=exp_dir,
                      is_progress_bar=not args.no_progress_bar,
                      checkpoint_every=args.checkpoint_every,
                      dataset=args.dataset)

    trainer.train(train_loader, epochs=args.epochs)

    # SAVE MODEL AND EXPERIMENT INFORMATION
    save_model(trainer.model, exp_dir, metadata=vars(args))

    logger.info('Finished after {:.1f} min.'.format((default_timer() - start) / 60))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
