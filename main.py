import argparse
import logging
import sys
from configparser import ConfigParser

from torch import optim

from disvae.vae import VAE
from disvae.encoder import get_Encoder
from disvae.decoder import get_Decoder
from disvae.training import Trainer
from disvae.evaluate import Evaluator
from utils.datasets import get_dataloaders, get_img_size
from utils.modelIO import save_model, load_model, load_metadata
from utils.helpers import (create_safe_directory, get_device, set_seed, get_n_param,
                           get_config_section, update_namespace_)


CONFIG_FILE = "hyperparam.ini"
TEST_FILE = "test_losses.log"


def parse_arguments(args_to_parse):
    """Parse the command line arguments.

    Parameters
    ----------
    args_to_parse: str
        String of arguments to parse
    """
    default_config = get_config_section([CONFIG_FILE], "Custom")

    description = "PyTorch implementation and evaluation of disentangled Variational AutoEncoders and metrics."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('name', type=str,
                         help="Name of the model for storing or loading purposes.")
    log_levels = ['critical', 'error', 'warning', 'info', 'debug']
    general.add_argument('-L', '--log-level', help="Logging levels.",
                         default=default_config['log_level'],
                         choices=log_levels)
    general.add_argument('--no-progress-bar', action='store_true',
                         default=default_config['no_progress_bar'],
                         help='Disables progress bar.')
    general.add_argument('--no-cuda', action='store_true',
                         default=default_config['no_cuda'],
                         help='Disables CUDA training, even when have one.')
    general.add_argument('-s', '--seed', type=int, default=default_config['seed'],
                         help='Random seed. Can be `None` for stochastic behavior.')

    # Learning options
    training = parser.add_argument_group('Training specific options')
    training.add_argument('--checkpoint-every',
                          type=int, default=default_config['checkpoint_every'],
                          help='Save a checkpoint of the trained model every n epoch.')
    datasets = ['mnist', "celeba", "chairs", "dsprites", "fashion"]
    training.add_argument('-d', '--dataset', help="Path to training data.",
                          default=default_config['dataset'], choices=datasets)
    experiments = ['custom', "debug"] + ["{}_{}".format(loss, data)
                                         for loss in ["betaH", "betaB", "factor", "batchTC"]
                                         for data in ["celeba", "chairs", "dsprites"]]
    training.add_argument('-x', '--experiment',
                          default=default_config['experiment'], choices=experiments,
                          help='Predefined experiments to run. If not `custom` this will overwrite some other arguments.')
    training.add_argument('-e', '--epochs',
                          type=int, default=default_config['epochs'],
                          help='Maximum number of epochs to run for.')
    training.add_argument('-b', '--batch-size',
                          type=int, default=default_config['batch_size'],
                          help='Batch size for training.')
    training.add_argument('--lr',
                          type=float, default=default_config['lr'],
                          help='Learning rate.')

    # Model Options
    model = parser.add_argument_group('Model specfic options')
    models = ['Burgess']
    model.add_argument('-m', '--model-type',
                       default=default_config['model'], choices=models,
                       help='Type of encoder and decoder to use.')
    model.add_argument('-z', '--latent-dim',
                       default=default_config['latent_dim'], type=int,
                       help='Dimension of the latent variable.')
    losses = ["VAE", "betaH", "betaB", "factor", "batchTC"]
    model.add_argument('-l', '--loss',
                       choices=losses, default=default_config['loss'],
                       help="Type of VAE loss function to use.")

    # Loss Specific Options
    betaH = parser.add_argument_group('BetaH specific parameters')
    betaH.add_argument('--betaH-B',
                       type=float, default=default_config['betaH_B'],
                       help="Weight of the KL (beta in the paper).")

    betaB = parser.add_argument_group('BetaB specific parameters')
    betaB.add_argument('--betaB-initC',
                       type=float, default=default_config['betaB_initC'],
                       help="Starting annealed capacity.")
    betaB.add_argument('--betaB-finC',
                       type=float, default=default_config['betaB_finC'],
                       help="Final annealed capacity.")
    betaB.add_argument('--betaB-stepsC',
                       type=float, default=default_config['betaB_stepsC'],
                       help="Number of training iterations for interpolating C.")
    betaB.add_argument('--betaB-G',
                       type=float, default=default_config['betaB_G'],
                       help="Weight of the KL divergence term (gamma in the paper).")

    factor = parser.add_argument_group('factor VAE specific parameters')
    factor.add_argument('--factor-G', type=float,
                        default=default_config['factor_G'],
                        help="Weight of the TC term (gamma in the paper).")
    factor.add_argument('--no-mutual-info', action='store_true',
                        default=default_config['no_mutual_info'],
                        help="Remove mutual information.")
    factor.add_argument('--lr-disc',
                        type=float, default=default_config['lr_disc'],
                        help='Learning rate of the discriminator.')

    batchTC = parser.add_argument_group('batchTC specific parameters')
    batchTC.add_argument('--batchTC-A', type=float,
                         default=default_config['batchTC_A'],
                         help="Weight of the MI term (alpha in the paper).")
    batchTC.add_argument('--batchTC-G', type=float,
                         default=default_config['batchTC_G'],
                         help="Weight of the dim-wise KL term (gamma in the paper).")
    batchTC.add_argument('--batchTC-B', type=float,
                         default=default_config['batchTC_B'],
                         help="Weight of the TC term (beta in the paper).")
    batchTC.add_argument('--no-mss', action='store_true',
                         default=default_config['no_mss'],
                         help="Whether to use minibatch weighted sampling instead of stratified.`")

    # Learning options
    evaluation = parser.add_argument_group('Evaluation specific options')
    evaluation.add_argument('--is-eval-only',
                            action='store_true', default=default_config['is_eval_only'],
                            help='Whether to only evaluate using precomputed model `name`.')
    evaluation.add_argument('--is-metrics', action='store_true', default=default_config['is_metrics'],
                            help="Whether to compute the disentangled metrcics.`")
    evaluation.add_argument('--no-test', action='store_true', default=default_config['no_test'],
                            help="Whether not to compute the test losses.`")
    evaluation.add_argument('-eb', '--eval-batchsize',
                            type=int, default=default_config['eval_batchsize'],
                            help='Batch size for evaluation.')

    args = parser.parse_args(args_to_parse)
    if args.experiment != 'custom':
        experiments_config = get_config_section([CONFIG_FILE], args.experiment)
        update_namespace_(args, experiments_config)

    if args.name is None:
        args.name = args.experiment

    return args


def main(args):
    """Main train and evaluation function.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments
    """
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())
    stream = logging.StreamHandler()
    stream.setLevel(args.log_level.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    set_seed(args.seed)
    device = get_device(is_gpu=not args.no_cuda)
    exp_dir = "experiments/{}".format(args.name)
    logger.info("Root directory for saving and loading experiments: {}".format(exp_dir))

    if not args.is_eval_only:

        create_safe_directory(exp_dir, logger=logger)

        if args.loss == "factor":
            logger.info("FactorVae needs 2 batches per iteration. To replicate this behavior while being consistent, we double the batch size and the the number of epochs.")
            args.batch_size *= 2
            args.epochs *= 2

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

        loss_kwargs = vars(args).copy()
        loss_kwargs["data_size"] = len(train_loader.dataset)
        trainer = Trainer(model, optimizer,
                          loss_type=args.loss,
                          latent_dim=args.latent_dim,
                          loss_kwargs=loss_kwargs,
                          device=device,
                          logger=logger,
                          save_dir=exp_dir,
                          is_progress_bar=not args.no_progress_bar,
                          checkpoint_every=args.checkpoint_every,
                          dataset=args.dataset)

        trainer(train_loader, epochs=args.epochs)

        # SAVE MODEL AND EXPERIMENT INFORMATION
        save_model(trainer.model, exp_dir, metadata=vars(args))

    if args.is_metrics or not args.no_test:
        model = load_model(exp_dir, is_gpu=not args.no_cuda)
        metadata = load_metadata(exp_dir)
        # TO-DO: currently uses train datatset
        test_loader = get_dataloaders(metadata["dataset"],
                                      batch_size=args.eval_batchsize, shuffle=False)

        loss_kwargs = vars(args).copy()
        loss_kwargs["data_size"] = len(test_loader.dataset)
        evaluator = Evaluator(model,
                              loss_type=args.loss,
                              loss_kwargs=loss_kwargs,
                              device=device,
                              logger=logger,
                              save_dir=exp_dir,
                              is_progress_bar=not args.no_progress_bar)

        evaluator(test_loader, is_metrics=args.is_metrics, is_losses=not args.no_test)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
