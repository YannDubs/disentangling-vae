# python main.py --name betaH_fashion2 -d dsprites -l betaH --lr 0.001 -b 16 -e 1 --betaH-B 15 --train_steps 15 --model-type=Burgess --plots=all

import argparse
import logging
import sys
import os
from configparser import ConfigParser
import wandb
import torch
import time
from torch import optim

from disvae import init_specific_model, Trainer, Evaluator
from disvae.utils.modelIO import save_model, load_model, load_metadata
from disvae.models.losses import LOSSES, RECON_DIST, get_loss_f
from disvae.models.vae import MODELS
from utils.datasets import get_dataloaders, get_img_size, DATASETS
from utils.helpers import (create_safe_directory, get_device, set_seed, get_n_param,
                           get_config_section, update_namespace_, FormatterNoDuplicate)
from utils.visualize import GifTraversalsTraining
from utils.miroslav import wandb_auth, latent_viz, cluster_metric
from utils.viz_helpers import get_samples

from utils.visualize import Visualizer

CONFIG_FILE = "hyperparam.ini"
RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom', "debug", "best_celeba", "best_dsprites"]
EXPERIMENTS = ADDITIONAL_EXP + ["{}_{}".format(loss, data)
                                for loss in LOSSES
                                for data in DATASETS]

PLOT_TYPES = ['generate-samples', 'data-samples', 'reconstruct', "traversals",
              'reconstruct-traverse', "gif-traversals", "all"]
              
def parse_arguments(args_to_parse):
    """Parse the command line arguments.

    Parameters
    ----------
    args_to_parse: list of str
        Arguments to parse (splitted on whitespaces).
    """
    default_config = get_config_section([CONFIG_FILE], "Custom")

    description = "PyTorch implementation and evaluation of disentangled Variational AutoEncoders and metrics."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('-name', '--name', type=str,
                         help="Name of the model for storing and loading purposes.")
    general.add_argument('-L', '--log-level', help="Logging levels.",
                         default=default_config['log_level'], choices=LOG_LEVELS)
    general.add_argument('--no-progress-bar', action='store_true',
                         default=default_config['no_progress_bar'],
                         help='Disables progress bar.')
    general.add_argument('--no-cuda', action='store_true',
                         default=default_config['no_cuda'],
                         help='Disables CUDA training, even when have one.')
    general.add_argument('-s', '--seed', type=int, default=default_config['seed'],
                         help='Random seed. Can be `None` for stochastic behavior.')
    general.add_argument('-max_traversal', '--max_traversal', type=float, default=0.475,
                         help='Random seed. Can be `None` for stochastic behavior.')

    general.add_argument('--is-show-loss', action='store_true',
                        help='Displays the loss on the figures (if applicable).')
    general.add_argument('--is-posterior', action='store_true',
                        help='Traverses the posterior instead of the prior.')
    general.add_argument('-i', '--idcs', type=int, nargs='+', default=[],
                        help='List of indices to of images to put at the begining of the samples.')
    general.add_argument("--plots", type=str, nargs='+', choices=PLOT_TYPES, default="all",
                        help="List of all plots to generate. `generate-samples`: random decoded samples. `data-samples` samples from the dataset. `reconstruct` first rnows//2 will be the original and rest will be the corresponding reconstructions. `traversals` traverses the most important rnows dimensions with ncols different samples from the prior or posterior. `reconstruct-traverse` first row for original, second are reconstructions, rest are traversals. `gif-traversals` grid of gifs where rows are latent dimensions, columns are examples, each gif shows posterior traversals. `all` runs every plot.")
    general.add_argument('--n-rows', type=int, default=6,
                        help='The number of rows to visualize (if applicable).')
    general.add_argument('--n-cols', type=int, default=7,
                        help='The number of columns to visualize (if applicable).')
    # Learning options
    training = parser.add_argument_group('Training specific options')
    training.add_argument('--checkpoint-every',
                          type=int, default=default_config['checkpoint_every'],
                          help='Save a checkpoint of the trained model every n epoch.')
    training.add_argument('-d', '--dataset', help="Path to training data.",
                          default=default_config['dataset'], choices=DATASETS)
    training.add_argument('-x', '--experiment',
                          default=default_config['experiment'], choices=EXPERIMENTS,
                          help='Predefined experiments to run. If not `custom` this will overwrite some other arguments.')
    training.add_argument('-e', '--epochs', type=int,
                          default=default_config['epochs'],
                          help='Maximum number of epochs to run for.')
    training.add_argument('-b', '--batch-size', type=int,
                          default=default_config['batch_size'],
                          help='Batch size for training.')
    training.add_argument('--lr', type=float, default=default_config['lr'],
                          help='Learning rate.')
    training.add_argument('--dry_run', type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False,
                        help='Whether to use WANDB in offline mode.')
    training.add_argument('--wandb_log', type=lambda x: False if x in ["False", "false", "", "None"] else True, default=True,
                help='Whether to use WANDB - this has implications for the training loop since if we want to log, we compute the metrics over training')                
    training.add_argument('--wandb_key', type=str, default=None,
                help='Path to WANDB key')    
    training.add_argument('--num_samples', type=int, default=None,
            help='How many samples to use. Useful for debugging')      
    training.add_argument('--train_steps', type=int, default=None,
            help='Number of training steps to use per epoch')
    training.add_argument('--higgins_drop_slow', type=bool, default=True,
        help='Whether to drop UMAP/TSNE etc. for computing Higgins metric (if we do not drop them, generating the data takes ~25 hours)')      

    # Model Options
    model = parser.add_argument_group('Model specfic options')
    model.add_argument('-m', '--model-type',
                       default=default_config['model'], choices=MODELS,
                       help='Type of encoder and decoder to use.')
    model.add_argument('-z', '--latent-dim', type=int,
                       default=default_config['latent_dim'],
                       help='Dimension of the latent variable.')
    model.add_argument('-l', '--loss',
                       default=default_config['loss'], choices=LOSSES,
                       help="Type of VAE loss function to use.")
    model.add_argument('-r', '--rec-dist', default=default_config['rec_dist'],
                       choices=RECON_DIST,
                       help="Form of the likelihood ot use for each pixel.")
    model.add_argument('-a', '--reg-anneal', type=float,
                       default=default_config['reg_anneal'],
                       help="Number of annealing steps where gradually adding the regularisation. What is annealed is specific to each loss.")

    # Loss Specific Options
    betaH = parser.add_argument_group('BetaH specific parameters')
    betaH.add_argument('--betaH-B', type=float,
                       default=default_config['betaH_B'],
                       help="Weight of the KL (beta in the paper).")

    betaB = parser.add_argument_group('BetaB specific parameters')
    betaB.add_argument('--betaB-initC', type=float,
                       default=default_config['betaB_initC'],
                       help="Starting annealed capacity.")
    betaB.add_argument('--betaB-finC', type=float,
                       default=default_config['betaB_finC'],
                       help="Final annealed capacity.")
    betaB.add_argument('--betaB-G', type=float,
                       default=default_config['betaB_G'],
                       help="Weight of the KL divergence term (gamma in the paper).")

    factor = parser.add_argument_group('factor VAE specific parameters')
    factor.add_argument('--factor-G', type=float,
                        default=default_config['factor_G'],
                        help="Weight of the TC term (gamma in the paper).")
    factor.add_argument('--lr-disc', type=float,
                        default=default_config['lr_disc'],
                        help='Learning rate of the discriminator.')

    btcvae = parser.add_argument_group('beta-tcvae specific parameters')
    btcvae.add_argument('--btcvae-A', type=float,
                        default=default_config['btcvae_A'],
                        help="Weight of the MI term (alpha in the paper).")
    btcvae.add_argument('--btcvae-G', type=float,
                        default=default_config['btcvae_G'],
                        help="Weight of the dim-wise KL term (gamma in the paper).")
    btcvae.add_argument('--btcvae-B', type=float,
                        default=default_config['btcvae_B'],
                        help="Weight of the TC term (beta in the paper).")

    # Learning options
    evaluation = parser.add_argument_group('Evaluation specific options')
    evaluation.add_argument('--is-eval-only', action='store_true',
                            default=default_config['is_eval_only'],
                            help='Whether to only evaluate using precomputed model `name`.')
    evaluation.add_argument('--is-metrics', action='store_true',
                            default=default_config['is_metrics'],
                            help="Whether to compute the disentangled metrcics. Currently only possible with `dsprites` as it is the only dataset with known true factors of variations.")
    evaluation.add_argument('--no-test', action='store_true',
                            default=default_config['no_test'],
                            help="Whether not to compute the test losses.`")
    evaluation.add_argument('--eval-batchsize', type=int,
                            default=default_config['eval_batchsize'],
                            help='Batch size for evaluation.')

    args = parser.parse_args(args_to_parse)
    if args.experiment != 'custom':
        if args.experiment not in ADDITIONAL_EXP:
            # update all common sections first
            model, dataset = args.experiment.split("_")
            common_data = get_config_section([CONFIG_FILE], "Common_{}".format(dataset))
            update_namespace_(args, common_data)
            common_model = get_config_section([CONFIG_FILE], "Common_{}".format(model))
            update_namespace_(args, common_model)

        try:
            experiments_config = get_config_section([CONFIG_FILE], args.experiment)
            update_namespace_(args, experiments_config)
        except KeyError as e:
            if args.experiment in ADDITIONAL_EXP:
                raise e  # only reraise if didn't use common section

    return args

def main(args):
    """Main train and evaluation function.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments
    """
    if args.dry_run:
        os.environ['WANDB_MODE'] = 'dryrun'
    try:
        wandb_auth(dir_path=args.wandb_key)
    except:
        print(f"Authentication for WANDB failed! Trying to disable it")
        os.environ["WANDB_MODE"] = "disabled"
    wandb.init(project='atmlbetavae', entity='atml', group="miroslav")
    wandb.config.update(args)

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
    exp_dir = os.path.join(RES_DIR, args.name+f"{time.time()}")
    logger.info("Root directory for saving and loading experiments: {}".format(exp_dir))

    if not args.is_eval_only:

        create_safe_directory(exp_dir, logger=logger)

        if args.loss == "factor":
            logger.info("FactorVae needs 2 batches per iteration. To replicate this behavior while being consistent, we double the batch size and the the number of epochs.")
            args.batch_size *= 2
            args.epochs *= 2

        # PREPARES DATA
        train_loader, raw_dataset = get_dataloaders(args.dataset,
                                       batch_size=args.batch_size,
                                       logger=logger,
                                       n_samples=args.num_samples)
        logger.info("Train {} with {} samples".format(args.dataset, len(train_loader.dataset)))

        # PREPARES MODEL
        args.img_size = get_img_size(args.dataset)  # stores for metadata
        model = init_specific_model(args.model_type, args.img_size, args.latent_dim)
        logger.info('Num parameters in model: {}'.format(get_n_param(model)))
        model = model.to(device)  # make sure trainer and viz on same device

        # TRAINS
        if args.model_type == "Burgess":
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif args.model_type == "Higginsdsprites":
            optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
        elif args.model_type == "Higginsconv":
            optimizer = optim.Adam(model.parameters(), lr=args.lr)


        gif_visualizer = GifTraversalsTraining(model, args.dataset, exp_dir)
        loss_f = get_loss_f(args.loss,
                            n_data=len(train_loader.dataset),
                            device=device,
                            **vars(args))
        trainer = Trainer(model, optimizer, loss_f,
                          device=device,
                          logger=logger,
                          save_dir=exp_dir,
                          is_progress_bar=not args.no_progress_bar,
                          gif_visualizer=gif_visualizer,
                          metrics_freq=10 if args.dataset in ['dsprites'] else 50,
                          seed=args.seed,
                          steps = args.train_steps,
                          dset_name=args.dataset,
                          higgins_drop_slow=args.higgins_drop_slow)

        trainer(train_loader,
                epochs=args.epochs,
                checkpoint_every=args.checkpoint_every,
                wandb_log = args.wandb_log)

        latents_plots, traversal_plots, cluster_score = {}, {}, {}
        latents_plots, latent_data, dim_reduction_models = latent_viz(model, train_loader, args.dataset, raw_dataset=raw_dataset, steps=100, device=device)
        

        model_dir = os.path.join(RES_DIR, args.name)
        viz = Visualizer(model=model,
                    model_dir=model_dir,
                    dataset=args.dataset,
                    max_traversal=args.max_traversal,
                    loss_of_interest='kl_loss_',
                    upsample_factor=1)

        traversal_plots = {}
        base_datum = next(iter(train_loader))[0][0].unsqueeze(dim=0)
        for model_name, model in dim_reduction_models.items():
            traversal_plots[model_name] = viz.latents_traversal_plot(model, data=base_datum, n_per_latent=50)

        # Original plots from the repo
        size = (args.n_rows, args.n_cols)
        # same samples for all plots: sample max then take first `x`data  for all plots
        num_samples = args.n_cols * args.n_rows
        samples = get_samples(dataset, num_samples, idcs=args.idcs)

        if "all" in args.plots:
            args.plots = [p for p in PLOT_TYPES if p != "all"]
        builtin_plots = {}
        plot_fnames = []
        for plot_type in args.plots:
            if plot_type == 'generate-samples':
                fname, plot = viz.generate_samples(size=size)
                builtin_plots["generate-samples"] = plot
            elif plot_type == 'data-samples':
                fname, plot = viz.data_samples(samples, size=size)
                builtin_plots["data-samples"] = plot
            elif plot_type == "reconstruct":
                fname, plot = viz.reconstruct(samples, size=size)
                builtin_plots["reconstruct"] = builtin_plots
            elif plot_type == 'traversals':
                fname, plot =viz.traversals(data=samples[0:1, ...] if args.is_posterior else None,
                            n_per_latent=args.n_cols,
                            n_latents=args.n_rows,
                            is_reorder_latents=True)
                builtin_plots["traversals"] = plot
            elif plot_type == "reconstruct-traverse":
                fname, plot = viz.reconstruct_traverse(samples,
                                        is_posterior=args.is_posterior,
                                        n_latents=args.n_rows,
                                        n_per_latent=args.n_cols,
                                        is_show_text=args.is_show_loss)
                builtin_plots["reconstruct-traverse"] = plot
            elif plot_type == "gif-traversals":
                fname, plot = viz.gif_traversals(samples[:args.n_cols, ...], n_latents=args.n_rows)
                builtin_plots["gif-traversals"] = plot
            else:
                raise ValueError("Unkown plot_type={}".format(plot_type))
            plot_fnames.append(fname)


        if args.wandb_log:
            wandb.log({"latents":latents_plots, "latent_traversal":traversal_plots, "cluster_metric":cluster_score, "builtin_plots":builtin_plots})
            for fname in plot_fnames:
                try:
                    wandb.save(fname)
                except Exception as e:
                    print(f"Failed to save {fname} to WANDB. Exception: {e}")


        # SAVE MODEL AND EXPERIMENT INFORMATION
        save_model(trainer.model, exp_dir, metadata=vars(args))

    if args.is_metrics or not args.no_test:
        print("Evaluation time.")
        model = load_model(exp_dir, is_gpu=not args.no_cuda)
        metadata = load_metadata(exp_dir)
        # TO-DO: currently uses train datatset
        test_loader, raw_dataset = get_dataloaders(metadata["dataset"],
                                      batch_size=args.eval_batchsize,
                                      shuffle=False,
                                      logger=logger)
        loss_f = get_loss_f(args.loss,
                            n_data=len(test_loader.dataset),
                            device=device,
                            **vars(args))
        evaluator = Evaluator(model, loss_f,
                              device=device,
                              logger=logger,
                              save_dir=exp_dir,
                              is_progress_bar=not args.no_progress_bar, 
                              use_wandb=True,
                              seed=args.seed,
                              higgins_drop_slow=args.higgins_drop_slow,
                              dset_name=args.dataset)

        evaluator(test_loader, is_metrics=args.is_metrics, is_losses=not args.no_test)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
