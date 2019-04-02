import argparse
import json
import os
import torch

import numpy as np
from torchvision.utils import save_image

from utils.datasets import get_dataloaders, get_background
from utils.helpers import FormatterNoDuplicate
from viz.visualize import Visualizer
from viz.viz_helpers import add_labels
from viz.log_plotter import LogPlotter

from main import RES_DIR
from disvae import init_specific_model
from disvae.utils.modelIO import load_model, load_checkpoints, load_metadata


def read_capacity_from_file(experiment_name):
    """ Read and return the min capacity, max capacity, interpolation, gamma as a tuple if the capacity
        is variable. Otherwise return the constant capacity as is.
        TODO: This is a bit brittle at the moment - We should take a look at fixing this for static beta later.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment, which is the name of the folder that the model is expected to be in.
    """
    meta_data = load_metadata(os.path.join(RES_DIR, experiment_name))

    min_capacity = meta_data['betaB_initC']
    max_capacity = meta_data['betaB_finC']
    interp_capacity = meta_data['betaB_stepsC']
    gamma = meta_data['betaB_G']
    return (min_capacity, max_capacity, interp_capacity, gamma)



def samples(experiment_name, num_samples=1, shuffle=True):
    """ Generate a number of samples from the dataset.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment, which is the name of the folder that the model is expected to be in.

        num_samples : int
            The number of samples to load from the dataset

        shuffle : bool
            Whether or not to shuffle the dataset before drawing samples.
    """
    meta_data = load_metadata(os.path.join(RES_DIR, experiment_name))
    dataset_name = meta_data['dataset']

    data_loader = get_dataloaders(batch_size=1, dataset=dataset_name, shuffle=shuffle)

    data_list = []
    for batch_idx, (new_data, _) in enumerate(data_loader):
        if num_samples == batch_idx:
            break
        data_list.append(new_data)
    return torch.cat(data_list, dim=0)


def get_sample(experiment_name, samples_to_retrieve):
    """ Retrieve a particular sample, or set of samples from the dataset.

        Parameters
        ----------

        experiment_name : str
            The name of the experiment (and the directory that the saved model is located)

        samples_to_retrieve : int or list
            Returns the nth sample. If a list, returns all samples contained in the list.
    """
    dataset = load_metadata(os.path.join(RES_DIR, experiment_name))["dataset"]
    if isinstance(samples_to_retrieve, list):
        num_samples = len(samples_to_retrieve)
    else:
        num_samples = 1
    data_loader = get_dataloaders(dataset, shuffle=False, batch_size=num_samples)

    if num_samples == 1:
        return data_loader.dataset[samples_to_retrieve]

    data_list = []
    for sample_idx in range(num_samples):
        new_data = data_loader.dataset[samples_to_retrieve[sample_idx]]
        data_list.append(new_data)

    return torch.cat(data_list, dim=0)


def snapshot_reconstruction(viz_list, epoch_list, experiment_name, num_samples, dataset, shuffle=True, file_name='snapshot_recon.png'):
    """ Reconstruct some data samples at different stages of training.

        Parameters
        ----------
        viz_list : list
            A list of Visualizer objects

        epoch_list : list
            List of epochs at which the snapshots were taken

        experiment_name : str
            The name of the experiment (and the directory that the saved model is located)

        num_samples : int
            The number of samples to include in the visualisation

        dataset : str
            The name of the dataset that the model was trained on

        shuffle : bool
            Determines if the samples should be drawn from a shuffled dataset or not

        file_name : str
            The name of the PNG file to create
    """
    torch_image_list = []
    data_samples = samples(experiment_name=experiment_name, num_samples=num_samples, shuffle=True)

    # Create original
    numpy_image = viz_list[0].reconstruction_comparisons(data=data_samples, size=(8, 8), exclude_recon=True)
    torch_image_list.append(numpy_image)
    # Now create reconstructions
    for viz in viz_list:
        numpy_image = viz.reconstruction_comparisons(data=data_samples, exclude_original=True)
        torch_image_list.append(numpy_image)

    reconstructions = torch.stack(torch_image_list, dim=0)[:, :, 0, :, :]
    capacity = read_capacity_from_file(experiment_name)

    if isinstance(capacity, tuple):
        capacity_list = np.linspace(capacity[0], capacity[1], capacity[2]).tolist()
    else:
        capacity_list = [capacity] * (viz_list + 1)

    selected_capacities = []
    for epoch_idx in epoch_list:
        selected_capacities.append(capacity_list[epoch_idx])
    reconstructions = torch.reshape(reconstructions, (-1, 1, reconstructions.shape[2], reconstructions.shape[3]))
    save_image(reconstructions.data, file_name, pad_value=(1 - get_background(dataset)))


def parse_arguments():
    """ Set up a command line interface for directing the experiment to be run.
    """
    parser = argparse.ArgumentParser(description="The primary script for running experiments on locally saved models.",
                                     formatter_class=FormatterNoDuplicate)

    # TODO: Do this properly as in main.py
    true_false = ['True', 'False']

    experiment = parser.add_argument_group('Predefined experiments')
    experiment.add_argument('-m', '--model-dir', required=True, type=str,
                            help='The name of the directory in which the model to run has been saved. This should be the name of the experiment')

    visualisation = parser.add_argument_group('Desired Visualisation')

    visualisation_options = ['visualise-dataset', 'random-samples', 'reconstruct-and-traverse', 'traverse-prior', 'traverse-one-latent-dim', 'random-reconstruction',
                             'heat-maps', 'display-avg-KL', 'traverse-posterior', 'show-disentanglement', 'snapshot-recon']
    visualisation.add_argument('-v', '--visualisation',
                               default='random-samples', choices=visualisation_options,
                               help='Predefined visualisation options which can be performed.')
    visualisation.add_argument('-s', '--sweep-dim',
                               default=0, help='The latent dimension to sweep (if applicable)')
    visualisation.add_argument('-n', '--num-samples', type=int,
                               default=1, help='The number of samples to visualise (if applicable).')
    visualisation.add_argument('-u', '--upsample-factor', type=int,
                               default=1, help='The scale factor with which to upsample the image.')
    visualisation.add_argument('-nd', '--num-dims-to-display', type=int,
                               default=10, help='The number of rows to visualise (if applicable).')
    visualisation.add_argument('-nt', '--num-traversal-increments', type=int,
                               default=10, help='The number of columns to visualise (if applicable).')
    visualisation.add_argument('-d', '--display-loss', type=str, default='False', choices=true_false,
                               help='If the loss should be displayed next to the posterior latent traversal dimensions.')
    visualisation.add_argument('-sp', '--select-prior', type=str, default='True', choices=true_false,
                               help='Option for selecting between prior and posterior traversals. If select-prior is False, we use posteriors.')
    visualisation.add_argument('-st', '--show-text', type=str, default='False', choices=true_false,
                               help='Show the KL divergence in the show-disentanglement figure.')
    traversal_type_opts = ['Absolute', 'Gaussian']
    visualisation.add_argument('-tt', '--traversal-type',
                               default='Absolute', choices=traversal_type_opts,
                               help='Predefined options for generating latent dimension traversals.')
    visualisation.add_argument('-mt', '--max-traversal', type=float, default=2,
                               help='The maximum displacement induced by a latent traversal. Symmetrical traversals are assumed.')

    dir_opts = parser.add_argument_group('directory options')
    dir_opts.add_argument('-l', '--log-dir', type=str, default='', help='Path to the log file containing the data to plot.')
    dir_opts.add_argument('-o', '--output-file-name', help='The full path name to use when saving the plot.')

    args = parser.parse_args()

    # TODO: Handle all this more elegantly
    if args.upsample_factor < 1:
        raise Exception('The upsample factor must be greater than or equal to 1')
    if args.max_traversal <= 0:
        raise Exception('The maximum traversal displacement must be greater than zero')

    if args.select_prior == 'True':
        args.select_prior = True
    else:
        args.select_prior = False

    if args.show_text == 'True':
        args.show_text = True
    else:
        args.show_text = False

    if args.display_loss == 'True':
        args.display_loss = True
    else:
        args.display_loss = False

    return args


def main(args):
    """ The primary entry point for carrying out experiments on pretrained models.
    """
    experiment_name = args.model_dir
    meta_data = load_metadata(os.path.join(RES_DIR, experiment_name))
    dataset_name = meta_data['dataset']

    # Assume symmetrical traversals
    traversal_type = args.traversal_type
    if traversal_type == 'Gaussian':
        traversal_range = (0.05, 0.95)
    else:
        traversal_range = (-1 * args.max_traversal, args.max_traversal)

    if args.visualisation == 'snapshot-recon':
        viz_list = []
        epoch_list = []
        model_list = load_checkpoints(directory=os.path.join(RES_DIR, experiment_name))
        for epoch_index, model in model_list:
            model.eval()
            viz_list.append(Visualizer(model=model, model_dir=os.path.join(RES_DIR, experiment_name), dataset=dataset_name, save_images=False))
            epoch_list.append(epoch_index)

    elif not args.visualisation == 'display-avg-KL':
        model = load_model(os.path.join(RES_DIR, experiment_name))
        model.eval()
        viz = Visualizer(
            model=model,
            model_dir=os.path.join(RES_DIR, experiment_name),
            dataset=dataset_name,
            traversal_type=traversal_type,
            traversal_range=traversal_range
            )

    visualisation_options = {
        'visualise-dataset': lambda: viz.visualise_data_set(
            data=samples(experiment_name=experiment_name, num_samples=args.num_samples, shuffle=True),
            file_name=os.path.join(RES_DIR, experiment_name, 'visualise_data_set.png')
            ),
        'random-samples': lambda: viz.generate_samples(
            file_name=os.path.join(RES_DIR, experiment_name, experiment_name + '-samples.png')
            ),
        'traverse-prior': lambda: viz.prior_traversal(
            file_name=os.path.join(RES_DIR, experiment_name, experiment_name + '-prior-traversal.png'),
            num_traversal_increments=args.num_traversal_increments,
            num_dims_to_display=args.num_dims_to_display,
            reorder_latent_dims=True,
            upsample_factor=args.upsample_factor
            ),
        'traverse-one-latent-dim': lambda: viz.latent_traversal_line(
            idx=args.sweep_dim,
            file_name=os.path.join(RES_DIR, experiment_name, experiment_name + '-line-traversal.png')
            ),
        'random-reconstruction': lambda: viz.reconstruction_comparisons(
            data=samples(experiment_name=experiment_name, num_samples=args.num_samples, shuffle=True),
            file_name=os.path.join(RES_DIR, experiment_name, experiment_name + '-random-reconstruction.png')
            ),
        'traverse-posterior': lambda: viz.traverse_posterior(
            data=samples(experiment_name=experiment_name, num_samples=1, shuffle=True),
            display_loss_per_dim=args.display_loss,
            file_name=os.path.join(RES_DIR, experiment_name, experiment_name + '-posterior-traversal.png'),
            num_dims_to_display=args.num_dims_to_display,
            num_traversal_increments=args.num_traversal_increments,
            upsample_factor=args.upsample_factor
            ),
        'reconstruct-and-traverse': lambda: viz.reconstruct_and_traverse(
            reconstruction_data=samples(experiment_name=experiment_name, num_samples=args.num_samples, shuffle=True),
            latent_sweep_data=samples(experiment_name=experiment_name, num_samples=1, shuffle=True),
            file_name=os.path.join(RES_DIR, experiment_name, 'reconstruct-and-traverse.png'), 
            base_directory=os.path.join(RES_DIR, experiment_name),
            select_prior=args.select_prior,
            show_text=args.show_text,
            nr_rows=args.num_dims_to_display,
            num_traversal_increments=args.num_samples
            ),
        'heat-maps': lambda: viz.generate_heat_maps(
            data=samples(experiment_name=experiment_name, num_samples=1024, shuffle=False),
            file_name=os.path.join(RES_DIR, experiment_name, experiment_name + '-heat-maps.png'),
            reorder=True
            ),
        'show-disentanglement': lambda: viz.show_disentanglement_fig2(
            reconstruction_data=samples(experiment_name=experiment_name, num_samples=args.num_samples, shuffle=True),
            latent_sweep_data=samples(experiment_name=experiment_name, num_samples=1, shuffle=True),
            heat_map_data=samples(experiment_name=experiment_name, num_samples=1024, shuffle=False),
            file_name=os.path.join(RES_DIR, experiment_name, 'show-disentanglement.png'), 
            base_directory = os.path.join(RES_DIR, experiment_name),
            select_prior = args.select_prior,
            show_text = args.show_text
            ),
        'display-avg-KL': lambda: LogPlotter(
            log_dir=args.log_dir,
            output_file_name=args.output_file_name
            ),
        'snapshot-recon': lambda: snapshot_reconstruction(
            viz_list=viz_list,
            epoch_list=epoch_list,
            experiment_name=experiment_name,
            num_samples=args.num_samples,
            dataset=dataset_name,
            shuffle=True,
            file_name=os.path.join(RES_DIR, experiment_name, experiment_name + '-snapshot-recon.png')
            )
    }

    return visualisation_options.get(args.visualisation, lambda: "Invalid visualisation option")()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
