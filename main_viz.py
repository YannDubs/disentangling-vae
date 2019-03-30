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

        Parameters
        ----------
        experiment_name : str
            The name of the experiment, which is the name of the folder that the model is expected to be in.
    """
    meta_data = load_metadata(os.path.join(RES_DIR, experiment_name))
    capacity = meta_data['capacity']

    if isinstance(capacity, list):
        min_capacity = capacity[0]
        max_capacity = capacity[1]
        interp_capacity = capacity[2]
        gamma = capacity[3]
        return (min_capacity, max_capacity, interp_capacity, gamma)
    else:
        return capacity


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

    data_loader = get_dataloaders(batch_size=num_samples, dataset=dataset_name, shuffle=shuffle)

    data_list = []
    for batch_idx, (new_data, _) in enumerate(data_loader):
        if num_samples == batch_idx:
            break
        data_list.append(new_data)
    return torch.cat(data_list, dim=0)


def get_sample(experiment_name, samples_to_retrieve):
    """ Retrieve a particular sample, or set of samples from the dataset.
    """
    dataset = load_metadata(os.path.join(RES_DIR, experiment_name))["dataset"]
    if isinstance(samples_to_retrieve, list):
        num_samples = len(samples_to_retrieve)
    else:
        num_samples = 1
    data_loader = get_dataloaders(dataset, shuffle=False, batch_size=num_samples)

    data_list = []
    for sample_idx in range(num_samples):

        if num_samples == 1:
            return data_loader.dataset[samples_to_retrieve]

        new_data = data_loader.dataset[samples_to_retrieve[sample_idx]]
        data_list.append(new_data)

    return torch.cat(data_list, dim=0)


def snapshot_reconstruction(viz_list, epoch_list, experiment_name, num_samples, dataset, shuffle=True, file_name='snapshot_recon.png'):
    """ Reconstruct some data samples at different stages of training.
    """
    tensor_image_list = []
    data_samples = samples(experiment_name=experiment_name, num_samples=num_samples, shuffle=True)

    # Create original
    tensor_image = viz_list[0].reconstruction_comparisons(data=data_samples, exclude_recon=True)
    tensor_image_list.append(tensor_image)
    # Now create reconstructions
    for viz in viz_list:
        tensor_image = viz.reconstruction_comparisons(data=data_samples, exclude_original=True)
        tensor_image_list.append(tensor_image)

    reconstructions = torch.stack(tensor_image_list, dim=0)
    capacity = read_capacity_from_file(experiment_name)

    if isinstance(capacity, tuple):
        capacity_list = np.linspace(capacity[0], capacity[1], capacity[2]).tolist()
    else:
        capacity_list = [capacity] * (viz_list + 1)

    selected_capacities = []
    for epoch_idx in epoch_list:
        selected_capacities.append(capacity_list[epoch_idx])

    # traversal_images_with_text = add_labels(
    #     label_name='C',
    #     tensor=reconstructions,
    #     num_rows=1,
    #     sorted_list=selected_capacities,
    #     dataset=dataset)

    # traversal_images_with_text.save(file_name)
    save_image(reconstructions.data, file_name, nrow=1, pad_value=(1 - get_background(dataset)))


def parse_arguments():
    """ Set up a command line interface for directing the experiment to be run.
    """
    parser = argparse.ArgumentParser(description="The primary script for running experiments on locally saved models.",
                                     formatter_class=FormatterNoDuplicate)

    experiment = parser.add_argument_group('Predefined experiments')
    experiment.add_argument('-m', '--model_dir', required=True, type=str,
                            help='The name of the directory in which the model to run has been saved. This should be the name of the experiment')

    visualisation = parser.add_argument_group('Desired Visualisation')
    visualisation_options = ['random_samples', 'traverse_prior', 'traverse_one_latent_dim', 'random_reconstruction',
                             'heat_maps', 'display_avg_KL', 'traverse_posterior', 'show_disentanglement', 'snapshot_recon']
    visualisation.add_argument('-v', '--visualisation',
                               default='random_samples', choices=visualisation_options,
                               help='Predefined visualisation options which can be performed.')
    visualisation.add_argument('-s', '--sweep_dim',
                               default=0, help='The latent dimension to sweep (if applicable)')
    visualisation.add_argument('-n', '--num_samples', type=int,
                               default=1, help='The number of samples to visualise (if applicable).')
    visualisation.add_argument('-d', '--display_loss', type=bool, default=False,
                               help='If the loss should be displayed next to the posterior latent traversal dimensions.')

    dir_opts = parser.add_argument_group('directory options')
    dir_opts.add_argument('-l', '--log_dir', help='Path to the log file containing the data to plot.')
    dir_opts.add_argument('-o', '--output_file_name', help='The full path name to use when saving the plot.')

    args = parser.parse_args()
    return args


def main(args):
    """ The primary entry point for carrying out experiments on pretrained models.
    """
    experiment_name = args.model_dir
    meta_data = load_metadata(os.path.join(RES_DIR, experiment_name))
    dataset_name = meta_data['dataset']

    if args.visualisation == 'snapshot_recon':
        viz_list = []
        epoch_list = []
        model_list = load_checkpoints(directory=os.path.join(RES_DIR, experiment_name))
        for epoch_index, model in model_list:
            model.eval()
            viz_list.append(Visualizer(model=model, model_dir=os.path.join(RES_DIR, experiment_name), dataset=dataset_name, save_images=False))
            epoch_list.append(epoch_index)

    elif not args.visualisation == 'display_avg_KL':
        model = load_model(os.path.join(RES_DIR, experiment_name))
        model.eval()
        viz = Visualizer(model=model, model_dir=os.path.join(RES_DIR, experiment_name), dataset=dataset_name)

    visualisation_options = {
        'random_samples': lambda: viz.samples(),
        'traverse_prior': lambda: viz.prior_traversal(),
        'traverse_one_latent_dim': lambda: viz.latent_traversal_line(idx=args.sweep_dim),
        'random_reconstruction': lambda: viz.reconstruction_comparisons(
            data=samples(experiment_name=experiment_name, num_samples=args.num_samples, shuffle=True)),
        'traverse_posterior': lambda: viz.traverse_posterior(
            data=samples(experiment_name=experiment_name, num_samples=1, shuffle=True), display_loss_per_dim=args.display_loss),
        'heat_maps': lambda: viz.generate_heat_maps(
            data=samples(experiment_name=experiment_name, num_samples=32 * 32, shuffle=False)),
        'show_disentanglement': lambda: viz.show_disentanglement_fig2(
            latent_sweep_data=samples(experiment_name=experiment_name, num_samples=1, shuffle=True),
            heat_map_data=samples(experiment_name=experiment_name, num_samples=32 * 32, shuffle=False)),
        'display_avg_KL': lambda: LogPlotter(log_dir=args.log_dir, output_file_name=args.output_file_name),
        'snapshot_recon': lambda: snapshot_reconstruction(
            viz_list=viz_list, epoch_list=epoch_list, experiment_name=experiment_name,
            num_samples=args.num_samples, dataset=dataset_name, shuffle=True)
    }

    return visualisation_options.get(args.visualisation, lambda: "Invalid visualisation option")()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
