import argparse
import json
import logging
import torch
import numpy as np

from disvae.vae import VAE
from utils.datasets import get_dataloaders
from viz.visualize import Visualizer
from utils.modelIO import load_model
from viz.log_plotter import LogPlotter
from torchvision.utils import save_image
from utils.datasets import get_background
from viz.viz_helpers import add_labels

def read_dataset_from_specs(path_to_specs):
    """ read the spec file from the path given
    """
    # Open specs file
    with open(path_to_specs) as specs_file:
        specs = json.load(specs_file)
    dataset = specs["dataset"]
    return dataset

def read_capacity_from_file(path_to_specs):
    """ Read and return the min capacity, max capacity, interpolation, gamma as a tuple if the capacity
        is variable. Otherwise return the constant capacity as is.
    """
    # Open specs file
    with open(path_to_specs) as specs_file:
        specs = json.load(specs_file)

    capacity = specs["capacity"]
    if isinstance(capacity, list):
        min_capacity = capacity[0]
        max_capacity = capacity[1]
        interp_capacity = capacity[2]
        gamma = capacity[3]
        return (min_capacity, max_capacity, interp_capacity, gamma)
    else:
        return capacity

def samples(experiment_name, logger, num_samples=1, batch_size=1, shuffle=True, specify_index=-1):
    """ generate a number of samples from the dataset
    """
    if not specify_index == -1:
        # Specifying and index to sample therefore the dataset will not be shuffled
        if logger is not None:
            logger.info('Specifying an index to sample therefore the dataset will not be shuffled')
        shuffle = False

    with open('experiments/{}/specs.json'.format(experiment_name)) as spec_file:
        spec_data = json.load(spec_file)
        dataset_name = spec_data['dataset']

        data_loader = get_dataloaders(batch_size=batch_size, dataset=dataset_name, shuffle=shuffle)

        data_list = []
        if not specify_index == -1:
            chosen_sample = data_loader.dataset[specify_index][0]
            chosen_sample = chosen_sample[None,:,:,:]
            data_list.append(chosen_sample)
            num_samples -= 1

        for batch_idx, (new_data, _) in enumerate(data_loader):
            if num_samples == batch_idx:
                break
            data_list.append(new_data)
        return torch.cat(data_list, dim=0)

def snapshot_reconstruction(viz_list, epoch_list, experiment_name, num_samples, dataset, shuffle=True,
                            specify_index=-1, logger=None, file_name='imgs/snapshot_recon.png'):
    """ Reconstruct some data samples at different stages of training. 
    """
    tensor_image_list = []
    data_samples = samples(experiment_name=experiment_name, logger=logger, num_samples=num_samples, shuffle=False, specify_index=specify_index)
    
    # Create original
    tensor_image = viz_list[0].reconstruction_comparisons(data=data_samples, exclude_recon=True)
    tensor_image_list.append(tensor_image)
    # Now create reconstructions
    for viz in viz_list:
        tensor_image = viz.reconstruction_comparisons(data=data_samples, exclude_original=True)
        tensor_image_list.append(tensor_image)

    reconstructions = torch.stack(tensor_image_list, dim=0)

    path_to_specs = 'experiments/{}/specs.json'.format(experiment_name)
    capacity = read_capacity_from_file(path_to_specs)

    if isinstance(capacity, tuple):
        capacity_list = np.linspace(capacity[0], capacity[1], capacity[2]).tolist()
    else:
        capacity_list = [capacity] * (viz_list + 1)

    selected_capacities = []
    for epoch_idx in epoch_list:
        selected_capacities.append(capacity_list[epoch_idx]) 

    save_image(reconstructions.data, file_name, nrow=1, pad_value=(1 - get_background(dataset)))

def parse_arguments():
    """ Set up a command line interface for directing the experiment to be run.
    """
    parser = argparse.ArgumentParser(description="The primary script for running experiments on locally saved models.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    model_dir = parser.add_argument_group('The directory where the model is located.')
    model_dir.add_argument('-m', '--model_dir', default='custom',
                           help='The name of the directory that the model has been saved to. This is usually the name of the experiment.')

    general = parser.add_argument_group('General configuration')
    log_levels = ['critical', 'error', 'warning', 'info', 'debug']
    general.add_argument('-L', '--log-level', help="Logging levels.",
                         default='info',
                         choices=log_levels)

    visualisation = parser.add_argument_group('Desired Visualisation')
    visualisation_options = ['random_samples', 'traverse_all_latent_dims', 'traverse_one_latent_dim', 'random_reconstruction', 
                             'heat_maps', 'display_avg_KL', 'recon_and_traverse_all', 'show_disentanglement', 'snapshot_recon']
    visualisation.add_argument('-v', '--visualisation',
                               default='random_samples', choices=visualisation_options,
                               help='Predefined visualisation options which can be performed.')
    visualisation.add_argument('-s', '--sweep_dim',
                               default=0, help='The latent dimension to sweep (if applicable)')
    visualisation.add_argument('-n', '--num_samples', type=int,
                               default=1, help='The number of samples to visualise (if applicable).')

    dir_opts = parser.add_argument_group('directory options')
    dir_opts.add_argument('-l', '--log_dir', help='Path to the log file containing the data to plot.')
    dir_opts.add_argument('-o', '--output_file_name', help='The full path name to use when saving the plot.')

    args = parser.parse_args()

    if args.log_dir is None:
        args.log_dir = args.model_dir + 'losses.log'

    return args


def main(args):
    """ The primary entry point for carrying out experiments on pretrained models.
    """ 
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())
    stream = logging.StreamHandler()
    stream.setLevel(args.log_level.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    experiment_name = args.model_dir
    dataset = read_dataset_from_specs('experiments/{}/specs.json'.format(experiment_name))

    if args.visualisation == 'snapshot_recon':
        viz_list = []
        epoch_list = []

        model_list = load_model(
            directory='experiments/{}'.format(experiment_name),
            load_snapshots=True
            )
        if not model_list:
            raise Exception('Please target a trained model which has snapshot models stored in the form:\nmodel-0.pt, model-1.pt, etc.')

        for epoch_index, model in model_list:
            model.eval()
            viz_list.append(Visualizer(model=model, model_dir='experiments/{}'.format(experiment_name), dataset=dataset, save_images=False))
            epoch_list.append(epoch_index)

    elif not args.visualisation == 'display_avg_KL':
        model = load_model('experiments/{}'.format(experiment_name))
        model.eval()

        if args.visualisation == 'show_disentanglement':
            reorder_latent_dims = True
        else:
            reorder_latent_dims = False

        viz = Visualizer(model=model, model_dir='experiments/{}'.format(experiment_name), dataset=dataset, reorder_latent_dims=reorder_latent_dims)

    visualisation_options = {
        'random_samples': lambda: viz.samples(),
        'traverse_all_latent_dims': lambda: viz.all_latent_traversals(size=5),
        'traverse_one_latent_dim': lambda: viz.latent_traversal_line(idx=args.sweep_dim),
        'random_reconstruction': lambda: viz.reconstruction_comparisons(
            data=samples(experiment_name=experiment_name, logger=logger, num_samples=args.num_samples, shuffle=True)),
        'recon_and_traverse_all': lambda: viz.recon_and_traverse_all(
            data=samples(experiment_name=experiment_name, logger=logger, num_samples=1, shuffle=True, specify_index=1)),
        'heat_maps': lambda: viz.generate_heat_maps(
            data=samples(experiment_name=experiment_name, logger=logger, num_samples=32 * 32, shuffle=False)),
        'show_disentanglement': lambda: viz.show_disentanglement_fig2(
            reconstruction_data=samples(experiment_name=experiment_name, logger=logger, num_samples=9, shuffle=True),
            latent_sweep_data=samples(experiment_name=experiment_name, logger=logger, num_samples=1, shuffle=True),
            heat_map_data=samples(experiment_name=experiment_name, logger=logger, num_samples=32 * 32, shuffle=False)),
        'display_avg_KL': lambda: LogPlotter(log_dir=args.log_dir, output_file_name=args.output_file_name),
        'snapshot_recon': lambda: snapshot_reconstruction(
            viz_list=viz_list, epoch_list=epoch_list, experiment_name=experiment_name,
            logger=logger, num_samples=args.num_samples, dataset=dataset, shuffle=True)
    }

    return visualisation_options.get(args.visualisation, lambda: "Invalid visualisation option")()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
