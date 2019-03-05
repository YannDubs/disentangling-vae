import argparse
import json
import torch
import matplotlib.pyplot as plt
import numpy as np

from disvae.vae import VAE
from utils.dataloaders import get_dataloaders
from viz.visualize import Visualizer as Viz


def samples(experiment_name, num_samples=1, batch_size=1, shuffle=True):
    """ generate a number of samples from the dataset
    """
    with open('experiments/{}/specs.json'.format(experiment_name)) as spec_file:
        spec_data = json.load(spec_file)
        dataset_name = spec_data['dataset']

        data_loader, _ = get_dataloaders(batch_size=batch_size, dataset=dataset_name, shuffle=shuffle)

        for batch_idx, (test_data, label) in enumerate(data_loader):
            if num_samples == batch_idx:
                break
            if batch_idx == 0:
                data = test_data
            else:
                data = torch.cat(tensors=(data, test_data))
        return data

def sprite_location():
    """ Read the sprite (x,y) locations as a 2 column numpy array.
    """
    sprite_dir = './data/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    dataset_zip = np.load(sprite_dir)
    latents_values = dataset_zip['latents_values']
    x_y_posn = latents_values[:32*32, -2:]

    return x_y_posn

def parse_arguments():
    """ Set up a command line interface for directing the experiment to be run.
    """
    parser = argparse.ArgumentParser(description="The primary script for running experiments on locally saved models.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    experiment = parser.add_argument_group('Predefined experiments')
    experiment_options = ['custom', 'vae_blob_x_y', 'beta_vae_blob_x_y', 'beta_vae_dsprite',
                          'beta_vae_celeba', 'beta_vae_colour_dsprite', 'beta_vae_chairs']
    experiment.add_argument('-x', '--experiment',
                            default='custom', choices=experiment_options,
                            help='Predefined experiments to run. If not `custom` this will set the correct other arguments.')

    visualisation = parser.add_argument_group('Desired Visualisation')
    visualisation_options = ['random_samples', 'traverse_all_latent_dims', 'traverse_one_latent_dim', 'random_reconstruction', 'heat_maps']
    visualisation.add_argument('-v', '--visualisation',
                               default='random_samples', choices=visualisation_options,
                               help='Predefined visualisation options which can be performed.')
    visualisation.add_argument('-s', '--sweep_dim',
                            default=0, help='The latent dimension to sweep (if applicable)')
    visualisation.add_argument('-n', '--num_samples', type=int,
                            default=1, help='The number of samples to visualise (if applicable).')

    args = parser.parse_args()
    return args


def main(args):
    """ The primary entry point for carrying out experiments on pretrained models.
    """
    experiment_name = args.experiment

    model = torch.load('experiments/{}/model.pt'.format(experiment_name))
    model.eval()
    viz = Viz(model)

    visualisation_options = {
        'random_samples': lambda: viz.samples(),
        'traverse_all_latent_dims': lambda: viz.all_latent_traversals(),
        'traverse_one_latent_dim': lambda: viz.latent_traversal_line(idx=args.sweep_dim),
        'random_reconstruction': lambda: viz.reconstructions(data=samples(experiment_name=experiment_name, num_samples=args.num_samples, shuffle=True)),
        'heat_maps': lambda: viz.generate_heat_maps(data=samples(experiment_name=experiment_name, num_samples=32*32, shuffle=False))
    }

    return visualisation_options.get(args.visualisation, lambda: "Invalid visualisation option")()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
