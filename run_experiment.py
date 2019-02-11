import argparse
import json
import torch
import matplotlib.pyplot as plt

from disvae.vae import VAE
from utils.dataloaders import get_dataloaders
from viz.visualize import Visualizer as Viz    


def random_samples(experiment_name, num_samples=1):
    """ generate a number of random samples from the dataset
    """
    with open('trained_models/{}/specs.json'.format(experiment_name)) as spec_file:
        spec_data = json.load(spec_file)
        dataset_name = spec_data['dataset']

        _, data_loader = get_dataloaders(batch_size=num_samples, dataset=dataset_name, shuffle=True)

        for (data, _) in data_loader:
            return data


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
    visualisation_options = ['random_samples', 'traverse_all_latent_dims', 'traverse_one_latent_dim', 'random_reconstruction']
    visualisation.add_argument('-v', '--visualisation',
                            default='random_samples', choices=visualisation_options,
                            help='Predefined visaulisation options which can be performed.')
    visualisation.add_argument('-s', '--sweep_dim',
                            default=0, help='The latent dimension to sweep (if applicable)')
    visualisation.add_argument('-n', '--num_samples',
                            default=1, help='The number of samples to visualise (if applicable).')

    args = parser.parse_args()
    return args

def main(args):
    """ The primary entry point for carrying out experiments on pretrained models.
    """
    experiment_name = args.experiment

    model = torch.load('trained_models/{}/model.pt'.format(experiment_name))
    model.eval()
    viz = Viz(model)

    visualisation_options = {
        'random_samples': lambda: viz.samples(),
        'traverse_all_latent_dims': lambda: viz.all_latent_traversals(),
        'traverse_one_latent_dim': lambda: viz.latent_traversal_line(idx=args.sweep_dim),
        'random_reconstruction': lambda: viz.reconstructions(data=random_samples(experiment_name=experiment_name, num_samples=args.num_samples))
    }

    return visualisation_options.get(args.visualisation, lambda : "Invalid visualisation option")()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
