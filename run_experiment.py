import argparse
import json
import torch

from disvae.vae import VAE
from utils.datasets import get_dataloaders
from viz.visualize import Visualizer as Viz
from utils.modelIO import load_model
from viz.log_plotter import LogPlotter


def samples(experiment_name, num_samples=1, batch_size=1, shuffle=True):
    """ generate a number of samples from the dataset
    """
    with open('experiments/{}/specs.json'.format(experiment_name)) as spec_file:
        spec_data = json.load(spec_file)
        dataset_name = spec_data['dataset']

        data_loader = get_dataloaders(batch_size=batch_size, dataset=dataset_name, shuffle=shuffle)

        data_list = []
        for batch_idx, (new_data, label) in enumerate(data_loader):
            if num_samples == batch_idx:
                break
            data_list.append(new_data)
        return torch.cat(data_list)


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
    visualisation_options = ['random_samples', 'traverse_all_latent_dims', 'traverse_one_latent_dim', 'random_reconstruction',
                             'heat_maps', 'display_avg_KL']
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
    return args


def main(args):
    """ The primary entry point for carrying out experiments on pretrained models.
    """
    if not args.visualisation == 'display_avg_KL':
        experiment_name = args.experiment

        model = load_model('experiments/{}'.format(experiment_name))
        model.eval()
        viz = Viz(model)

    visualisation_options = {
        'random_samples': lambda: viz.samples(),
        'traverse_all_latent_dims': lambda: viz.all_latent_traversals(),
        'traverse_one_latent_dim': lambda: viz.latent_traversal_line(idx=args.sweep_dim),
        'random_reconstruction': lambda: viz.reconstructions(data=samples(experiment_name=experiment_name, num_samples=args.num_samples, shuffle=True)),
        'heat_maps': lambda: viz.generate_heat_maps(data=samples(experiment_name=experiment_name, num_samples=32 * 32, shuffle=False)),
        'display_avg_KL': lambda: LogPlotter(log_dir=args.log_dir, output_file_name=args.output_file_name)
    }

    return visualisation_options.get(args.visualisation, lambda: "Invalid visualisation option")()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
