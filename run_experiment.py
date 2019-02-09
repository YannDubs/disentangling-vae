import argparse
import json
import torch
import matplotlib.pyplot as plt

from disvae.vae import VAE
from utils.dataloaders import get_dataloaders
from viz.visualize import Visualizer as Viz    


def retrieve_test_data(experiment_name, batch_size=32):
    """ Get the test set for the relevant experiment..
    """
    with open('trained_models/{}/specs.json'.format(experiment_name)) as spec_file:
        spec_data = json.load(spec_file)
        dataset_name = spec_data['dataset']

    _, test = get_dataloaders(batch_size=batch_size, dataset=dataset_name)
    return test


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
    args = parser.parse_args()
    return args

def main(args):
    """ The primary entry point for carrying out experiments on pretrained models.
    """
    experiment_name = args.experiment

    model = torch.load('trained_models/{}/model.pt'.format(experiment_name))
    model.eval()
    viz = Viz(model)
    # viz.save_images = False  # Return tensors instead of saving images

    samples = viz.samples()
    traversals = viz.all_latent_traversals()

    # test_data = retrieve_test_data(experiment_name)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
