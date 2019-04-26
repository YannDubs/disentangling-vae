#!/usr/bin/env bash

# This file is just a copy paste of the commands used to generate
# the visualisation in the report. This script should not be run
# in it's current form.

## 1. Visualising the datasets:
python main_viz.py -m betaB_celeba -v visualise-dataset -n 64
python main_viz.py -m betaB_dsprites -v visualise-dataset -n 64
python main_viz.py -m betaB_mnist -v visualise-dataset -n 64
python main_viz.py -m betaB_chairs -v visualise-dataset -n 64

## 2. Big image with all datsets and core models prior traversals:
# We do 90% of Gaussian for the prior - 5 by 5 images each
python main_viz.py -m VAE_celeba -v traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py -m VAE_chairs -v traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py -m VAE_dsprites -v traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py -m vae_mnist -v traverse-prior -u 2 -tt Gaussian -nt 5 -nd 5

python main_viz.py -m betaB_celeba -v traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py -m betaB_chairs -v traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py -m betaB_dsprites -v traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py -m betaB_mnist -v traverse-prior -u 2 -tt Gaussian -nt 5 -nd 5

python main_viz.py -m betaH_celeba -v traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py -m betaH_chairs -v traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py -m betaH_dsprites -v traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py -m betaH_mnist -v traverse-prior -u 2 -tt Gaussian -nt 5 -nd 5

python main_viz.py -m factor_celeba -v traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py -m factor_chairs -v traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py -m factor_dsprites -v traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py -m factor_mnist -v traverse-prior -u 2 -tt Gaussian -nt 5 -nd 5

python main_viz.py -m batchTC_celeba -v traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py -m batchTC_chairs -v traverse-prior  -tt Gaussian -nt 5 -nd 5
python main_viz.py -m batchTC_dsprites -v traverse-prior  -tt Gaussian -nt 5 -nd 5
python main_viz.py -m batchTC_mnist -v traverse-prior -u 2  -tt Gaussian -nt 5 -nd 5
# Then concatenate with Bart's main_concatenate.py scrpt (which I don't know how to use)


## 3. Show all latents for factor in more detail alongside reconstructions (still a prior traversals with 90%)
python main_viz.py -m factor_celeba -v reconstruct-and-traverse -tt Gaussian -nt 10 -nd 10 -n 10
python main_viz.py -m factor_chairs -v reconstruct-and-traverse  -tt Gaussian -nt 10 -nd 10 -n 10
python main_viz.py -m factor_dsprites -v reconstruct-and-traverse  -tt Gaussian -nt 10 -nd 10 -n 10
python main_viz.py -m factor_celeba_noMI -v reconstruct-and-traverse -tt Gaussian -nt 10 -nd 10 -n 10
python main_viz.py -m l1_factor_celeba -v reconstruct-and-traverse -tt Gaussian -nt 10 -nd 10 -n 10
python main_viz.py -m anneal_factor_celeba -v reconstruct-and-traverse -tt Gaussian -nt 10 -nd 10 -n 10
# We can do these for the more tropical models as well
# We also have the option to show the KL for these

## 4. Replicate Burgress et all.
# Figure 2
# We traverse between [-2, 2]
python main_viz.py -m betaB_dsprites -v show-disentanglement -sp False -st True -n 9
python main_viz.py -m VAE_dsprites -v show-disentanglement -sp False -st True -n 9

# Figure 3
python main_viz.py -m betaB_dsprites -v snapshot-recon -n 8

# Figure 4
# Prior traversals again, this time absolute range [-1.5 1.5]
python main_viz.py -m betaB_chairs -v reconstruct-and-traverse -tt Absolute -nt 10 -nd 10 -n 10 -mt 1.5 -sp True
python main_viz.py -m betaB_dsprites -v reconstruct-and-traverse -tt Absolute -nt 10 -nd 10 -n 10 -mt 1.5 -sp True

