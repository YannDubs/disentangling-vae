#!/usr/bin/env bash

# This file is just a copy paste of the commands used to generate
# the visualisation in the report. This script should not be run
# in it's current form.

## 1. Visualising the datasets:
python main_viz.py betaB_celeba visualise-dataset -n 64
python main_viz.py betaB_dsprites visualise-dataset -n 64
python main_viz.py betaB_mnist visualise-dataset -n 64
python main_viz.py betaB_chairs visualise-dataset -n 64

## 2. Big image with all datsets and core models prior traversals:
# We do 90% of Gaussian for the prior - 5 by 5 images each
python main_viz.py VAE_celeba traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py VAE_chairs traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py VAE_dsprites traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py vae_mnist traverse-prior -u 2 -tt Gaussian -nt 5 -nd 5

python main_viz.py betaB_celeba traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py betaB_chairs traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py betaB_dsprites traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py betaB_mnist traverse-prior -u 2 -tt Gaussian -nt 5 -nd 5

python main_viz.py betaH_celeba traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py betaH_chairs traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py betaH_dsprites traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py betaH_mnist traverse-prior -u 2 -tt Gaussian -nt 5 -nd 5

python main_viz.py factor_celeba traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py factor_chairs traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py factor_dsprites traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py factor_mnist traverse-prior -u 2 -tt Gaussian -nt 5 -nd 5

python main_viz.py batchTC_celeba traverse-prior -tt Gaussian -nt 5 -nd 5
python main_viz.py batchTC_chairs traverse-prior  -tt Gaussian -nt 5 -nd 5
python main_viz.py batchTC_dsprites traverse-prior  -tt Gaussian -nt 5 -nd 5
python main_viz.py batchTC_mnist traverse-prior -u 2  -tt Gaussian -nt 5 -nd 5
# Then concatenate with Bart's main_concatenate.py scrpt (which I don't know how to use)


## 3. Show all latents for factor in more detail alongside reconstructions (still a prior traversals with 90%)
python main_viz.py factor_celeba reconstruct-and-traverse -tt Gaussian -nt 10 -nd 10 -n 10
python main_viz.py factor_chairs reconstruct-and-traverse  -tt Gaussian -nt 10 -nd 10 -n 10
python main_viz.py factor_dsprites reconstruct-and-traverse  -tt Gaussian -nt 10 -nd 10 -n 10
python main_viz.py factor_celeba_noMI reconstruct-and-traverse -tt Gaussian -nt 10 -nd 10 -n 10
python main_viz.py l1_factor_celeba reconstruct-and-traverse -tt Gaussian -nt 10 -nd 10 -n 10
python main_viz.py anneal_factor_celeba reconstruct-and-traverse -tt Gaussian -nt 10 -nd 10 -n 10
# We can do these for the more tropical models as well
# We also have the option to show the KL for these

## 4. Replicate Burgress et all.
# Figure 2
# We traverse between [-2, 2]
python main_viz.py betaB_dsprites show-disentanglement -sp False -st True -n 9
python main_viz.py VAE_dsprites show-disentanglement -sp False -st True -n 9

# Figure 3
python main_viz.py betaB_dsprites snapshot-recon -n 8

# Figure 4
# Prior traversals again, this time absolute range [-1.5 1.5]
python main_viz.py betaB_chairs reconstruct-and-traverse -tt Absolute -nt 10 -nd 10 -n 10 -mt 1.5 -sp True
python main_viz.py betaB_dsprites reconstruct-and-traverse -tt Absolute -nt 10 -nd 10 -n 10 -mt 1.5 -sp True

