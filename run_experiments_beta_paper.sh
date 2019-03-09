#! /bin/sh


# Figure 2
# - run model: gaussian blob (beta = 1)
python main.py -x vae_blob_x_y
# - run model: gaussian blob (beta = 150)
python main.py -x beta_vae_blob_x_y
# - run experiment: gaussian blob (beta = 1)
python run_experiment.py -x vae_blob_x_y -v traverse_all_latent_dims
# - run experiment: gaussian blob (beta = 150)
python run_experiment.py -x beta_vae_blob_x_y -v traverse_all_latent_dims

# Figure 3
# - run model: dsprites (controlled bottleneck)
python main.py -x beta_vae_dsprite
# - run experiment: dsprites, training-iterations vs KL

# - run experiment: dsprites, Log likelihood vs KL

# - run experiment: dsprites, random reconstruction
python run_experiment.py -x beta_vae_dsprite -v random_reconstruction -n 8

# Figure 4
# - run model: coloured dSprites (controlled capacity increase)
python main.py -x beta_vae_colour_dsprite
# - run model: 3D Chairs (controlled capacity increase)
python main.py -x beta_vae_chairs
# - run experiment: coloured dSprites reconstruction
python run_experiment -x beta_vae_colour_dsprite -v random_reconstruction -n 10
# - run experiment: coloured dSprites latent dimensions
python run_experiment -x beta_vae_colour_dsprite -v traverse_all_latent_dims
# - run experiment: 3D Chairs latent dimensions
python run_experiment -x beta_vae_chairs -v random_reconstruction -n 10
# - run experiment: 3D Chairs latent dimensions
python run_experiment -x beta_vae_chairs -v traverse_all_latent_dims


