python run_experiment.py -x betaB_celeba -v traverse_all_latent_dims
mv imgs/all_traversals.png experiments/betaB_celeba

python run_experiment.py -x betaB_chairs -v traverse_all_latent_dims
mv imgs/all_traversals.png experiments/betaB_chairs

python run_experiment.py -x betaB_dsprites -v traverse_all_latent_dims
mv imgs/all_traversals.png experiments/betaB_dsprites

python run_experiment.py -x factor_celeba -v traverse_all_latent_dims
mv imgs/all_traversals.png experiments/factor_celeba

python run_experiment.py -x factor_chairs -v traverse_all_latent_dims
mv imgs/all_traversals.png experiments/factor_chairs

python run_experiment.py -x factor_dsprites -v traverse_all_latent_dims
mv imgs/all_traversals.png experiments/factor_dsprites

