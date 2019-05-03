#!/usr/bin/env bash

# RUN every element in the blocks in parallel ! Remove `&` at the end if don't
# want all in parallel

logger="plot_all.out"
echo "STARTING" > $logger


# many idcs sameas https://github.com/1Konny/FactorVAE/blob/master/solver.py
# to compare
cherry_celeba_idcs="88413 176606 179144 32260 191281 143307 101535 70059 87889 131612 "
cherry_mnist_idcs="1 40 25 7 92 41001 90 41002 823 41219" # take every number in order
cherry_dsprites_idcs="92595 339150 656090" #take every shape: square ellipse heart
cherry_chairs_idcs="40919 5172 22330"


echo "### GIF GRID ###" >> $logger
kwargs="-s 1234 -c 3 -r 5 -t 2"
for loss in factor btcvae betaB betaH VAE
do
    echo " " >> $logger
    echo $loss >> $logger

    python main_viz.py "$loss"_celeba gif-traversals -i $cherry_celeba_idcs $kwargs &
    python main_viz.py "$loss"_chairs gif-traversals -i $cherry_chairs_idcs $kwargs &
    python main_viz.py "$loss"_mnist gif-traversals -u 2 -i $cherry_mnist_idcs $kwargs &
    python main_viz.py "$loss"_dsprites gif-traversals -i $cherry_dsprites_idcs $kwargs &

    wait
done

python << END
from utils.viz_helpers import plot_grid_gifs
grid_files = [["results/{}_{}/posterior_traversals.gif".format(loss,data)
               for data in ["dsprites","celeba","chairs", "mnist"]]
              for loss in ["VAE", "betaH", "betaB", "factor", "btcvae"]]
plot_grid_gifs("results/grid_posteriors.gif", grid_files)
END


# Has to do geenral plots after the gif grid as don't want the previous temporary plots
echo "### General Plots ###" >> $logger
kwargs="-s 1234 -c 10 -r 10 -t 2 --is-show-loss --is-posterior"
for loss in  factor btcvae betaB betaH VAE
do
    echo " " >> $logger
    echo $loss >> $logger

    python main_viz.py "$loss"_celeba all -i $cherry_celeba_idcs $kwargs &
    python main_viz.py "$loss"_chairs all -i $cherry_chairs_idcs $kwargs &
    python main_viz.py "$loss"_mnist all -i $cherry_mnist_idcs $kwargs &
    python main_viz.py "$loss"_dsprites all -i $cherry_dsprites_idcs $kwargs &

    wait
done

python main_viz.py best_celeba all -i $cherry_celeba_idcs $kwargs
python main_viz.py best_dsprites all -i $cherry_dsprites_idcs $kwargs


echo "### GIF GRID MUTUAL INFO ###" >> $logger
kwargs="-s 1234 -c 3 -r 5 -t 2"
for dataset in dsprites celeba
do
    echo " " >> $logger
    echo $dataset >> $logger

    for alpha in -5 -1 0 1 5
    do
        cherry=cherry_"$dataset"_idcs
        python main_viz.py btcvae_"$dataset"_a$alpha gif-traversals -i ${!cherry} $kwargs &
    done
    wait
done

python << END
from utils.viz_helpers import plot_grid_gifs
grid_files = [["results/btcvae_{}_a{}/posterior_traversals.gif".format(data, alpha)
               for data in ["dsprites","celeba"]]
              for alpha in [-5, -1, 0, 1, 5]]
plot_grid_gifs("results/grid_posteriors_mutual_info.gif", grid_files)
END



