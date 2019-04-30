#!/usr/bin/env bash

# RUN every element in the blocks in parallel ! Remove `&` at the end if don't
# want all in parallel

logger="metrics_all.out"
echo "STARTING" > $logger
for dataset in dsprites # only dsprites currently has true factor of var.
do
    for loss in btcvae betaH betaB factor VAE
    do
        echo " " >> $logger
        echo $loss >> $logger
        python main.py "$loss"_"$dataset" -x "$loss"_"$dataset" --is-metrics --is-eval-only  --no-test --no-progress-bar &
    done
    wait
done
