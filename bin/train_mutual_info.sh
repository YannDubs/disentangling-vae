#!/usr/bin/env bash

# Remove `&` at the end if don't want all in parallel

loss="btcvae"
for dataset in dsprites celeba
do
    for alpha in -5 -1 0 1 5
    do
        python main.py "$loss"_"$dataset"_a$alpha -x "$loss"_"$dataset" --btcvae-A $alpha --no-progress-bar &
    done
done
