#!/usr/bin/env bash

# RUN every element in the blocks in parallel ! Remove `&` at the end if don't
# want all in parallel

for dataset in celeba dsprites chairs
do
    for loss in betaH betaB factor batchTC
    do
        python main.py "$loss"_"$dataset" -x "$loss"_"$dataset"  --no-progress-bar &
    done
    wait
done
