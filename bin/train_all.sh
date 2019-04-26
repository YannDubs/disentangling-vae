#!/usr/bin/env bash

# RUN every element in the blocks in parallel ! Remove `&` at the end if don't
# want all in parallel

logger="train_all.out"
for loss in betaB factor betaH VAE #factor # batchTC
do
    echo " " >> $logger
    echo $loss >> $logger
    for dataset in celeba dsprites chairs mnist
    do
        echo $dataset >> $logger
        python main.py "$loss"_"$dataset" -x "$loss"_"$dataset"  --no-progress-bar &
    done
    wait
done

