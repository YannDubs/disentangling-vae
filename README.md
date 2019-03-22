# Disentangled VAE 

Work In Progress...

This repository contains code to investigate disentangling in VAE as well as compare 5 different losses using a single model:

* **Standard VAE Loss** from [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
* **β-VAE<sub>H</sub>** from [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/pdf?id=Sy2fzU9gl)
* **β-VAE<sub>B</sub>** from [Understanding disentangling in β-VAE](https://arxiv.org/abs/1804.03599)
* **FactorVAE** from [Disentangling by Factorising](https://arxiv.org/pdf/1802.05983.pdf)
* **β-TCVAE** from [Isolating Sources of Disentanglement in Variational Autoencoders](https://arxiv.org/abs/1802.04942)

Notes:
- Tested for python >= 3.6
- Tested for CPU and GPU

## Install

```
# clone repo
pip install -r requirements.txt
```

## Run

Use `python main.py <saving-name> <param>` to train and/or evaluate a model. 

### Output
This will create a directory `experiments/<saving-name>/` which will contain:

* **model.pt**: The model at the end of training. 
* **model-**`i`**.pt**: Model checkpoint after `i` iterations. By default saves every 10.
* **specs.json**: The parameters used to run the program (default and modified with CLI).
* **training.gif**: GIF of laten traversals of the latent dimensions Z at every epoch.
* **train_losses.log**: All (sub-)losses computed during training.
* **test_losses.log**: ALl (sub-)losses computed at the end of training with the model in evaluate mode (no sampling). Only if not `--no-test`
* **disentanglement_metric.pth**: dictionary of tensors (pytorch object) containing the MIG metric. Only if `--is-metric` (slow).


### Help
```
usage: main.py [-h] [-L {critical,error,warning,info,debug}]
               [--no-progress-bar] [--no-cuda] [-s SEED]
               [--checkpoint-every CHECKPOINT_EVERY]
               [-d {mnist,celeba,chairs,dsprites,fashion}]
               [-x {custom,debug,betaH_celeba,betaH_chairs,betaH_dsprites,betaB_celeba,betaB_chairs,betaB_dsprites,factor_celeba,factor_chairs,factor_dsprites,batchTC_celeba,batchTC
_chairs,batchTC_dsprites}]
               [-e EPOCHS] [-b BATCH_SIZE] [--lr LR] [-m {Burgess}]
               [-z LATENT_DIM] [-l {VAE,betaH,betaB,factor,batchTC}]
               [--betaH-B BETAH_B] [--betaB-initC BETAB_INITC]
               [--betaB-finC BETAB_FINC] [--betaB-stepsC BETAB_STEPSC]
               [--betaB-G BETAB_G] [--factor-G FACTOR_G] [--no-mutual-info]
               [--lr-disc LR_DISC] [--batchTC-A BATCHTC_A]
               [--batchTC-G BATCHTC_G] [--batchTC-B BATCHTC_B] [--no-mss]
               [--is-eval-only] [--is-metrics] [--no-test]
               [-eb EVAL_BATCHSIZE]
               name

PyTorch implementation and evaluation of disentangled Variational AutoEncoders
and metrics.

optional arguments:
  -h, --help            show this help message and exit

General options:
  name                  Name of the model for storing or loading purposes.
  -L, --log-level {critical,error,warning,info,debug}
                        Logging levels. (default: info)
  --no-progress-bar     Disables progress bar. (default: False)
  --no-cuda             Disables CUDA training, even when have one. (default:
                        False)
  -s, --seed SEED       Random seed. Can be `None` for stochastic behavior.
                        (default: 1234)

Training specific options:
  --checkpoint-every CHECKPOINT_EVERY
                        Save a checkpoint of the trained model every n epoch.
                        (default: 10)
  -d, --dataset {mnist,celeba,chairs,dsprites,fashion}
                        Path to training data. (default: mnist)
  -x, --experiment {custom,debug,betaH_celeba,betaH_chairs,betaH_dsprites,betaB_celeba,betaB_chairs,betaB_dsprites,factor_celeba,factor_chairs,factor_dsprites,batchTC_celeba,batchTC
_chairs,batchTC_dsprites}
                        Predefined experiments to run. If not `custom` this
                        will overwrite some other arguments. (default: custom)
  -e, --epochs EPOCHS   Maximum number of epochs to run for. (default: 100)
  -b, --batch-size BATCH_SIZE
                        Batch size for training. (default: 64)
  --lr LR               Learning rate. (default: 0.0005)

Model specfic options:
  -m, --model-type {Burgess}
                        Type of encoder and decoder to use. (default: Burgess)
  -z, --latent-dim LATENT_DIM
                        Dimension of the latent variable. (default: 10)
  -l, --loss {VAE,betaH,betaB,factor,batchTC}
                        Type of VAE loss function to use. (default: betaB)

BetaH specific parameters:
  --betaH-B BETAH_B     Weight of the KL (beta in the paper). (default: 4)

BetaB specific parameters:
  --betaB-initC BETAB_INITC
                        Starting annealed capacity. (default: 0)
  --betaB-finC BETAB_FINC
                        Final annealed capacity. (default: 25)
  --betaB-stepsC BETAB_STEPSC
                        Number of training iterations for interpolating C.
                        (default: 100000)
  --betaB-G BETAB_G     Weight of the KL divergence term (gamma in the paper).
                        (default: 1000)

factor VAE specific parameters:
  --factor-G FACTOR_G   Weight of the TC term (gamma in the paper). (default:
                        10)
  --no-mutual-info      Remove mutual information. (default: False)
  --lr-disc LR_DISC     Learning rate of the discriminator. (default: 0.0005)

batchTC specific parameters:
  --batchTC-A BATCHTC_A
                        Weight of the MI term (alpha in the paper). (default:
                        1)
  --batchTC-G BATCHTC_G
                        Weight of the dim-wise KL term (gamma in the paper).
                        (default: 1)
  --batchTC-B BATCHTC_B
                        Weight of the TC term (beta in the paper). (default:
                        11)
  --no-mss              Whether to use minibatch weighted sampling instead of
                        stratified.` (default: False)

Evaluation specific options:
  --is-eval-only        Whether to only evaluate using precomputed model
                        `name`. (default: False)
  --is-metrics          Whether to compute the disentangled metrcics. 
                        Currently only possible with `dsprites` as it is the
                        only dataset with known true factors of variations.
                        (default: False)
  --no-test             Whether not to compute the test losses.` (default:
                        False)
  -eb, --eval-batchsize EVAL_BATCHSIZE
                        Batch size for evaluation. (default: 1000)
```

## Data

Current datasets that can be used:
- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)
- [3D Chairs](https://www.di.ens.fr/willow/research/seeing3Dchairs)
- [Celeba](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [2D Shapes / Dsprites](https://github.com/deepmind/dsprites-dataset/)

The dataset will be downloaded the first time you try running it and will be stored in `data` for future uses. The download will take time and might not work anymore if the download links change. In this case either:

1. Open an issue 
2. Change the URLs (`urls["train"]`) for the dataset you want in `utils/datasets.py` (please open a PR in this case :) )
3. Download by hand the data and save it with the same names (not recommended)
