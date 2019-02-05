# Disentangled VAE 

## Install

```
pip install -r requirements.txt
```

## Run

```
python main.py -h
```

## TO - DO:
@Bart / @ Dave : 
    - modify disvae.encoder.py + disvae.decoder.py + disvae.vae.py + intialization.py + `_loss_function` (disvae.trainer.py) to replicate the results in beta VAE
    - to be sure that your code does exactly the same as the paper you will have to manually download their data sets (although teh dataloaders are already coded) and replicate their results
@Aleco : 
    - add the experiment scripts in `experiments`
    - define the correct experiment parameters in `default_experiment` and `main` functions of `main.py`


? : 
- download all datasets other than MNIST in dataloaders (although dataloaders already here) by testing if folder doesn't exist (witry try / except statement) and downloading if it doesn't (typically using `urllib`).
- test CUDA
