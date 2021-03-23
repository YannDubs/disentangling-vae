import torch

from disvae.utils.modelIO import load_model

MODEL_PATH = "results/fid_testing"
MODEL_NAME = "model.pt"
GPU_AVAILABLE = True

model = load_model(directory=MODEL_PATH, is_gpu=GPU_AVAILABLE, filename=MODEL_NAME)
model.eval()

# Insert code to test here
