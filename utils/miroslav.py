import wandb
import os
import itertools
from sklearn.cluster import KMeans
from sklearn import manifold
from sklearn import decomposition
import sklearn.metrics
from tqdm import tqdm
import torch
import itertools
import matplotlib.pyplot as plt
import umap
import numpy as np
from tqdm import tqdm

def wandb_auth(fname: str = "nas_key.txt", dir_path=None):
  gdrive_path = "/content/drive/MyDrive/colab/wandb/nas_key.txt"
  if "WANDB_API_KEY" in os.environ:
      wandb_key = os.environ["WANDB_API_KEY"]
  elif os.path.exists(os.path.abspath("~" + os.sep + ".wandb" + os.sep + fname)):
      # This branch does not seem to work as expected on Paperspace - it gives '/storage/~/.wandb/nas_key.txt'
      print("Retrieving WANDB key from file")
      f = open("~" + os.sep + ".wandb" + os.sep + fname, "r")
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key
  elif os.path.exists("/root/.wandb/"+fname):
      print("Retrieving WANDB key from file")
      f = open("/root/.wandb/"+fname, "r")
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key

  elif os.path.exists(
      os.path.expandvars("%userprofile%") + os.sep + ".wandb" + os.sep + fname
  ):
      print("Retrieving WANDB key from file")
      f = open(
          os.path.expandvars("%userprofile%") + os.sep + ".wandb" + os.sep + fname,
          "r",
      )
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key
  elif os.path.exists(gdrive_path):
      print("Retrieving WANDB key from file")
      f = open(gdrive_path, "r")
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key
  elif os.path.exists(os.path.join(dir_path, fname)):
      print(f"Retrieving WANDB key from file at {os.path.join(dir_path, fname)}")
      f = open(os.path.join(dir_path, fname), "r")
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key
  wandb.login()

def graph_latent_samples(samples, labels):
    fig = plt.figure()
    # fig, ax = plt.subplots()
    plt.scatter(samples[:,0], samples[:,1],
        c=list(itertools.chain.from_iterable(labels)),
        cmap=plt.cm.get_cmap('jet', 10))
    plt.colorbar()
    return fig

def latent_metrics(true_data, labels, embedded_data):
    results = {}
    results["cluster"] = cluster_metric(true_data, labels, 5)

    return results
    

def star_shape(dset):
    periods = [1, 32, 32*32, 32*32*40, 32*32*40*6]
    base_idx = [16,16,20, 3, 1]
    limits = [32, 32, 40, 6, 3]
    traversals = [[] for _ in range(5)]
    for i in range(5):
        traversals[i] = [sum([j*periods[k] if k == i else base_idx[k]*periods[k] for k in range(5)]) for j in range(limits[i])]

    return traversals


def latent_viz(model, loader, dataset_name, raw_dataset, steps=100, device='cuda' if torch.cuda.is_available() else 'cpu', method="all", seed=1):

    if dataset_name in ["mnist", "fashion", "cifar10", "celeba"]:
        n_classes = 10
    elif dataset_name in ["dsprites"]:
        n_classes = 5

    if method == "all":
        method = ["tsne", "densumap", "pca"]
    if type(method) is str:
        method = [method] # For consistent iteration later

    class_samples = [[] for _ in range(n_classes)]
    post_means = [[] for _ in range(n_classes)]
    post_logvars = [[] for _ in range(n_classes)]
    post_samples = [[] for _ in range(n_classes)]

    # Data for training embeddings
    with torch.no_grad():
        model.eval()
        for step, (x,y) in enumerate(loader):
            post_mean, post_logvar = model.encoder(x.to(device))
            samples = model.reparameterize(post_mean, post_logvar)
            if step > steps:
                break
            for idx in range(len(y)):
                proper_slot = y[idx].item()
                class_samples[proper_slot].append(x[idx])
                post_means[proper_slot].append(post_mean[idx])
                post_logvars[proper_slot].append(post_logvar[idx])
                post_samples[proper_slot].append(samples[idx].cpu().numpy())

    if dataset_name in ["mnist", "fashion", "cifar10"]:
        post_samples_viz = post_samples
    elif dataset_name in ["dsprites"]:
        special_idxs = star_shape('dsprites')
        class_samples_viz = [[raw_dataset[j] for j in special_idxs[i]] for i in range(5)]
        post_means_viz = [[] for _ in range(n_classes)]
        post_logvars_viz = [[] for _ in range(n_classes)]
        post_samples_viz = [[] for _ in range(n_classes)]
        for i, latent_traversals in enumerate(class_samples_viz):
            for x in latent_traversals:
                post_mean, post_logvar = model.encoder(x.unsqueeze(dim=0).to(device))
                samples = model.reparameterize(post_mean, post_logvar)
                post_means_viz[i].append(post_mean)
                post_logvars_viz[i].append(post_logvar)
                post_samples_viz.append(samples[0].cpu().numpy())

    true_labels = [[x]*len(class_samples[x]) for x in range(len(class_samples))]
    plots = {}
    dim_reduction_models = {}
    for viz in tqdm(method, desc="Iterating over dim. reduction methods"):
        if viz == 'tsne':
            dim_reduction_model = manifold.TSNE(n_components=2, random_state=seed)
            dim_reduction_samples = dim_reduction_model.fit_transform(list(itertools.chain.from_iterable(post_samples_viz)))
        elif viz == "densumap":
            flat_samples = [np.array(single_class)
                for single_class in post_samples_viz] # UMAP doesnt support CHW data shape but it must be flat
            flat_samples = np.concatenate(flat_samples)
            dim_reduction_model = umap.UMAP(random_state=seed, densmap=True).fit(flat_samples)
            dim_reduction_samples = dim_reduction_model.embedding_

        elif viz == "pca":
            dim_reduction_model = decomposition.PCA(n_components=2, random_state=seed)
            dim_reduction_samples = dim_reduction_model.fit_transform(list(itertools.chain.from_iterable(post_samples_viz)))
        plot = graph_latent_samples(dim_reduction_samples, true_labels)
        dim_reduction_models[viz] = dim_reduction_model
        plots[viz] = plot

    model.train()

    all_data = {"class_samples":class_samples, "post_means":post_means, 
        "post_logvars":post_logvars, "post_samples":post_samples, 
        "labels":true_labels, "dim_reduction_samples":dim_reduction_samples}
    return plots, all_data, dim_reduction_models


def cluster_metric(post_samples, labels, n_clusters):
    labels = list(itertools.chain.from_iterable(labels))
    post_samples = list(itertools.chain.from_iterable(post_samples))
    kmeans = KMeans(n_clusters, random_state=1).fit(post_samples)
    cluster_assignments = kmeans.predict(post_samples)
    homogeneity = sklearn.metrics.homogeneity_score(labels, cluster_assignments)
    return homogeneity



