import wandb
import os
import itertools
from sklearn.cluster import KMeans
from sklearn import manifold
import sklearn.metrics
from tqdm import tqdm
import torch
import itertools
import matplotlib.pyplot as plt

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

def latent_viz(model, loader, dataset, steps=100, device='cpu'):

    if dataset in ["mnist", "fashion", "cifar10"]:
        n_classes = 10

    class_samples = [[] for _ in range(n_classes)]
    post_means = [[] for _ in range(n_classes)]
    post_logvars = [[] for _ in range(n_classes)]
    post_samples = [[] for _ in range(n_classes)]

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

    true_labels = [[x]*len(class_samples[x]) for x in range(len(class_samples))]
    dim_reduction_model = manifold.TSNE(n_components=2, random_state=1)
    dim_reduction_samples = dim_reduction_model.fit_transform(list(itertools.chain.from_iterable(post_samples)))
    plot = graph_latent_samples(dim_reduction_samples, true_labels)

    model.train()

    all_data = {"class_samples":class_samples, "post_means":post_means, 
        "post_logvars":post_logvars, "post_samples":post_samples, 
        "labels":true_labels, "dim_reduction_samples":dim_reduction_samples}
    return plot, all_data, dim_reduction_model


def cluster_metric(post_samples, labels, n_clusters):
    labels = list(itertools.chain.from_iterable(labels))
    post_samples = list(itertools.chain.from_iterable(post_samples))
    kmeans = KMeans(n_clusters, random_state=1).fit(post_samples)
    cluster_assignments = kmeans.predict(post_samples)
    homogeneity = sklearn.metrics.homogeneity_score(labels, cluster_assignments)
    return homogeneity



