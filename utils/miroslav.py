import wandb
import os

def wandb_auth(fname: str = "nas_key.txt"):
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
  wandb.login()

def latent_viz(model, loader, dataset, steps=100, device='cpu'):
    from sklearn import manifold
    from tqdm import tqdm
    import torch
    import itertools
    import matplotlib.pyplot as plt

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
                post_means[proper_slot].append(post_mean[0, idx])
                post_logvars[proper_slot].append(post_logvar[0, idx])
                post_samples[proper_slot].append(samples[idx].numpy())

    true_labels = [[x]*len(class_samples[x]) for x in range(len(class_samples))]
    tsne = manifold.TSNE(n_components=2, random_state=1)
    samples_tsne = tsne.fit_transform(list(itertools.chain.from_iterable(post_samples)))
    plt.scatter(samples_tsne[:,0],samples_tsne[:,1],
        c=list(itertools.chain.from_iterable(true_labels)),
        cmap=plt.cm.get_cmap('jet', 10))
    plt.colorbar()
    plt.show()
    model.train()

    all_data = {"class_samples":class_samples, "post_means":post_means, 
        "post_logvars":post_logvars, "post_samples":post_samples}
    return plt, all_data
