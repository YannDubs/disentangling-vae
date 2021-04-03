import os
import logging
import math
from functools import reduce
from collections import defaultdict
import json
from timeit import default_timer
import sys
from numpy.random import RandomState
import time

from tqdm import trange, tqdm
import numpy as np
import torch
from torch import pca_lowrank
import openTSNE
import umap

from disvae.models.losses import get_loss_f
from disvae.utils.math import log_density_gaussian
from disvae.utils.modelIO import save_metadata
from disvae.models.linear_model import weight_reset
from disvae.models.linear_model import Classifier
from utils.fid import get_fid_value

from sklearn import decomposition
from sklearn import manifold
from sklearn import linear_model
import wandb

TEST_LOSSES_FILE = "test_losses.log"
METRICS_FILENAME = "metrics.log"
METRIC_HELPERS_FILE = "metric_helpers.pth"


class Evaluator:
    """
    Class to handle training of model.

    Parameters
    ----------
    model: disvae.vae.VAE

    loss_f: disvae.models.BaseLoss
        Loss function.

    device: torch.device, optional
        Device on which to run the code.

    logger: logging.Logger, optional
        Logger.

    save_dir : str, optional
        Directory for saving logs.

    is_progress_bar: bool, optional
        Whether to use a progress bar for training.
    """

    def __init__(self, model, loss_f,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 logger=logging.getLogger(__name__),
                 save_dir="results",
                 is_progress_bar=True, 
                 use_wandb=True, 
                 higgins_drop_slow=True,
                 use_NN_classifier =False,
                 no_shape_classifier = True,
                 seed=None,
                 dset_name=None):

        self.device = device
        self.loss_f = loss_f
        self.model = model.to(self.device)
        self.logger = logger
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger.info("Testing Device: {}".format(self.device))
        self.use_wandb=use_wandb
        self.seed = seed
        self.higgins_drop_slow = higgins_drop_slow
        self.dset_name=dset_name
        self.use_NN_classifier = use_NN_classifier       #flag if sklearn or manual classifier
        self.no_shape_classifier = no_shape_classifier
    def __call__(self, data_loader, is_metrics=False, is_losses=True):
        """Compute all test losses.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        is_metrics: bool, optional
            Whether to compute and store the disentangling metrics.

        is_losses: bool, optional
            Whether to compute and store the test losses.
        """
        start = default_timer()
        is_still_training = self.model.training
        self.model.eval()

        metrics, losses = None, None
        if is_metrics:
            self.logger.info('Computing metrics...')
            metrics = self.compute_metrics(data_loader, dataset=self.dset_name)
            self.logger.info('Metrics: {}'.format(metrics))
            # save_metadata(metrics, self.save_dir, filename=METRICS_FILENAME)

        if is_losses:
            self.logger.info('Computing losses...')
            losses = self.compute_losses(data_loader)
            self.logger.info('Losses: {}'.format(losses))
            # save_metadata(losses, self.save_dir, filename=TEST_LOSSES_FILE)

        if is_still_training:
            self.model.train()

        self.logger.info('Finished evaluating after {:.1f} min.'.format((default_timer() - start) / 60))

        return metrics, losses

    def compute_losses(self, dataloader):
        """Compute all test losses.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        """
        self.model.eval()
        storer = defaultdict(list)
        for data, _ in tqdm(dataloader, leave=False, disable=not self.is_progress_bar):
            data = data.to(self.device)

            try:
                recon_batch, latent_dist, latent_sample = self.model(data)
                _ = self.loss_f(data, recon_batch, latent_dist, self.model.training,
                                storer, latent_sample=latent_sample)
            except ValueError:
                # for losses that use multiple optimizers (e.g. Factor)
                _ = self.loss_f.call_optimize(data, self.model, None, storer)

        losses = {k: sum(v) / len(dataloader) for k, v in storer.items()}
        self.model.train()
        return losses

    def compute_metrics(self, dataloader, dataset=None):
        """Compute all the metrics.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        """
        # if self.use_wandb:
        #      wandb.config["latent_size"] = self.model.latent_dim
        #      wandb.config["classifier_hidden_size"] = 512
        #      wandb.config["sample_size"] = 300   
        
        self.model.eval()
        accuracies, fid, mig, aam = None, None, None, None # Default values. Not all metrics can be computed for all datasets

        # Need to create a new small dataset for FID. The default dataloaders we get in would typically be shuffled as well, so we need to remove that
        if dataset != '3dshapes' and dataset != 'mpi3dtoy': #takes way too long for these too currently even for smaller sets
            total_len = 0
            max_len = 50000
            small_dset_x = []
            small_dset_y = []
            for x,y in dataloader:
                small_dset_x.append(x)
                small_dset_y.append(y)
                total_len += len(x)
                if total_len > max_len:
                    break

            fid = get_fid_value(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.cat(small_dset_x), torch.cat(small_dset_y)), batch_size=dataloader.batch_size, shuffle=False), self.model)
        
        if dataset in ['dsprites', 'mpi3dtoy', '3dshapes']: # Most metrics are only applicable for datasets with ground truth variation factors
            try:
                lat_sizes = dataloader.dataset.lat_sizes
                lat_names = dataloader.dataset.lat_names
                lat_imgs = dataloader.dataset.imgs
            except AttributeError:
                raise ValueError("Dataset needs to have known true factors of variations to compute the metric. This does not seem to be the case for {}".format(type(dataloader.__dict__["dataset"]).__name__))
            
            self.logger.info("Computing the disentanglement metric")
            method_names = ["VAE", "PCA", "ICA", "T-SNE","UMAP", "DensUMAP"]
            accuracies = self._disentanglement_metric(dataloader.dataset, method_names, sample_size=300, n_epochs=20000, dataset_size=7500, hidden_dim=512, use_non_linear=False)
            #sample size is key for VAE, for sample size 50 only 88% accuarcy, compared to 95 for 200 sample sze
            #non_linear_accuracies = self._disentanglement_metric(["VAE", "PCA", "ICA"], 50, lat_sizes, lat_imgs, n_epochs=150, dataset_size=5000, hidden_dim=128, use_non_linear=True) #if hidden dim too large -> no training possible
            if self.use_wandb:
                # wandb.log(accuracies)
                wandb.save("disentanglement_metrics.h5")

            self.logger.info("Computing the empirical distribution q(z|x).")
            samples_zCx, params_zCx = self._compute_q_zCx(dataloader)
            len_dataset, latent_dim = samples_zCx.shape

            self.logger.info("Estimating the marginal entropy.")
            # marginal entropy H(z_j)
            H_z = self._estimate_latent_entropies(samples_zCx, params_zCx)

            # conditional entropy H(z|v)
            samples_zCx = samples_zCx.view(*lat_sizes, latent_dim)
            params_zCx = tuple(p.view(*lat_sizes, latent_dim) for p in params_zCx)
            H_zCv = self._estimate_H_zCv(samples_zCx, params_zCx, lat_sizes, lat_names)

            H_z = H_z.cpu()
            H_zCv = H_zCv.cpu()

            # I[z_j;v_k] = E[log \sum_x q(z_j|x)p(x|v_k)] + H[z_j] = - H[z_j|v_k] + H[z_j]
            mut_info = - H_zCv + H_z
            sorted_mut_info = torch.sort(mut_info, dim=1, descending=True)[0].clamp(min=0)

            metric_helpers = {'marginal_entropies': H_z, 'cond_entropies': H_zCv}
            mig = self._mutual_information_gap(sorted_mut_info, lat_sizes, storer=metric_helpers).item()
            aam = self._axis_aligned_metric(sorted_mut_info, storer=metric_helpers).item()
            torch.save(metric_helpers, os.path.join(self.save_dir, METRIC_HELPERS_FILE))

            

        metrics = {'DM': accuracies, 'MIG': mig, 'AAM': aam, 'FID': fid}
        print(f"Evaluated metrics for {dataset} as: {metrics}")
        self.model.train()
        return metrics


    def _disentanglement_metric(self, dataset, method_names, sample_size, n_epochs=6000, dataset_size = 1000, hidden_dim = 256, use_non_linear = False):

        #train models for all concerned methods and stor them in a dict
        methods = {}
        runtimes = {}
        for method_name in tqdm(method_names, desc="Iterating over methods for the Higgins disentanglement metric"):
            if method_name == "VAE":
                methods["VAE"] = self.model

            elif method_name == "PCA":   
                start = time.time() 
                self.logger.info("Training PCA...")
                pca = decomposition.PCA(n_components=self.model.latent_dim, whiten = True)
                if dataset.imgs.ndim == 4:
                    data_imgs = dataset.imgs[:,:,:,0]
                    imgs_pca = np.reshape(data_imgs, (data_imgs.shape[0], data_imgs.shape[1]**2))
                else: 
                    imgs_pca = np.reshape(dataset.imgs, (dataset.imgs.shape[0], dataset.imgs.shape[1]**2))
                size = min(25000, len(imgs_pca))

                idx = np.random.randint(len(imgs_pca), size = size)
                imgs_pca = imgs_pca[idx, :]       #not enough memory for full dataset -> repeat with random subsets               
                pca.fit(imgs_pca)
                methods["PCA"] = pca
                self.logger.info("Done")
                # if self.use_wandb:
                #     wandb.config["PCA_training_size"] = size
                runtimes[method_name] = time.time()-start
                    

            elif method_name == "ICA":
                start = time.time() 
                self.logger.info("Training ICA...")
                ica = decomposition.FastICA(n_components=self.model.latent_dim, max_iter=400)
                if dataset.imgs.ndim == 4:
                    data_imgs = dataset.imgs[:,:,:,0]
                    imgs_ica = np.reshape(data_imgs, (data_imgs.shape[0], data_imgs.shape[1]**2))
                else:
                    imgs_ica = np.reshape(dataset.imgs, (dataset.imgs.shape[0], dataset.imgs.shape[1]**2))
                size = min(1500, len(imgs_ica))
                # if self.use_wandb:
                #     wandb.config["ICA_training_size"] = size
                idx = np.random.randint(len(imgs_ica), size = size)
                imgs_ica = imgs_ica[idx, :]       #not enough memory for full dataset -> repeat with random subsets 
                ica.fit(imgs_ica)
                methods["ICA"] = ica
                self.logger.info("Done")
                runtimes[method_name] = time.time()-start

            elif method_name == "T-SNE":
                continue
                # start = time.time() 
                # self.logger.info("Training T-SNE...")
                # tsne = manifold.TSNE(n_components=self.model.latent_dim, method='exact')
                # # imgs_tsne = np.reshape(imgs, (imgs.shape[0], imgs.shape[1]**2))
                # # size = min(5000, len(imgs_tsne))
                # # idx = np.random.randint(len(imgs_tsne), size = size)
                # # imgs_tsne = imgs_tsne[idx, :]       #not enough memory for full dataset -> repeat with random subsets 
                # # tsne = tsne.fit(imgs_tsne)
                # methods["T-SNE"] = tsne
                # self.logger.info("Done")
                # runtimes[method_name] = time.time()-start

            elif method_name == "UMAP":
                if self.higgins_drop_slow:
                    continue
                else:
                    start = time.time() 
                    import umap
                    self.logger.info("Training UMAP...")
                    umap_model = umap.UMAP(random_state=self.seed, densmap=False, n_components=self.model.latent_dim)
                    imgs_umap = np.reshape(imgs, (imgs.shape[0], imgs.shape[1]**2))
                    size = min(25000, len(imgs_umap))
                    idx = np.random.randint(len(imgs_umap), size = size)
                    imgs_umap = imgs_umap[idx, :]       #not enough memory for full dataset -> repeat with random subsets 
                    umap_model.fit(imgs_umap)
                    methods["UMAP"] = umap_model
                    self.logger.info("Done")
                    runtimes[method_name] = time.time()-start

            elif method_name == "DensUMAP":
                continue
                # start = time.time() 
                # self.logger.info("Training UMAP...")
                # umap_model = umap.UMAP(random_state=self.seed, densmap=True, n_components=self.model.latent_dim)
                # # imgs_umap = np.reshape(imgs, (imgs.shape[0], imgs.shape[1]**2))
                # # size = min(25000, len(imgs_umap))
                # # idx = np.random.randint(len(imgs_umap), size = size)
                # # imgs_umap = imgs_umap[idx, :]       #not enough memory for full dataset -> repeat with random subsets 
                # # umap_model.fit(imgs_umap)
                # methods["DensUMAP"] = umap_model
                # self.logger.info("Done")
                # runtimes[method_name] = time.time()-start

            else: 
                raise ValueError("Unknown method : {}".format(method_name))

        if self.use_wandb:
            try:
                wandb.log(runtimes)
            except:
                pass
        #compute training- and test data for linear classifier     
        
        #latent dim = length of z_b_diff for arbitrary method = output dimension of linear classifier
        latent_dim = 10
        data_train, data_test = {}, {}

        for method in methods:
            data_train[method] = [], []
            data_test[method] = [], []

        for i in tqdm(range(dataset_size), desc="Generating datasets for Higgins metric"):
            data = self._compute_z_b_diff_y(methods, sample_size, dataset)
            for method in methods:
                data_train[method][0].append(data[method][0])
                data_train[method][1].append(data[method][1])
            if i <= int(dataset_size*0.2):
                
                data = self._compute_z_b_diff_y(methods, sample_size, dataset)
                for method in methods:
                    data_test[method][0].append(data[method][0])
                    data_test[method][1].append(data[method][1])

        test_acc = {"logreg":{},"linear":{}, "nonlinear":{}}
        for model_class in ["linear", "nonlinear", "logreg"]:
            if model_class in ["linear", "nonlinear"]:
                model = Classifier(latent_dim,hidden_dim,len(dataset.lat_sizes), use_non_linear= True if model_class =="nonlinear" else False)
                
                model.to(self.device)
                model.train()

                #log softmax with NLL loss 
                criterion = torch.nn.NLLLoss()
                optim = torch.optim.Adam(model.parameters(), lr=0.01 if model_class =="linear" else 0.001, weight_decay=0 if model_class == "linear" else 1e-4)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=5000, min_lr=0.00001)

                for method in tqdm(methods.keys(), desc = "Training classifiers for the Higgins metric"):
                    if method == "ICA":
                        optim = torch.optim.Adam(model.parameters(), lr=1 if model_class =="linear" else 0.001, weight_decay=0 if model_class == "linear" else 1e-4)
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=5000, min_lr=0.00001)

                    print(f'Training the classifier for model {method}')
                    for e in tqdm(range(n_epochs if model_class == "linear" else round(n_epochs/2)), desc="Iterating over epochs while training the Higgins classifier"):
                        model.train()
                        optim.zero_grad()
                        
                        X_train, Y_train = data_train[method]
                        X_train, Y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.long)
                        X_train = X_train.to(self.device)
                        Y_train = Y_train.to(self.device)
                        
                        X_test , Y_test = data_test[method]
                        X_test, Y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.long)
                        X_test = X_test.to(self.device)
                        Y_test = Y_test.to(self.device)

                        scores_train = model(X_train)   
                        loss = criterion(scores_train, Y_train)
                        loss.backward()
                        optim.step()
                        scheduler.step(loss)
                        
                        if (e+1) % 2000 == 0:
                            model.eval()
                            with torch.no_grad():
                                scores_test = model(X_test)   
                                test_loss = criterion(scores_test, Y_test)
                                tqdm.write(f'In this epoch {e+1}/{n_epochs}, Training loss: {loss.item():.4f}, Test loss: {test_loss.item():.4f}')
                                model.eval()
                                scores_train = model(X_train)
                                scores_test = model(X_test)
                                _, prediction_train = scores_train.max(1)
                                _, prediction_test = scores_test.max(1)

                                train_acc = (prediction_train==Y_train).sum().float()/len(X_train)
                                test_acc[model_class][method] = (prediction_test==Y_test).sum().float()/len(X_test)
                                tqdm.write(f'Accuracy of {method} on training set: {train_acc.item():.4f}, test set: {test_acc[model_class][method].item():.4f}')
                            model.train()
                    
                    model.eval()
                    with torch.no_grad():
                        
                        scores_train = model(X_train)
                        scores_test = model(X_test)
                        _, prediction_train = scores_train.max(1)
                        _, prediction_test = scores_test.max(1)

                        train_acc = (prediction_train==Y_train).sum().float()/len(X_train)
                        test_acc[model_class][method] = (prediction_test==Y_test).sum().float()/len(X_test)
                        print(f'Accuracy of {method} on training set: {train_acc.item():.4f}, test set: {test_acc[model_class][method].item():.4f}')
                        
                    model.apply(weight_reset)
                
            elif model_class in ["logreg"]:
                
                # data_train, data_test = {}, {}

                # for method in methods:
                #     data_train[method] = [], []
                #     data_test[method] = [], []

                # for i in tqdm(range(dataset_size), desc="Generating datasets for Higgins metric"):
                #     data = self._compute_z_b_diff_y(methods, sample_size, dataset)
                #     for method in methods:
                #         data_train[method][0].append(data[method][0])
                #         data_train[method][1].append(data[method][1])
                #     if i <= int(dataset_size*0.2):
                        
                #         data = self._compute_z_b_diff_y(methods, sample_size, dataset)
                #         for method in methods:
                #             data_test[method][0].append(data[method][0])
                #             data_test[method][1].append(data[method][1])

                for method in tqdm(methods.keys(), desc = "Training classifiers for the Higgins metric"):
                    classifier = linear_model.LogisticRegression(max_iter=500)
                    X_train, Y_train = data_train[method]
                    X_test, Y_test = data_test[method]

                    #X_train, Y_train = X_train.cpu().detach().numpy(), Y_train.cpu().detach().numpy()
                    #X_test, Y_test = X_test.cpu().detach().numpy(), Y_test.cpu().detach().numpy()

                    classifier.fit(X_train, Y_train)
                    train_acc = np.mean(classifier.predict(X_train)==Y_train)
                    test_acc[model_class][method] = np.mean(classifier.predict(X_test)==Y_test)
                    print(f'Accuracy of {method} on training set: {train_acc:.4f}, test set: {test_acc[model_class][method].item():.4f}')

        return test_acc

    def _compute_z_b_diff_y(self, methods, sample_size, dataset):
        
        #if dsprites:
        if self.dset_name == "dsprites" and self.no_shape_classifier:
            y = np.random.randint(1, dataset.lat_sizes.size, size=1)
        else:
            y = np.random.randint(dataset.lat_sizes.size, size=1)
        
        
        #y = np.random.randint(dataset.lat_sizes.size, size=1)
        y_lat = np.random.randint(dataset.lat_sizes[y], size=sample_size)

        # optional helper function to show images
        def show_images_grid(imgs_, num_images=25):
            ncols = int(np.ceil(num_images**0.5))
            nrows = int(np.ceil(num_images / ncols))
            _, axes = plt.pyplot.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
            axes = axes.flatten()

            for ax_i, ax in enumerate(axes):
                if ax_i < num_images:
                    ax.imshow(imgs_[ax_i][0], cmap='Greys_r',  interpolation='nearest')
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.axis('off')
            plt.pyplot.show()
        
        imgs_sampled1  = dataset.images_from_data_gen(sample_size, y, y_lat)
        imgs_sampled2  = dataset.images_from_data_gen(sample_size, y, y_lat)

        res = {}
        #calculate the expectation values of the normal distributions in the latent representation for the given images
        for method in methods.keys():
            if method == "VAE":
                with torch.no_grad():
                    mu1, _ = self.model.encoder(imgs_sampled1.to(self.device))
                    mu2, _ = self.model.encoder(imgs_sampled2.to(self.device))  
                    if not self.use_NN_classifier:
                        mu1 = mu1.cpu().detach().numpy()
                        mu2 = mu2.cpu().detach().numpy()  
            elif method == "PCA":
                pca = methods[method]
                #flatten images
                if dataset.imgs.ndim == 4:
                    imgs_sampled_pca1 = torch.reshape(imgs_sampled1[:,0,:,:], (imgs_sampled1.shape[0], imgs_sampled1.shape[2]**2))
                    imgs_sampled_pca2 = torch.reshape(imgs_sampled2[:,0,:,:], (imgs_sampled2.shape[0], imgs_sampled2.shape[2]**2))
                else:
                    imgs_sampled_pca1 = torch.reshape(imgs_sampled1, (imgs_sampled1.shape[0], imgs_sampled1.shape[2]**2))
                    imgs_sampled_pca2 = torch.reshape(imgs_sampled2, (imgs_sampled2.shape[0], imgs_sampled2.shape[2]**2))

                if not self.use_NN_classifier:
                    mu1 = pca.transform(imgs_sampled_pca1)
                    mu2 = pca.transform(imgs_sampled_pca2)
                else:
                    mu1 = torch.from_numpy(pca.transform(imgs_sampled_pca1)).float()
                    mu2 = torch.from_numpy(pca.transform(imgs_sampled_pca2)).float()

            elif method == "ICA":
                ica = methods[method]
                #flatten images
                if dataset.imgs.ndim == 4:
                    imgs_sampled_ica1 = torch.reshape(imgs_sampled1[:,0,:,:], (imgs_sampled1.shape[0], imgs_sampled1.shape[2]**2))
                    imgs_sampled_ica2 = torch.reshape(imgs_sampled2[:,0,:,:], (imgs_sampled2.shape[0], imgs_sampled2.shape[2]**2))
                else:   
                   imgs_sampled_ica1 = torch.reshape(imgs_sampled1, (imgs_sampled1.shape[0], imgs_sampled1.shape[2]**2))
                   imgs_sampled_ica2 = torch.reshape(imgs_sampled2, (imgs_sampled2.shape[0], imgs_sampled2.shape[2]**2))
                

                if not self.use_NN_classifier:
                    mu1 = ica.transform(imgs_sampled_ica1)
                    mu2 = ica.transform(imgs_sampled_ica2)
                else:
                    mu1 = torch.from_numpy(ica.transform(imgs_sampled_ica1)).float()
                    mu2 = torch.from_numpy(ica.transform(imgs_sampled_ica2)).float()

            elif method == "T-SNE":
                continue
                # tsne = methods[method]
                
                # #flatten images
                # imgs_sampled_tsne1 = torch.reshape(imgs_sampled1, (imgs_sampled1.shape[0], imgs_sampled1.shape[2]**2))
                # imgs_sampled_tsne2 = torch.reshape(imgs_sampled2, (imgs_sampled2.shape[0], imgs_sampled2.shape[2]**2))
                
                # mu1 = torch.from_numpy(tsne.fit_transform(imgs_sampled_tsne1)).float()
                # mu2 = torch.from_numpy(tsne.fit_transform(imgs_sampled_tsne2)).float()
            elif method == "UMAP":
                if self.higgins_drop_slow:
                    continue
                else:
                    umap = methods[method]
                    #flatten images
                    imgs_sampled1 = imgs_sampled1[0:100]
                    imgs_sampled2 = imgs_sampled2[0:100]
                    if dataset.imgs.ndim == 4:
                        imgs_sampled_umap1 = torch.reshape(imgs_sampled1[:,0,:,:], (imgs_sampled1.shape[0], imgs_sampled1.shape[2]**2))
                        imgs_sampled_umap2 = torch.reshape(imgs_sampled2[:,0,:,:], (imgs_sampled2.shape[0], imgs_sampled2.shape[2]**2))
                    else:   
                        imgs_sampled_umap1 = torch.reshape(imgs_sampled1, (imgs_sampled1.shape[0], imgs_sampled1.shape[2]**2))
                        imgs_sampled_umap2 = torch.reshape(imgs_sampled2, (imgs_sampled2.shape[0], imgs_sampled2.shape[2]**2))
                    if not self.use_NN_classifier:
                        mu1 = umap.transform(imgs_sampled_umap1).float()
                        mu2 = umap.transform(imgs_sampled_umap2).float()
                    else:
                        mu1 = torch.from_numpy(umap.transform(imgs_sampled_umap1)).float()
                        mu2 = torch.from_numpy(umap.transform(imgs_sampled_umap2)).float()
            elif method == "DensUMAP":
                continue
                # densumap = methods[method]
                # #flatten images
                # imgs_sampled_densumap1 = torch.reshape(imgs_sampled1, (imgs_sampled1.shape[0], imgs_sampled1.shape[2]**2))
                # imgs_sampled_densumap2 = torch.reshape(imgs_sampled2, (imgs_sampled2.shape[0], imgs_sampled2.shape[2]**2))
                
                # mu1 = torch.from_numpy(densumap.fit_transform(imgs_sampled_densumap1)).float()
                # mu2 = torch.from_numpy(densumap.fit_transform(imgs_sampled_densumap2)).float()
                
            else: 
                raise ValueError("Unknown method : {}".format(method)) 

            if not self.use_NN_classifier:
                res[method] = np.mean(np.abs(mu1 - mu2), axis=0), y[0]
            else:    
                z_diff = torch.abs(torch.sub(mu1, mu2))
                z_diff_b = torch.mean(z_diff, 0)
                res[method] = z_diff_b, torch.from_numpy(y)

        return res

    def _mutual_information_gap(self, sorted_mut_info, lat_sizes, storer=None):
        """Compute the mutual information gap as in [1].

        References
        ----------
           [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
           autoencoders." Advances in Neural Information Processing Systems. 2018.
        """
        # difference between the largest and second largest mutual info
        delta_mut_info = sorted_mut_info[:, 0] - sorted_mut_info[:, 1]
        # NOTE: currently only works if balanced dataset for every factor of variation
        # then H(v_k) = - |V_k|/|V_k| log(1/|V_k|) = log(|V_k|)
        H_v = torch.from_numpy(lat_sizes).float().log()
        mig_k = delta_mut_info / H_v
        mig = mig_k.mean()  # mean over factor of variations

        if storer is not None:
            storer["mig_k"] = mig_k
            storer["mig"] = mig

        return mig

    def _axis_aligned_metric(self, sorted_mut_info, storer=None):
        """Compute the proposed axis aligned metrics."""
        numerator = (sorted_mut_info[:, 0] - sorted_mut_info[:, 1:].sum(dim=1)).clamp(min=0)
        aam_k = numerator / sorted_mut_info[:, 0]
        aam_k[torch.isnan(aam_k)] = 0
        aam = aam_k.mean()  # mean over factor of variations

        if storer is not None:
            storer["aam_k"] = aam_k
            storer["aam"] = aam

        return aam

    def _compute_q_zCx(self, dataloader):
        """Compute the empiricall disitribution of q(z|x).

        Parameter
        ---------
        dataloader: torch.utils.data.DataLoader
            Batch data iterator.

        Return
        ------
        samples_zCx: torch.tensor
            Tensor of shape (len_dataset, latent_dim) containing a sample of
            q(z|x) for every x in the dataset.

        params_zCX: tuple of torch.Tensor
            Sufficient statistics q(z|x) for each training example. E.g. for
            gaussian (mean, log_var) each of shape : (len_dataset, latent_dim).
        """
        len_dataset = len(dataloader.dataset)
        latent_dim = self.model.latent_dim
        n_suff_stat = 2

        q_zCx = torch.zeros(len_dataset, latent_dim, n_suff_stat, device=self.device)

        n = 0
        with torch.no_grad():
            for x, label in dataloader:
                batch_size = x.size(0)
                idcs = slice(n, n + batch_size)
                q_zCx[idcs, :, 0], q_zCx[idcs, :, 1] = self.model.encoder(x.to(self.device))
                n += batch_size

        params_zCX = q_zCx.unbind(-1)
        samples_zCx = self.model.reparameterize(*params_zCX)

        return samples_zCx, params_zCX

    def _estimate_latent_entropies(self, samples_zCx, params_zCX,
                                   n_samples=10000):
        r"""Estimate :math:`H(z_j) = E_{q(z_j)} [-log q(z_j)] = E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]`
        using the emperical distribution of :math:`p(x)`.

        Note
        ----
        - the expectation over the emperical distributio is: :math:`q(z) = 1/N sum_{n=1}^N q(z|x_n)`.
        - we assume that q(z|x) is factorial i.e. :math:`q(z|x) = \prod_j q(z_j|x)`.
        - computes numerically stable NLL: :math:`- log q(z) = log N - logsumexp_n=1^N log q(z|x_n)`.

        Parameters
        ----------
        samples_zCx: torch.tensor
            Tensor of shape (len_dataset, latent_dim) containing a sample of
            q(z|x) for every x in the dataset.

        params_zCX: tuple of torch.Tensor
            Sufficient statistics q(z|x) for each training example. E.g. for
            gaussian (mean, log_var) each of shape : (len_dataset, latent_dim).

        n_samples: int, optional
            Number of samples to use to estimate the entropies.

        Return
        ------
        H_z: torch.Tensor
            Tensor of shape (latent_dim) containing the marginal entropies H(z_j)
        """
        len_dataset, latent_dim = samples_zCx.shape
        device = samples_zCx.device
        H_z = torch.zeros(latent_dim, device=device)

        # sample from p(x)
        samples_x = torch.randperm(len_dataset, device=device)[:n_samples]
        # sample from p(z|x)
        samples_zCx = samples_zCx.index_select(0, samples_x).view(latent_dim, n_samples)

        mini_batch_size = 10
        samples_zCx = samples_zCx.expand(len_dataset, latent_dim, n_samples)
        mean = params_zCX[0].unsqueeze(-1).expand(len_dataset, latent_dim, n_samples)
        log_var = params_zCX[1].unsqueeze(-1).expand(len_dataset, latent_dim, n_samples)
        log_N = math.log(len_dataset)
        with trange(n_samples, leave=False, disable=self.is_progress_bar) as t:
            for k in range(0, n_samples, mini_batch_size):
                # log q(z_j|x) for n_samples
                idcs = slice(k, k + mini_batch_size)
                log_q_zCx = log_density_gaussian(samples_zCx[..., idcs],
                                                 mean[..., idcs],
                                                 log_var[..., idcs])
                # numerically stable log q(z_j) for n_samples:
                # log q(z_j) = -log N + logsumexp_{n=1}^N log q(z_j|x_n)
                # As we don't know q(z) we appoximate it with the monte carlo
                # expectation of q(z_j|x_n) over x. => fix a single z and look at
                # proba for every x to generate it. n_samples is not used here !
                log_q_z = -log_N + torch.logsumexp(log_q_zCx, dim=0, keepdim=False)
                # H(z_j) = E_{z_j}[- log q(z_j)]
                # mean over n_samples (i.e. dimesnion 1 because already summed over 0).
                H_z += (-log_q_z).sum(1)

                t.update(mini_batch_size)

        H_z /= n_samples

        return H_z

    def _estimate_H_zCv(self, samples_zCx, params_zCx, lat_sizes, lat_names):
        """Estimate conditional entropies :math:`H[z|v]`."""
        latent_dim = samples_zCx.size(-1)
        len_dataset = reduce((lambda x, y: x * y), lat_sizes)
        H_zCv = torch.zeros(len(lat_sizes), latent_dim, device=self.device)
        for i_fac_var, (lat_size, lat_name) in enumerate(zip(lat_sizes, lat_names)):
            idcs = [slice(None)] * len(lat_sizes)
            for i in range(lat_size):
                self.logger.info("Estimating conditional entropies for the {}th value of {}.".format(i, lat_name))
                idcs[i_fac_var] = i
                # samples from q(z,x|v)
                samples_zxCv = samples_zCx[idcs].contiguous().view(len_dataset // lat_size,
                                                                   latent_dim)
                params_zxCv = tuple(p[idcs].contiguous().view(len_dataset // lat_size, latent_dim)
                                    for p in params_zCx)

                H_zCv[i_fac_var] += self._estimate_latent_entropies(samples_zxCv, params_zxCv
                                                                    ) / lat_size
        return H_zCv
