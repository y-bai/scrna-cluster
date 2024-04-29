#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam, SGD, AdamW 
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, StepLR

import scanpy as sc

from .nets import DualVAE, ScDualGNNet
from .loss import daulvae_loss, CenterLoss
from .utils import data_preprocess, target_distribution, EarlyStopping
from .dataset import SingleCellDataset
from .evalution import evals

import logging
logger = logging.getLogger(__name__)


class scDualGN():
    def __init__(self, adata, n_enc_1=512, n_enc_2=128, n_z=32, batch_size=1024,
                 device='cuda:0', verbosity=False, lr=0.001):
        """ initialize the parameters for scDualGN model

        Parameters
        ----------
        adata: anndata
            containing single cells (after preprocessing)

        n_enc_1: int, default = 512
            the number of neurons in the first encoder layer of model

        n_enc_2: int, default = 128
            the number of neurons in the second encoder layer

        n_z: int, default = 32
            the length of z vector in embedding space
        
        batch_size: int, default = 1024
            the size of batch for training the model

        device: string, default = 'cuda:0'
            can be 'cuda:0', 'cuda:1',...,or 'cpu'
            device to be used for running scDualGN

        verbosity: bool, default = False
            specify if print information during running

        """
        self.adata = adata
        self.input_dataset = SingleCellDataset(self.adata)
        self.n_cell = len(self.input_dataset)
        self.n_input = adata.X.shape[1]
        
        self.n_enc_1 = n_enc_1
        self.n_enc_2 = n_enc_2
        self.n_z = n_z
        
        # self.lr = 0.001 if self.n_cell < 1e5 else 0.0001 if self.n_cell < 6e5 else 0.00001 if self.n_cell < 1e6 else 0.000001
        # self.lr = 0.0001 if self.n_cell < 7e3 else 0.001 if self.n_cell < 3e4 else 0.001 #
        self.lr = lr # 0.01 if self.n_cell < 7e3 else 0.001 if self.n_cell < 3e4 else 0.001
        
        # print(self.lr)
        
        self.batch_size = batch_size
        
        self.device = torch.device(device)
        self.verbosity = verbosity
    
    def pretrain(self, alpha=0.05, beta=20, gamma=4):
        """
        pre-train the dual-VAE network in scDualGN model

        Parameters
        ----------
        alpha: float, default = 0.05
            weight factor for KL loss in daul-VAE net

        beta: float, default = 20
            weight factor for MSE_hat loss in daul-VAE net

        gamma: float, default = 4
            weight factor for MSE_tilde loss in daul-VAE net

        """

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        if self.verbosity:
            logger.info("dual-VAE pretrain start...")
        self.dual_vae = DualVAE(n_input=self.n_input,
                                n_enc_1=self.n_enc_1, 
                                n_enc_2=self.n_enc_2, 
                                n_z=self.n_z)
        self.dual_vae.to(self.device)
        
        batch_size =  512 if self.n_cell <= 1e4 else 1024 # self.batch_size 
        dloader = torch.utils.data.DataLoader(self.input_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
        
        # pre_lr = 0.001 if self.n_cell < 1e5 else 0.0001 if self.n_cell < 6e5 else 0.00001 if self.n_cell < 1e6 else 0.000001
        
        # optimizer = SGD(self.dual_vae.parameters(), lr=self.lr,  momentum=0.9, weight_decay=5e-4)
        optimizer = AdamW(self.dual_vae.parameters(), lr=self.lr, weight_decay=5e-4)
        
        # scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
        # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

        n_epochs =  100 if self.n_cell < 2e4 else 50
        # n_epochs =  50 if self.n_cell < 2e4 else 30
        self.dual_vae.train()
        for epoch in range(n_epochs):
            # the ith epoch loss
            i_loss = 0.
            i_mse_loss = 0
            i_mse1_loss = 0
            i_kl_loss = 0
            
            for batch_idx, (x, _) in enumerate(dloader):
                x = x.to(device=self.device)
                optimizer.zero_grad()
                x_bar, z, mu, log_var, x_bar1 = self.dual_vae(x)
                loss, mse_loss, mse1_loss, kl_loss = daulvae_loss(x_bar, x_bar1, x, mu, log_var,
                                                              alpha=self.alpha,
                                                              beta=self.beta,
                                                              gamma=self.gamma)
                i_loss += loss.item()
                i_mse_loss += mse_loss.item()
                i_mse1_loss += mse1_loss.item()
                i_kl_loss += kl_loss.item()
                
                loss.backward()
                # print(loss.grad)
                # torch.nn.utils.clip_grad_norm_(parameters=self.dual_vae.parameters(), max_norm=60, norm_type=2.0)
                
                optimizer.step()
                
            scheduler.step()
            
            if self.verbosity:
                i_epoch_loss = i_loss / (batch_idx + 1)
                i_epoch_mse_loss = i_mse_loss / (batch_idx + 1)
                i_epoch_mse1_loss = i_mse1_loss / (batch_idx + 1)
                i_epoch_kl_loss = i_kl_loss / (batch_idx + 1)
                
                logger.info("Epoch {}/{},Overall loss:{:.4f},MSE:{:.4f},MSE1:{:.4f},KL: {:.4f}".format(
                    epoch+1, n_epochs, i_epoch_loss, i_epoch_mse_loss, i_epoch_mse1_loss, i_epoch_kl_loss))
                
        if self.verbosity:    
            logger.info("dual-VAE pretrain finished")
        
        z_embeddings = []
        x_bars = []
        batch_size = min(self.n_cell, 1024 * 8)
        init_eval_loader = torch.utils.data.DataLoader(self.input_dataset,
                                                   batch_size=batch_size, shuffle=False)
        self.dual_vae.eval()
        for batch_idx, (x, _) in enumerate(init_eval_loader):
            x = x.to(device=self.device)
            x_bar, z, _, _, _ = self.dual_vae(x)
            z_embeddings.append(z.detach().cpu().numpy())
            x_bars.append(x_bar.detach().cpu().numpy())

        # make the list of 2d array as single 2d array
        self.init_z = np.concatenate(z_embeddings)
        if self.verbosity:
            logger.info("obtain daul-VAE z-vector and x_bar")
        
        return self
                
    def cluster(self, init_algorithm='kmeans', n_cluster=10,
                resolution=0.3, n_neighbors=20, eta=1, nu=1, n_epochs=30,
                evalution=True, eval_save_path=None):
        """
        clustering single cells

        the clustering by scDualGN first obtains pseudo-labels and centers
        using the specified initial algorithm.

        Parameters
        ----------
        init_algorithm: string, default='kmeans', lower case
            can be 'kmeans' or 'leiden'
            the algorithm to perform initial clustering

        n_cluster: int,
            the number of clusters, which is required
            when using kmeans algorithm

        resolution: float, default = 0.3
            resolution for 'leiden', which is required
            when using leiden algorithm

        n_neighbors: int, default = 20
            the number of neighbors for knn when performing 'leiden'

        eta: float, default = 1
            weight factor KL loss during clustering

        nu: float, default = 1
            eight factor for Center loss during clustering

        evalution: bool
            whether calculating ARI,NMI and ACC. adata['cell_type'] is required when evalution=True

        eval_save_path: string
            the .csv file path to save the evaluation results when evalution=True

        """

        self._init_clustering(init_algorithm=init_algorithm, n_cluster=n_cluster,
                              resolution=resolution, n_neighbors=n_neighbors)
        
        self.scdualgn = ScDualGNNet(self.n_cluster, n_input=self.n_input,
                                    n_enc_1=self.n_enc_1, n_enc_2=self.n_enc_2, n_z=self.n_z)

        self.scdualgn.dual_vae.load_state_dict(self.dual_vae.state_dict())
        self.scdualgn.to(self.device)
        self.scdualgn.cluster_layer.data = torch.tensor(self.init_cluster_centers).to(device=self.device)
        
        # center loss
        center_loss = CenterLoss(self.n_cluster, self.n_z, device=self.device)
        # weight_cent = 0.01
        
        # params = list(self.scdualgn.parameters()) + list(center_loss.parameters())
        # optimizer_model = SGD(params, lr=self.lr) # here lr is the overall learning rate
        
        optimizer_model = SGD(self.scdualgn.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4) # 0.9, 5e-4
        optimizer_centloss = SGD(center_loss.parameters(), lr=1e-6)
        
        # optimizer_model = AdamW(self.scdualgn.parameters(), lr=self.lr, weight_decay=5e-04)
        # optimizer_centloss = AdamW(center_loss.parameters(), lr=1e-6)
        
        # scheduler = CosineAnnealingLR(optimizer_model, T_max=10, eta_min=0)
        scheduler = CosineAnnealingWarmRestarts(optimizer_model, T_0=10, T_mult=2) # 10
        # scheduler = StepLR(optimizer_model, step_size=5, gamma=0.1)
        
        y_pred_last = self.init_cluster_labels
        
        if evalution:
            assert 'celltype' in self.adata.obs, f'adata does not have celltype attribute'
            adata_y = np.array(self.adata.obs['celltype'], dtype=int)  # true label
        
        c_batch_size = min(self.n_cell, 1024*2) if self.n_cell < 1e4 else self.batch_size 
        dataloader = torch.utils.data.DataLoader(self.input_dataset, batch_size=c_batch_size, shuffle=True)
        if self.verbosity:
            logger.info('perform clustering using scDualGN iteratively...')
        self.scdualgn.train()
        
        # n_pat = 20 if self.n_cell < 1e5 else 8
        # min_d = 10 if self.n_cell < 1e5 else 50
        # estop = EarlyStopping(patience=10, min_delta=10)
        
        last_epoch = False
        for epoch in range(n_epochs):
            
            if epoch + 1 == n_epochs:
                last_epoch = True
                
            if epoch % 3 == 0 or last_epoch:
                init_q = []
                ixs = []
                for eval_x, ix in dataloader:
                    eval_x = eval_x.to(device=self.device)
                    _, _, i_z, _, _, i_q = self.scdualgn(eval_x)
                    init_q.append(i_q.data.cpu().numpy())
                    ixs.append(ix.data.numpy())
                
                init_q = torch.from_numpy(np.concatenate(init_q))
                init_p = target_distribution(init_q)
            
                ixs = np.concatenate(ixs)
                # print(ixs, len(ixs))
                sorted_ixs = torch.from_numpy(np.argsort(ixs))
                # print(sorted_ixs, len(sorted_ixs))
                
                # original order by index
                init_q = init_q[sorted_ixs]
                init_p = init_p[sorted_ixs]
                
                y_pred_q = init_q.data.numpy().argmax(1)
                if evalution:
                    acc_q, nmi_q, air_q, hs_q, cs_q, purity = evals(adata_y, y_pred_q, self.verbosity)
                delta_label = np.sum(y_pred_q != y_pred_last).astype(np.float32) / y_pred_q.shape[0]
                
                # print(delta_label)
                # calinski_harabasz_score(y_pred_q)
                
                y_pred_last = y_pred_q
                
          
            t_overall_loss = 0
            t_daulvae_loss = 0
            t_kl_loss = 0
            t_centerloss = 0
            
            for i_batch, (batch_x, idx) in enumerate(dataloader):
                batch_x = batch_x.to(device=self.device)
                idx = idx.to(self.device)
                
                optimizer_model.zero_grad()
                optimizer_centloss.zero_grad()
                
                x_bar, x_bar1, z, mu, log_var, q = self.scdualgn(batch_x)
                
                # dual-VAE loass
                dvae_loss, _, _, _ = daulvae_loss(x_bar, x_bar1, batch_x, mu, log_var,
                                                  alpha=self.alpha, beta=self.beta,
                                                  gamma=self.gamma) 
                # clustering KL loss
                kl_loss = torch.nn.functional.kl_div(q.log(), init_p[idx].to(self.device))
                # center loss
                cl_loss = center_loss(z, torch.from_numpy(y_pred_last)[idx].to(self.device))
                cl_loss *= nu 
                overall_loss = dvae_loss + eta * kl_loss + cl_loss
                # overall_loss = dvae_loss + eta * kl_loss
                
                t_overall_loss += overall_loss.item() # standard python number
                t_daulvae_loss += dvae_loss.item()
                t_kl_loss += kl_loss.item()
                t_centerloss += cl_loss.item()
                
                overall_loss.backward()
                # torch.nn.utils.clip_grad_norm_(parameters=self.scdualgn.parameters(), max_norm=60, norm_type=2.0)
                
                # multiple (1./nu) in order to remove the effect of alpha on updating centers
                for param in center_loss.parameters():
                    param.grad.data *= (1. / nu)
                
                optimizer_model.step()
                optimizer_centloss.step()
                
            scheduler.step()
            
            epoch_overall_loss = t_overall_loss / (i_batch + 1)
            epoch_daulvae_loss = t_daulvae_loss / (i_batch + 1)
            epoch_kl_loss = t_kl_loss / (i_batch + 1)
            epoch_centerloss = t_centerloss / (i_batch + 1)
            
            # estop(epoch_overall_loss)
            # if estop.early_stop:
            #     logger.info('Reached tolerance threshold. Stopping training.')
            #     break
            
            if self.verbosity:
                logger.info('Epoch {}/{}, Loss - overall: {:.4f},daul_VAE:{:.4f},KL:{:.4f},Center:{:.4f}'.format(
                    epoch+1, n_epochs, epoch_overall_loss, 
                    epoch_daulvae_loss, epoch_kl_loss, epoch_centerloss))
        
        # final embedding
        eval_batch_size = min(self.n_cell, 1024 * 4)
        eval_data_loader = torch.utils.data.DataLoader(self.input_dataset, 
                                                       batch_size=eval_batch_size, shuffle=False)
        self.scdualgn.eval()
        q_final = []
        z_final = []
        for x_dataloader, _ in eval_data_loader:
            x_dataloader = x_dataloader.to(device=self.device)
            _, _, n_z, _, _, n_q = self.scdualgn(x_dataloader)
            q_final.append(n_q.data.cpu().numpy())
            z_final.append(n_z.data.cpu().numpy())

        q_final = torch.from_numpy(np.concatenate(q_final)).to(self.device)
        p_final = target_distribution(q_final)
        self.y_pred_label = q_final.data.cpu().numpy().argmax(1)
        # self.y_pred_p_final = p_final.data.cpu().numpy().argmax(1)
        self.z = np.concatenate(z_final)
        if self.verbosity:
            logger.info('clustering finished.')
        if evalution:
            q_final_acc, q_final_nmi, q_final_ari, q_final_hs, q_final_cs, q_final_purity = evals(adata_y, self.y_pred_label, self.verbosity)
            if eval_save_path is not None:
                res = {'ACC': q_final_acc, 'NMI': q_final_nmi, 'ARI': q_final_ari, 'HS': q_final_hs, 'CS': q_final_cs, 'PURITY': q_final_purity}
                pd.DataFrame.from_dict(res).to_csv(index=False)
        print('acc {:.4f}, nmi {:.4f}, ari {:.4f}, hs {:.4f}, cs {:.4f}, purity: {:.4f}'.format(q_final_acc, q_final_nmi, q_final_ari, q_final_hs, q_final_cs, q_final_purity))
        
    def batch_correct(self):
        pass
    
    def denoise(self):
        pass
    
    def save_model(self, path=None):
        torch.save(self.scdualgn.state_dict(), path)

    def _init_clustering(self, init_algorithm='kmeans', n_cluster=10,
                         resolution=0.3, n_neighbors=20):

        if init_algorithm == 'kmeans':
            self.n_cluster = n_cluster
            if self.verbosity:
                logger.info('perform kmeans to initially cluster....')
            
            if self.n_cell < 4e4:

                from sklearn.cluster import KMeans
                kmeans_c = KMeans(n_clusters=self.n_cluster, random_state=42, n_init=20).fit(self.init_z)
                self.init_cluster_labels = kmeans_c.labels_
                self.init_cluster_centers = kmeans_c.cluster_centers_
            
            else:
                import faiss
                
                d = self.init_z.shape[1]
                
                kmeans = faiss.Clustering(d, self.n_cluster)
                kmeans.niter = 300
                kmeans.seed = 42
                kmeans.nredo = 20
                
                # otherwise the kmeans implementation sub-samples the training set
                # kmeans.max_points_per_centroid = 1e7
                
                if self.device.type == 'cuda':
                    res = faiss.StandardGpuResources()
                    flat_config = faiss.GpuIndexFlatConfig()
                    flat_config.device = int(torch.cuda.current_device())
                    index = faiss.GpuIndexFlatL2(res, d, flat_config)
                else:
                    index = faiss.IndexFlatL2(d)
                
                kmeans.train(self.init_z, index)
                centroids = faiss.vector_float_to_array(kmeans.centroids)
                self.init_cluster_centers = centroids.reshape(self.n_cluster, d)
                
                dists, pred_labels = index.search(self.init_z, 1)
                self.init_cluster_labels = pred_labels.ravel()

        if init_algorithm == 'leiden':
            if self.verbosity:
                logger.info('perform leiden to initially cluster....')

            assert self.adata is not None, f'adata is None'
            self.adata.obsm['X_pre_z'] = self.init_z
            sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, use_rep="X_pre_z")
            sc.tl.leiden(self.adata, resolution=resolution)
            cluster_center = []
            uni_label_lst = np.unique(self.adata.obs['leiden']).tolist()
            for i in uni_label_lst:
                cluster_center.append(
                    np.mean(self.adata.obsm['X_pre_z'][self.adata.obs['leiden'] == i], axis=0))
            self.init_cluster_centers = np.array(cluster_center)
            self.init_cluster_labels = np.array(self.adata.obs['leiden']).astype('int32')
            self.n_cluster = len(uni_label_lst)
            
        
        if init_algorithm == 'gmm':
            self.n_cluster = n_cluster
            if self.verbosity:
                logger.info('perform gmm to initially cluster....')

            # from sklearnex import patch_sklearn
            # patch_sklearn()

            from sklearn.mixture import GaussianMixture             
            gmm_c = GaussianMixture(n_components=self.n_cluster, random_state=42, warm_start=True).fit(self.init_z) #
            self.init_cluster_labels = gmm_c.predict(self.init_z)
            self.init_cluster_centers = gmm_c.means_
            
            # from sklearnex import unpatch_sklearn
            # unpatch_sklearn()
        
        if init_algorithm == 'birch':
            self.n_cluster = n_cluster
            if self.verbosity:
                logger.info('perform birch to initially cluster....')

            # from sklearnex import patch_sklearn
            # patch_sklearn()
            
            from sklearn.cluster import Birch, KMeans

            birch_c = Birch(threshold=0.5, n_clusters=KMeans(n_clusters=self.n_cluster, n_init=20, random_state=42)).fit(self.init_z)
            self.init_cluster_labels = birch_c.predict(self.init_z)
            
            cluster_center = []
            uni_label_lst = np.unique(self.init_cluster_labels).tolist()
            for i in uni_label_lst:
                cluster_center.append(
                    np.mean(self.init_z[self.init_cluster_labels == i], axis=0))
            self.init_cluster_centers = np.array(cluster_center)
            
            # from sklearnex import unpatch_sklearn
            # unpatch_sklearn()
        if self.verbosity:
            logger.info('initial n cluster: {}'.format(len(np.unique(self.init_cluster_labels))))
            
        