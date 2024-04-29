#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from torch.optim import Adam
from torch.nn.parameter import Parameter

import scanpy as sc

from .nets import DualVAE, ScDualGNNet
from .loss import daulvae_loss, CenterLoss
from .utils import data_preprocess, target_distribution
from .dataset import SingleCellDataset
from .evalution import evals

import logging
logger = logging.getLogger(__name__)


def run_scDualGN(adata, n_enc_1=512, n_enc_2=128, n_z=32, batch_size=1024*8, save_path=None,
                n_epochs=30, n_epoch_update_pq=1, cluster_alg='kmeans', n_cluster=10, resolution=0.3, 
                n_neighbors=20, alpha_dualvae=0.01, beta_daulvae=20, gamma_dualvae=2, eta_klloss=1, 
                nu_centerloss=1, device='cuda:0'):
    """
    main function to cluster using scDualGN net

    Parameters
    ----------
    adata: raw data after process,
    n_enc_1: default=512,
        the number of neurons in the first encoder layer,
    n_enc_2: default=128,
        the number of neurons in the second encoder layer,
    n_z: default=32,
        the length of z vector in embedding space,
    batch_size: default=1024*8,
        batch_size used to train whole scDaulGN net,
    save_path: default=None,
        path to save the trained scDualGN,
    n_epochs: default=30,
        the number of epochs to train scDaulGN
    n_epoch_update_pq: default=1
        the number of epoch-interval to update q and p for calculating KL loss.
    cluster_alg: 'kmeans' or 'leiden'
        the algorithm to perform initial clusterings,
    n_cluster: required when cluster_alg=='kmeans',
        the number of clusters
    resolution: required when cluster_alg=='leiden'
        resolution for 'leiden'
    n_neighbors: default=20,
        number of neighbors for knn when performing 'leiden',
    alpha_dualvae:recommended value: 0.005*20=0.01 for any-sized dataset,
        weight factor for KL loss in daul-VAE net,
    beta_daulvae: recommended value: 20,
        weight factor for MSE_hat loss in daul-VAE net,
    gamma_dualvae: recommended value: 2 for any-sized datasets
        weight factor for MSE_tilde loss in daul-VAE net,
    eta_klloss: recommended value: 1 for any-sized datasets, 
        weight factor KL loss for clustering,
    nu_centerloss: recommended value: 1, other value has negative effect.
        weight factor for Center loss, 
    device:  value: 'cuda:0',...or 'cuda:4', or 'cpu'
        specify the device
    Returns
    -------
    z_final: final z vectore in embedding space, (n_cells, n_z), numpy
    y_pred_q_final: final predicted labels, (n_cells,), numpy
    model: trained scDualGN
    """

    # # preprocess adata
    # adata = data_preprocess(adata)
    # build dataset
    input_dset = SingleCellDataset(adata)

    # determine n_epochs and lr according to the number of cells
    n_cell = len(input_dset)
    lr = 0.001 if n_cell < 1.1e5 else 0.0001 if n_cell < 6e5 else 0.00001 if n_cell < 1e6 else 0.000001

    device = torch.device(device)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('device: {}'.format(device))

    # pretrain
    pre_train_n_epochs =  100 if n_cell < 1e4 else 50
    dual_vae_model = DualVAE(n_input=adata.X.shape[1],
                     n_enc_1=n_enc_1, n_enc_2=n_enc_2, n_z=n_z)
    dual_vae_model.to(device)
    init_z_embeddings, init_x_bars, _, _, _, _ = pretrain_dualvae(
        dual_vae_model, input_dset, device=device, n_epochs=pre_train_n_epochs, lr=lr,
        alpha_dualvae=alpha_dualvae, beta_daulvae=beta_daulvae, gamma_dualvae=gamma_dualvae)
    n_clusters, init_cluster_labels, init_cluster_centers = init_cluster(
        init_z_embeddings, cluster_alg=cluster_alg, adata=adata,
        n_cluster=n_cluster, resolution=resolution, n_neighbors=n_neighbors)

    # scDualGN scDualGN
    model = ScDualGNNet(n_clusters, n_input=adata.X.shape[1],
                     n_enc_1=n_enc_1, n_enc_2=n_enc_2, n_z=n_z)
    model.dual_vae.load_state_dict(dual_vae_model.state_dict())
    model.to(device)
    # initialize clustering results
    model.cluster_layer.data = torch.tensor(init_cluster_centers).to(device=device)

    # center loss
    center_loss = CenterLoss(n_cluster, n_z, device=device)
    weight_cent = 0.001

    optimizer_model = Adam(model.parameters(), lr=lr)
    optimizer_centloss = Adam(center_loss.parameters(), lr=1e-6)
    
    y_pred_last = init_cluster_labels
    switcher = False
    n_patience = 0
    
    assert 'celltype' in adata.obs
    adata_y = np.array(adata.obs['celltype'], dtype=int)  # true label

    dset = SingleCellDataset(adata)
    # dset.set_y(init_cluster_labels)

    train_batch_size = min(n_cell, batch_size)
    train_data_loader = torch.utils.data.DataLoader(dset, batch_size=train_batch_size, shuffle=True)

    logger.info('training scDualGN...')
    model.train()
    for epoch in range(n_epochs):

        if epoch % n_epoch_update_pq == 0:
            # model.eval()
            # update p and q
            # with torch.no_grad():
            init_q = []
            ixs = []
            for eval_x, ix in train_data_loader:
                eval_x = eval_x.to(device=device)
                _, _, i_z, _, _, i_q = model(eval_x)
                init_q.append(i_q.data.cpu().numpy())
                ixs.append(ix.data.numpy())
                    
            init_q = torch.from_numpy(np.concatenate(init_q))
            init_p = target_distribution(init_q)
            
            ixs = np.concatenate(ixs)
            sorted_ixs = torch.from_numpy(np.argsort(ixs))
            # original order by index
            init_q = init_q[sorted_ixs]
            init_p = init_p[sorted_ixs]

            y_pred_q = init_q.data.numpy().argmax(1)
            
            acc_q, nmi_q, air_q = evals(adata_y, y_pred_q)
            delta_label = np.sum(y_pred_q != y_pred_last).astype(np.float32) / y_pred_q.shape[0]
            y_pred_last = y_pred_q
            
            logger.info('delta_label: {:.4f}'.format(delta_label))
            if epoch > 0 and delta_label < 0.001:
                logger.info('n_patience: {}'.format(n_patience))
                if switcher:
                    n_patience += 1
                else:
                    n_patience = 0
                switcher = True
                if n_patience == 5:  #
                    logger.info('Reached tolerance threshold. Stopping training.')
                    logger.info("final_acc: {}, final_nmi: {}, final_ari: {}".format(acc_q, nmi_q, air_q))
                    break
            else:
                switcher = False

        t_overall_loss = 0
        t_daulvae_loss = 0
        t_kl_loss = 0
        t_centerloss = 0

        for i_batch, (batch_x, idx) in enumerate(train_data_loader):
            batch_x = batch_x.to(device=device)
            idx = idx.to(device)

            optimizer_model.zero_grad()
            optimizer_centloss.zero_grad()

            x_bar, x_bar1, z, mu, log_var, q = model(batch_x)

            # dual-VAE loass
            dvae_loss, _, _, _ = daulvae_loss(x_bar, x_bar1, batch_x, mu, log_var,
                                              alpha=alpha_dualvae, beta=beta_daulvae,
                                              gamma=gamma_dualvae)
            # clustering KL loss
            kl_loss = torch.nn.functional.kl_div(q.log(), init_p[idx].to(device))
            # center loss
            cl_loss = center_loss(z, torch.from_numpy(y_pred_last)[idx].to(device))
            overall_loss = dvae_loss + eta_klloss * kl_loss + nu_centerloss * cl_loss

            t_overall_loss += overall_loss.item()
            t_daulvae_loss += dvae_loss.item()
            t_kl_loss += kl_loss.item()
            t_centerloss += cl_loss.item()

            overall_loss.backward()
            optimizer_model.step()

            for param in center_loss.parameters():
                param.grad.data *= (1. / weight_cent)
            optimizer_centloss.step()

        epoch_overall_loss = t_overall_loss / (i_batch + 1)
        epoch_daulvae_loss = t_daulvae_loss / (i_batch + 1)
        epoch_kl_loss = t_kl_loss / (i_batch + 1)
        epoch_centerloss = t_centerloss / (i_batch + 1)

        logger.info('Epoch {}/{}, Loss - overall: {:.4f},daul_VAE:{:.4f},KL:{:.4f},Center:{:.4f}'.format(
            epoch+1, n_epochs, epoch_overall_loss, epoch_daulvae_loss, epoch_kl_loss, epoch_centerloss
        ))
    if save_path is not None:
        # save whole scDualGN
        torch.save(model.state_dict(), save_path)

    # predict
    eval_batch_size = min(n_cell, 1024 * 4)
    eval_data_loader = torch.utils.data.DataLoader(dset, batch_size=eval_batch_size, shuffle=False)
    model.eval()
    q_final = []
    z_final = []
    for x_dataloader,_ in eval_data_loader:
        x_dataloader = x_dataloader.to(device=device)

        _, _, n_z, _, _, n_q = model(x_dataloader)
        q_final.append(n_q.data.cpu().numpy())
        z_final.append(n_z.data.cpu().numpy())

    q_final = torch.from_numpy(np.concatenate(q_final)).to(device)
    p_final = target_distribution(q_final)
    y_pred_q_final = q_final.data.cpu().numpy().argmax(1)
    y_pred_p_final = p_final.data.cpu().numpy().argmax(1)
    z_final = np.concatenate(z_final)

    logger.info('scDualGN train finished.')

    q_final_acc, q_final_nmi, q_final_ari = evals(adata_y, y_pred_q_final)

    return z_final, y_pred_q_final, model


def init_cluster(z_embeddings, cluster_alg='kmeans', adata=None, n_cluster=10, resolution=0.3, n_neighbors=20):
    '''
    initially clustering
    Parameters
    ----------
    z_embeddings: initial z vector
    cluster_alg, clustering algorithm, could be 'kmeans' and 'leiden'
    adata: original adata, required when cluster_alg == 'leiden'
    n_cluster: number of clusters,required when  cluster_alg == 'kmeans'
    resolution: resolution for 'leiden' algorithm

    Returns
    -------
    _n_cluster: number of clusters
    cluster_label: clustering label for each cell, (n_cell,)
    cluster_centers: vectors of cluster centers, (_n_cluster, n_z)
    '''

    if cluster_alg == 'kmeans':
        logger.info('perform kmeans to cluster....')
        # from sklearnex import patch_sklearn
        # patch_sklearn()
        
        from sklearn.cluster import KMeans
        kmeans_c = KMeans(n_clusters=n_cluster, n_init=20, random_state=666).fit(z_embeddings)
        cluster_label = kmeans_c.labels_
        cluster_centers = kmeans_c.cluster_centers_
        _n_cluster = n_cluster
        
        # from sklearnex import unpatch_sklearn
        # unpatch_sklearn()

    if cluster_alg == 'leiden':
        logger.info('perform leiden to cluster....')
        assert adata is not None, f'adata is None'
        adata.obsm['X_embeded_pre_z'] = z_embeddings
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_embeded_pre_z")
        sc.tl.leiden(adata, resolution=resolution)
        cluster_center = []
        uni_label_lst = np.unique(adata.obs['leiden']).tolist()
        for i in uni_label_lst:
            cluster_center.append(np.mean(adata.obsm['X_embeded_pre_z'][adata.obs['leiden'] == i], axis=0))
        cluster_centers = np.array(cluster_center)
        cluster_label = np.array(adata.obs['leiden']).astype('int32')
        _n_cluster = len(uni_label_lst)

    return _n_cluster, cluster_label, cluster_centers


def pretrain_dualvae(dualvae_model, input_dataset, device, n_epochs=100, lr=0.001,
                     alpha_dualvae=0.1, beta_daulvae=20, gamma_dualvae=2):
    """
    pretrain daul-vae to get z-vector in embedding space

    Parameters
    ----------
    dualvae_model
    input_dataset: pytorch dataset
    n_epochs: number of epochs when training scDualGN.
        <10k: n_epochs = 100
        otherwise: n_epochs = 50
    lr: learning rate, will be changed depending on size of dataset.
        <10k cells dataset, lr = 0.001
        >10k and < 600k cells, lr = 0.0001
        >600k and < 1000k cells: lr = 0.00001
        >1000k  cells: lr = 0.000001
    alpha_dualvae: weight factor for KL loss in daul-VAE net,
        recommended values: 0.005*20 for any-sized datasets
    beta_daulvae: weight factor for MSE_hat loss in daul-VAE net,
        recommended values: 20
    gamma_dualvae: weight factor for MSE_tilde loss in daul-VAE net,
        recommended values: 2 for any-sized datasets

    Returns
    -------
    z_embeddings:       z-vectors in embedding space, (n_cells, n_z), numpy
    x_bars:             reconstructed X from z-vector (i.e., primary decoder), (n_cells, n_hvg), numpy
    t_epoch_loss:       overall loss for pretrained dual-VAE net, list with length of n_epochs
    t_epoch_mse_loss:   loss for MSE loss from pretrained dual-VAE net, list with length of n_epochs
    t_epoch_mse1_loss:  loss for MSE1 loss from pretrained dual-VAE net, list with length of n_epochs
    t_epoch_kl_loss:    loss for KL loss from pretrained dual-VAE net, list with length of n_epochs
    """

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('pretrain device: {}'.format(device))

    logger.info("start pretraining...")
    len_data = len(input_dataset)
    #batch_size_pretain = min(len_data, 512)
    batch_size_pretain = 512 if len_data <= 1e4 else 2048
    
    pretrain_loader = torch.utils.data.DataLoader(input_dataset,
                                                  batch_size=batch_size_pretain,
                                                  shuffle=True)
    optimizer = Adam(dualvae_model.parameters(), lr=lr, weight_decay=0.001)
    dualvae_model.train()

    t_epoch_loss = []
    t_epoch_mse_loss = []
    t_epoch_mse1_loss = []
    t_epoch_kl_loss = []
    for epoch in range(n_epochs):
        # the ith epoch loss
        i_loss = 0.
        i_mse_loss = 0
        i_mse1_loss = 0
        i_kl_loss = 0

        for batch_idx, (x, _) in enumerate(pretrain_loader):
            x = x.to(device=device)

            optimizer.zero_grad()
            x_bar, z, mu, log_var, x_bar1 = dualvae_model(x)

            loss, mse_loss, mse1_loss, kl_loss = daulvae_loss(x_bar, x_bar1, x, mu, log_var,
                                                              alpha=alpha_dualvae,
                                                              beta=beta_daulvae,
                                                              gamma=gamma_dualvae)
            i_loss += loss.item()
            i_mse_loss += mse_loss.item()
            i_mse1_loss += mse1_loss.item()
            i_kl_loss += kl_loss.item()

            loss.backward()
            optimizer.step()

        i_epoch_loss = i_loss / (batch_idx + 1)
        i_epoch_mse_loss = i_mse_loss / (batch_idx + 1)
        i_epoch_mse1_loss = i_mse1_loss / (batch_idx + 1)
        i_epoch_kl_loss = i_kl_loss / (batch_idx + 1)

        logger.info("Epoch {}/{},Overall loss:{:.4f},MSE:{:.4f},MSE1:{:.4f},KL: {:.4f}".format(
            epoch, n_epochs, i_epoch_loss, i_epoch_mse_loss, i_epoch_mse1_loss, i_epoch_kl_loss))

        t_epoch_loss.append(i_epoch_loss)
        t_epoch_mse_loss.append(i_epoch_mse_loss)
        t_epoch_mse1_loss.append(i_epoch_mse1_loss)
        t_epoch_kl_loss.append(i_epoch_kl_loss)

    logger.info("dual-VAE pretrain finished!")
    # embedding building and initially clustering
    z_embeddings = []
    x_bars = []

    batch_size_eval = min(len_data, 1024 * 8)
    init_eval_loader = torch.utils.data.DataLoader(input_dataset,
                                                   batch_size=batch_size_eval, shuffle=False)
    dualvae_model.eval()
    for batch_idx, (x, _) in enumerate(init_eval_loader):
        x = x.to(device=device)
        x_bar, z, _, _, _ = dualvae_model(x)
        z_embeddings.append(z.detach().cpu().numpy())
        x_bars.append(x_bar.detach().cpu().numpy())

    # make the list of 2d array as single 2d array
    z_embeddings = np.concatenate(z_embeddings)
    x_bars = np.concatenate(x_bars)
    logger.info("obtain daul-VAE z-vector and x_bar")
    return (z_embeddings, x_bars,
            t_epoch_loss, t_epoch_mse_loss, t_epoch_mse1_loss, t_epoch_kl_loss)

