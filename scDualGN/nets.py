#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter

import logging
logger = logging.getLogger(__name__)


class DualVAE(torch.nn.Module):
    """
    dual-VAE net without clustering part

    Parameters
    ----------
    n_input: the number of HVG (i.e, n_feature of input), recommended: 2500
    n_enc_1: the number of neurons in the first encoder layer, recommended: 512
    n_enc_2: the number of neurons in the second encoder layer, recommended: 128
    n_z: the length of z vector in embedding space, recommended 32
    """

    def __init__(self, n_input=2500,
                 n_enc_1=512,
                 n_enc_2=128,
                 n_z=32):
        super(DualVAE, self).__init__()

        n_dec_1, n_dec_2 = n_enc_2, n_enc_1
        
        leaky_rate = 0.1

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_input, n_enc_1),
            torch.nn.BatchNorm1d(n_enc_1),
            #torch.nn.Dropout(0.4),
            torch.nn.LeakyReLU(leaky_rate),
            # torch.nn.ReLU(),
            
            torch.nn.Linear(n_enc_1, n_enc_2),
            torch.nn.BatchNorm1d(n_enc_2),
            # torch.nn.Dropout(0.4),
            torch.nn.LeakyReLU(leaky_rate))
            # torch.nn.ReLU())

        self.n_z = n_z
        
        self.z_layer_mu = torch.nn.Linear(n_enc_2, n_z)
        self.z_layer_logvar = torch.nn.Linear(n_enc_2, n_z)
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_z, n_dec_1),
            torch.nn.BatchNorm1d(n_dec_1),
            #torch.nn.Dropout(0.4),
            torch.nn.LeakyReLU(leaky_rate),
            # torch.nn.ReLU(),
            
            torch.nn.Linear(n_dec_1, n_dec_2),
            torch.nn.BatchNorm1d(n_dec_2),
            #torch.nn.Dropout(0.4),
            torch.nn.LeakyReLU(leaky_rate))
            # torch.nn.ReLU())
        
        self.x_bar_layer = torch.nn.Linear(n_dec_2, n_input)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)
            # n = module.in_features
            # y = 1.0/np.sqrt(n)
            # module.weight.data.uniform_(-y, y)
            # module.bias.data.fill_(0)
            
        if isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight.data, 1)
            nn.init.constant_(module.bias.data, 0)

    def _sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample
    
    def forward(self, x):

        h_e = self.encoder(x)
        mu = self.z_layer_mu(h_e)
        log_var = self.z_layer_logvar(h_e)
        
        # auxiluary decoder from mu
        h_d1 = self.decoder(mu)
        x_bar1 = self.x_bar_layer(h_d1)
        
        z = self._sampling(mu, log_var)

        # primary decoder, share the parameter
        # decoder from z: N(mu, sigma)
        h_d = self.decoder(z)
        x_bar = self.x_bar_layer(h_d)
        
        return x_bar, z, mu, log_var, x_bar1


class ScDualGNNet(torch.nn.Module):
    def __init__(self, n_clusters, n_input=2500, n_enc_1=512, n_enc_2=128, n_z=32):
        """

        Parameters
        ----------
        n_input: the number of HVG (i.e, n_feature of input)
        n_enc_1: the number of neurons in the first encoder layer
        n_enc_2: the number of neurons in the second encoder layer
        n_z: the length of z vector in embedding space
        n_clusters: the number of clusters
        """

        super(ScDualGNNet, self).__init__()

        self.dual_vae = DualVAE(
            n_input=n_input,
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_z=n_z)

        # degree to calculate q
        self.v = 1
        self.n_z = n_z

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, self.n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
                
    def forward(self, x):
        # deep embedded cluster
        # Dual Net Module
        x_bar, z, mu, log_var, x_bar1 = self.dual_vae(x)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, x_bar1, z, mu, log_var, q

