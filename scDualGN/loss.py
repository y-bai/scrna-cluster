#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes, feat_dim, device=None):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + torch.pow(
            self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


# dual network loss
def daulvae_loss(x_hat_pri, x_tilde_aux, x, mu, log_var,
                 alpha=0.01, beta=20, gamma=2):
    """
    calculate loss for dual network without clustering.

    Parameters
    ----------
    x_hat_pri: reconstructed X from primary decoder (from z vector)
    x_tilde_aux: reconstructed X from auxiliary decoder (from mu vector)
    x: original x
    mu: mu vectore
    log_var: log-transformed variance
    alpha: weight factor for KL loss, alpha*2
    beta: default = 20, weight factor for MSE loss between x and x_hat_pri
    gamma: weight factor for MSE between x_tilde_aux and x_hat_pri

    Returns
    -------
    d_loss: overall loss of dual network
    mse_hat_pri: mean loss between x and x_hat_pri
    mse_tilde_aux: mean loss between x_tilde_aux and x_hat_pri
    kl_vae: mean kl loss
    """

    mse_hat_pri = torch.sum(torch.pow((x - x_hat_pri), 2), 1)
    # mse_hat_pri = F.mse_loss(x, x_hat_pri)
    mse_tilde_aux = torch.sum(torch.pow((x_tilde_aux - x_hat_pri), 2), 1)
    # mse_tilde_aux = F.mse_loss(x_tilde_aux, x_hat_pri)
    kl_vae = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), 1)

    d_loss = torch.mean(alpha * kl_vae + beta * mse_hat_pri + gamma * mse_tilde_aux)

    return d_loss, torch.mean(mse_hat_pri), torch.mean(mse_tilde_aux), torch.mean(kl_vae)
