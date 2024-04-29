import torch
import scanpy as sc

import logging
logger = logging.getLogger(__name__)

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def add_noise(inputs, device):
    inputs = inputs.to(device=device)
    noise = torch.FloatTensor(inputs.shape)
    noise = noise.to(device=device)
    torch.randn(inputs.shape, out=noise)
    return ((inputs + noise) * 0.5).to(device=device)


def data_preprocess(adata, min_genes=200, min_cells=3,
                    target_sum=1e4, n_top_genes=2000, max_value=10, log_norm=True):
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    if log_norm:
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=max_value)
    return adata

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0, verb=True):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verb = verb
        
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            # logger.info('early stop counter: {}/{}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True