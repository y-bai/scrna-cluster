#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@Author: Yong Bai, yong.bai@hotmail.com
@Time: 2022/6/21 9:30
@License: (C) Copyright 2021-2025. 
@File: main.py
@Desc:

'''
import sys
import os
import argparse
import logging

import scanpy as sc

def main(in_args):
    
    data_file = in_args.data_file
    lr = in_args.lr
    alpha = in_args.alpha
    gamma = in_args.gamma
    nu = in_args.nu
    preprocess = in_args.preprocess
    normalization = in_args.normalization
    verbosity = in_args.verbosity
    evaluation = in_args.evaluation
    save_path = in_args.save_path
    
    if not os.path.isfile(data_file) or os.path.splitext(data_file)[1] != '.h5ad':
        print('please specify scRNA-seq data file (.h5ad), program exit ...')
        sys.exit(1)
    if evaluation and save_path is None:
        print('please specify result-saving path. program exit...')
        sys.exit(1)
    
    adata = sc.read_h5ad(data_file)
    
    # preprocessing data
    if preprocess:
        print('preprocessing data...')
        sc.pp.filter_cells(adata, min_genes=100)
        sc.pp.filter_genes(adata, min_cells=3)
        if normalization:
            sc.pp.normalize_total(adata,target_sum=1e4)
            sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2500)
        adata = adata[:, adata.var.highly_variable]
        sc.pp.scale(adata, max_value=10)
    
    # clustering
    lr, alpha, gamma, nu = 0.01, 0.01, 4, 0.01
    print('=====================================')
    print('lr:{}, alpha:{}, gamma:{}, nu:{}'.format(lr, alpha, gamma, nu))
    start = time.time()
    torch.cuda.empty_cache()
    scdualgn_model = scDualGN.scDualGN(adata, n_z=32,device='cuda:1', batch_size=1024*2, verbosity=True, lr=lr).pretrain(alpha=alpha, beta=1, gamma=gamma)
    scdualgn_model.cluster(n_cluster=11, eta=1, nu=nu, n_epochs=64)

    end = time.time()
    print('running time = {}'.format(end-start))
    
        
    print('hello')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='scDualGN model')
    
    args.add_argument('-d', '--data_file', type=str,  required=True,
                      help='.h5ad file containing scRNA-seq data')
    
    args.add_argument('-l', '--lr',
                      default=0.0001, type=float,
                      help='learning rate')
    
    args.add_argument('-a', '--alpha',
                      default=0.01, type=float,
                      help='weight factor for KL loss in daul-VAE net')
    
    args.add_argument('-g', '--gamma',
                      default=4, type=float,
                      help='weight factor for MSE_tilde loss in daul-VAE net')
    
    args.add_argument('-u', '--nu',
                      default=0.01, type=float,
                      help='eight factor for Center loss during clustering')
    
    args.add_argument('-p', '--preprocess',
                      default=True, type=str2bool,
                      help='scRNA-seq data process')
    
    args.add_argument('-n', '--normalization',
                      default=True, type=str2bool,
                      help='scRNA-seq data normlization')
    
    args.add_argument('-v', '--verbosity',
                      default=False, type=str2bool,
                      help='specify if printing information during running')
    
    args.add_argument('-e', '--evaluation',
                      default=False, type=str2bool,
                      help='specify if calculating evaluation metrics such as ARI and NMI')
    
    args.add_argument('-s', '--save_path', type=str, required=True,
                      help='results-saving path, results saved as .h5ad and/or evaluation metrics')
    
    in_args = args.parse_args()
    main(in_args)
    
    # python scdualgn_main.py -e True
