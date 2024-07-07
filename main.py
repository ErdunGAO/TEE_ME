import torch
import os

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.vae_framework import VF_VCNet

from data.data import load_simu_data
from utils.eval import curve
from utils.args import get_argparser
from utils.setup import setup
import logging
import numpy as np
from trainer.train import train_VF_VCNet


def main():

    # load the args and setup the environment
    args = get_argparser()
    setup(args)
    logging.info('Finished setup')

    # Data preparation
    #NOTE This is only for the single run case
    logging.info('==> Preparing data..')
    load_path = os.path.join(args.data_dir,
                                str(args.t_noise_dist)+'_'+ str(args.me_dist),
                                str(args.me_std),
                                str(args.seed))
    
    save_path = args.work_dir

    
    train_loader, test_matrix, t_grid = load_simu_data(load_path,
                                                       args.num_obs,
                                                       args.batch_size)


    x_dim = 6
    # Model preparation
    logging.info('==> Building model..')
    if args.model == 'vf_vcnet':
        model = VF_VCNet(x_dim=x_dim, n_components=args.n_components)
        model._initialize_weights()
    else:
        raise NotImplementedError

    # Training
    logging.info('==> Training model..')
    if args.model == 'vf_vcnet':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        model.train()
        for epoch in range(args.n_epochs):
            train_VF_VCNet(model,
                        train_loader,
                        optimizer,
                        me_dist=args.me_dist,
                        me_std=args.me_std,
                        y_coeff=args.y_coeff,
                        alpha=args.alpha,
                        beta=args.beta,
                        epoch=epoch,
                        device=None)

    else:
        raise NotImplementedError

    # Evaluation
    logging.info('==> Evaluating model..')
    if args.model in ['vf_vcnet']:
        _, mse, pe = curve(model, test_matrix, t_grid, args.dataset_name)
    else:
        raise NotImplementedError

    logging.info(f"mse: {mse}")
    np.save(os.path.join(save_path, 'mse.npy'), mse)

    logging.info(f"pe: {pe}")
    np.save(os.path.join(save_path, 'pe.npy'), pe)


if __name__ == '__main__':
  main()