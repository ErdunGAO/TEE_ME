import numpy as np
import json
import pandas as pd
import torch
import os

from tqdm import tqdm
import argparse
import pickle

from torch.distributions import Laplace

import random





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate ihdp data')
    # parser.add_argument('--data_path', type=str, default='dataset/ihdp_add/ihdp.csv', help='data path')
    parser.add_argument('--save_dir', type=str, default='dataset/ihdp_add/gene_data', help='dir to save generated data')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--t_noise_dist', type=str, default='normal', help='type of the noise for t') # NOTE: not used currently
    parser.add_argument('--me_dist', type=str, default='normal', help='distribution of measurement error')
    parser.add_argument('--me_std', type=float, default=0.2, help='std of the measurement error')
    # parser.add_argument('--num_eval', type=int, default=10, help='num of dataset for evaluating the methods')
    # parser.add_argument('--num_tune', type=int, default=2, help='num of dataset for tuning the parameters')

    args = parser.parse_args()

    save_path = args.save_dir

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # path = args.data_path
    # ihdp = pd.read_csv(path)
    # ihdp = ihdp.to_numpy()
    # ihdp = ihdp[:, 2:27]  # delete the first column (data idx)/ delete the second coloum (treatment)
    # ihdp = torch.from_numpy(ihdp)
    # ihdp = ihdp.float()

    ihdp = np.random.rand(4000, 25)
    ihdp = torch.from_numpy(ihdp)
    ihdp = ihdp.float()

    n_feature = ihdp.shape[1]
    n_data = ihdp.shape[0]

    # 0 1 2 4 5 -> continuous

    # normalize the data
    for _ in range(n_feature):
        minval = min(ihdp[:, _]) * 1.
        maxval = max(ihdp[:, _]) * 1.
        ihdp[:, _] = (1. * (ihdp[:, _] - minval))/maxval

    # cate_idx = torch.tensor([3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
    cate_idx1 = torch.tensor([3,6,7,8,9,10,11,12,13,14])
    cate_idx2 = torch.tensor([15,16,17,18,19,20,21,22,23,24])

    alpha = 5.
    cate_mean1 = torch.mean(ihdp[:, cate_idx1], dim=1).mean()
    cate_mean2 = torch.mean(ihdp[:, cate_idx1], dim=1).mean()
    tem = torch.tanh((torch.sum(ihdp[:, cate_idx2], dim=1)/10. - cate_mean2) * alpha)

    def x_t(x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[4]
        x5 = x[5]
        t = x1/(1. + x2) + max(x3, x4, x5)/(0.2 + min(x3, x4, x5)) + torch.tanh((torch.sum(x[cate_idx2])/10. - cate_mean2) * alpha) - 2.

        return t

    def x_t_link(t):
        return 1. / (1. + torch.exp(-2. * t))

    def t_x_y(t, x):
        # only x1, x3, x4 are useful
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[4]
        x5 = x[5]

        # v1
        factor1 = 0.5
        factor2 = 1.5

        # v2
        factor1 = 1.5
        factor2 = 0.5

        # original
        # factor1 = 1.
        # factor2 = 1.

        # y = 1. / (1.2 - t) * torch.sin(t * 3. * 3.14159) * (
        #             factor1 * torch.tanh((torch.sum(x[cate_idx1]) / 10. - cate_mean1) * alpha) +
        #             factor2 * torch.exp(0.2 * (x1 - x5)) / (0.1 + min(x2, x3, x4)))

        y = 1. / (2 - t) * torch.sin(t * 1.5 * 3.14159) * (
                    factor1 * torch.tanh((torch.sum(x[cate_idx1]) / 10. - cate_mean1) * alpha) +
                    factor2 * torch.exp(0.2 * (x1 - x5)) / (0.1 + min(x2, x3, x4)))
        
        return y

    # def ihdp_matrix():
    #     data_matrix = torch.zeros(n_data, n_feature+2)

    #     # get data matrix
    #     for _ in range(n_data):
    #         x = ihdp[_, :]
    #         t = x_t(x)
    #         t += torch.randn(1)[0] * 0.5
    #         t = x_t_link(t)
    #         y = t_x_y(t, x)
    #         y += torch.randn(1)[0] * 0.5

    #         data_matrix[_, 0] = t
    #         data_matrix[_, n_feature+1] = y
    #         data_matrix[_, 1: n_feature+1] = x

    #     # get t_grid
    #     t_grid = torch.zeros(2, n_data)
    #     t_grid[0, :] = data_matrix[:, 0].squeeze()

    #     for i in tqdm(range(n_data)):
    #         psi = 0
    #         t = t_grid[0, i]
    #         for j in range(n_data):
    #             x = data_matrix[j, 1: n_feature+1]
    #             psi += t_x_y(t, x)
    #         psi /= n_data
    #         t_grid[1, i] = psi

    #     return data_matrix, t_grid
    
    def new_ihdp_matrix(t_noise_dist='normal', me_dist='normal', me_std=None):

        covariates_matrix = torch.zeros(n_data, n_feature)
        t_clean_matrix = torch.zeros(n_data, 1)
        t_matrix = torch.zeros(n_data, 1)
        t_me_matrix = torch.zeros(n_data, 1)
        y_matrix = torch.zeros(n_data, 1)

        # get data matrix
        for _ in range(n_data):

            x = ihdp[_, :]
            covariates_matrix[_, :] = x

            t = x_t(x)
            t_clean = x_t_link(t)

            # add noise, with std 0.5
            if t_noise_dist == 'normal':
                t = t_clean + torch.randn(1)[0] * 0.5
            elif t_noise_dist == 'uniform':
                t = t_clean + torch.rand(1)[0] - 0.5
            else:
                raise ValueError("Invalid t noise distribution. Must be 'normal' or 'uniform'.")
            
            # add measurement error
            if me_dist == 'normal':
                t_me = t + torch.randn(1)[0] * me_std
            elif me_dist == 'laplace':
                t_me = t + Laplace(0, me_std).sample()
            else:
                raise ValueError("Invalid measurement error distribution. Must be 'normal' or 'laplace'.")

            y = t_x_y(t, x)
            y += torch.randn(1)[0] * 0.5

            y_matrix[_,:] = y
            t_matrix[_,:] = t
            t_clean_matrix[_,:] = t_clean
            t_me_matrix[_,:] = t_me

            # data_matrix[_, 0] = t
            # data_matrix[_, n_feature+1] = y
            # data_matrix[_, 1: n_feature+1] = x

        # get t_grid
        t_grid = torch.zeros(2, n_data)
        t_grid[0, :] = t_matrix.squeeze()

        for i in tqdm(range(n_data)):
            psi = 0
            t = t_grid[0, i]
            for j in range(n_data):
                x = covariates_matrix[j, :]
                psi += t_x_y(t, x)
            psi /= n_data
            t_grid[1, i] = psi

        # return data_matrix, t_grid

        return {'covariates_matrix': covariates_matrix,
                't_clean_matrix': t_clean_matrix,
                't_matrix': t_matrix,
                't_me_matrix': t_me_matrix,
                'y_matrix': y_matrix,
                't_grid': t_grid}

    # # simulate the data
    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)

    # data_file = os.path.join(args.save_dir, 'data_all.pkl')
    # if not os.path.exists(data_file):

    #     data_all = new_ihdp_matrix(t_noise_dist=args.t_noise_dist,
    #                            me_dist=args.me_dist,
    #                            me_std=args.me_std)
        
    #     with open(data_file, 'wb') as f:
    #         pickle.dump(data_all, f)
    #         f.close()

    # dm, tg = ihdp_matrix()
    # torch.save(dm, args.save_dir + '/data_matrix.pt')
    # torch.save(tg, args.save_dir + '/t_grid.pt')

    # generate splitting
    save_path = os.path.join(args.save_dir,
                             str(args.t_noise_dist)+'_'+str(args.me_dist),
                             str(args.me_std),
                             str(args.seed))
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    data_file = os.path.join(save_path, 'data_all.pkl')
    data_all = new_ihdp_matrix(t_noise_dist=args.t_noise_dist,
                               me_dist=args.me_dist,
                               me_std=args.me_std)
        
    with open(data_file, 'wb') as f:
        pickle.dump(data_all, f)
        f.close()
    
    idx_list = torch.randperm(n_data)
    idx_train = idx_list[0:3200]
    idx_test = idx_list[3200:]

    torch.save(idx_train, save_path + '/idx_train.pt')
    torch.save(idx_test, save_path + '/idx_test.pt')

    np.savetxt(save_path + '/idx_train.txt', idx_train.numpy())
    np.savetxt(save_path + '/idx_test.txt', idx_test.numpy())

    # for _ in range(args.num_eval):
    #     print('generating eval set: ', _)
    #     data_path = os.path.join(save_path, 'eval', str(_))
    #     if not os.path.exists(data_path):
    #         os.makedirs(data_path)

    #     idx_list = torch.randperm(n_data)
    #     idx_train = idx_list[0:471]
    #     idx_test = idx_list[471:]

    #     torch.save(idx_train, data_path + '/idx_train.pt')
    #     torch.save(idx_test, data_path + '/idx_test.pt')

    #     np.savetxt(data_path + '/idx_train.txt', idx_train.numpy())
    #     np.savetxt(data_path + '/idx_test.txt', idx_test.numpy())

    # for _ in range(args.num_tune):
    #     print('generating tuning set: ', _)
    #     data_path = os.path.join(save_path, 'tune', str(_))
    #     if not os.path.exists(data_path):
    #         os.makedirs(data_path)

    #     idx_list = torch.randperm(n_data)
    #     idx_train = idx_list[0:471]
    #     idx_test = idx_list[471:]

    #     torch.save(idx_train, data_path + '/idx_train.pt')
    #     torch.save(idx_test, data_path + '/idx_test.pt')

    #     np.savetxt(data_path + '/idx_train.txt', idx_train.numpy())
    #     np.savetxt(data_path + '/idx_test.txt', idx_test.numpy())