import numpy as np
import json
import pandas as pd
import torch
import os

from tqdm import tqdm
import argparse

from torch.distributions import Laplace
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate news data')
    parser.add_argument('--data_path', type=str, default='dataset/news/news_pp.npy', help='data path')
    parser.add_argument('--save_dir', type=str, default='dataset/news/gene_data', help='dir to save generated data')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--t_noise_dist', type=str, default='normal', help='type of the noise for t') # NOTE: not used currently
    parser.add_argument('--me_dist', type=str, default='normal', help='distribution of measurement error')
    parser.add_argument('--me_std', type=float, default=0.2, help='std of the measurement error')
    # parser.add_argument('--num_eval', type=int, default=10, help='num of dataset for evaluating the methods')
    # parser.add_argument('--num_tune', type=int, default=2, help='num of dataset for tuning the parameters')

    args = parser.parse_args()
    save_path = args.save_dir

    # load data
    path = args.data_path
    news = np.load(path)
    #
    # # normalize data
    for _ in range(news.shape[1]):
        max_freq = max(news[:,_])
        news[:,_] = news[:,_] / max_freq

    num_data = news.shape[0]
    num_feature = news.shape[1]

    np.random.seed(args.seed)
    v1 = np.random.randn(num_feature)
    v1 = v1/np.sqrt(np.sum(v1**2))
    v2 = np.random.randn(num_feature)
    v2 = v2/np.sqrt(np.sum(v2**2))
    v3 = np.random.randn(num_feature)
    v3 = v3/np.sqrt(np.sum(v3**2))

    def x_t(x):
        alpha = 2
        tt = np.sum(v3 * x) / (2. * np.sum(v2 * x))
        beta = (alpha - 1)/tt + 2 - alpha
        beta = np.abs(beta) + 0.0001
        t = np.random.beta(alpha, beta, 1)
        return t

    def t_x_y(t, x):
        res1 = max(-2, min(2, np.exp(0.3 * (np.sum(3.14159 * np.sum(v2 * x) / np.sum(v3 * x)) - 1))))
        res2 = 20. * (np.sum(v1 * x))
        res = 2 * (4 * (t - 0.5)**2 * np.sin(0.5 * 3.14159 * t)) * (res1 + res2)
        return res

    # def news_matrix():
    #     data_matrix = torch.zeros(num_data, num_feature+2)
    #     # get data matrix
    #     for _ in range(num_data):
    #         x = news[_, :]
    #         t = x_t(x)
    #         y = torch.from_numpy(t_x_y(t, x))
    #         x = torch.from_numpy(x)
    #         t = torch.from_numpy(t)
    #         y += torch.randn(1)[0] * np.sqrt(0.5)

    #         data_matrix[_, 0] = t
    #         data_matrix[_, num_feature+1] = y
    #         data_matrix[_, 1: num_feature+1] = x

    #     # get t_grid
    #     t_grid = torch.zeros(2, num_data)
    #     t_grid[0, :] = data_matrix[:, 0].squeeze()

    #     for i in tqdm(range(num_data)):
    #         psi = 0
    #         t = t_grid[0, i].numpy()
    #         for j in range(num_data):
    #             x = data_matrix[j, 1: num_feature+1].numpy()
    #             psi += t_x_y(t, x)
    #         psi /= num_data
    #         t_grid[1, i] = psi

    #     return data_matrix, t_grid

    def new_news_matrix(t_noise_dist='normal', me_dist='normal', me_std=None):

        #TODO delete the t_noise_dist term, we use current version to guaratee that all the function api are the same

        covariates_matrix = torch.zeros(num_data, num_feature)
        t_matrix = torch.zeros(num_data, 1)
        t_me_matrix = torch.zeros(num_data, 1)
        y_matrix = torch.zeros(num_data, 1)

        # get data matrix
        for _ in range(num_data):
            
            x = news[_, :]
            covariates_matrix[_, :] = torch.from_numpy(x)

            t = x_t(x)
            t = torch.from_numpy(t)

            # add measurement error
            if me_dist == 'normal':
                t_me = t + torch.randn(1)[0] * me_std
            elif me_dist == 'laplace':
                t_me = t + Laplace(0, me_std).sample()
            else:
                raise ValueError("Invalid measurement error distribution. Must be 'normal' or 'laplace'.")
            
            y = t_x_y(t, x)
            y += torch.randn(1)[0] * np.sqrt(0.5)

            # data_matrix[_, 0] = t
            # data_matrix[_, num_feature+1] = y
            # data_matrix[_, 1: num_feature+1] = x

            y_matrix[_, :] = y
            t_matrix[_, :] = t
            t_me_matrix[_, :] = t_me

        # get t_grid
        t_grid = torch.zeros(2, num_data)
        t_grid[0, :] = t_matrix.squeeze()

        for i in tqdm(range(num_data)):
            psi = 0
            t = t_grid[0, i].numpy()
            for j in range(num_data):
                # x = data_matrix[j, 1: num_feature+1].numpy()
                x = covariates_matrix[j, :].numpy()
                psi += t_x_y(t, x)
            psi /= num_data
            t_grid[1, i] = psi

        # return data_matrix, t_grid
        return {'covariates_matrix': covariates_matrix,
                't_matrix': t_matrix,
                't_me_matrix': t_me_matrix,
                'y_matrix': y_matrix,
                't_grid': t_grid}
    
    # generate splitting
    save_path = os.path.join(args.save_dir,
                             str(args.t_noise_dist)+'_'+str(args.me_dist),
                             str(args.me_std),
                             str(args.seed))
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    data_file = os.path.join(save_path, 'data_all.pkl')
    data_all = new_news_matrix(t_noise_dist=args.t_noise_dist,
                               me_dist=args.me_dist,
                               me_std=args.me_std)
        
    with open(data_file, 'wb') as f:
        pickle.dump(data_all, f)
        f.close()
    
    idx_list = torch.randperm(num_data)
    idx_train = idx_list[0:2500]
    idx_test = idx_list[2500:]

    torch.save(idx_train, save_path + '/idx_train.pt')
    torch.save(idx_test, save_path + '/idx_test.pt')

    np.savetxt(save_path + '/idx_train.txt', idx_train.numpy())
    np.savetxt(save_path + '/idx_test.txt', idx_test.numpy())

    # dm, tg = news_matrix()

    # torch.save(dm, save_path + '/data_matrix.pt')
    # torch.save(tg, save_path + '/t_grid.pt')

    # # generate eval splitting
    # for _ in range(args.num_eval):
    #     print('generating eval set: ', _)
    #     data_path = os.path.join(save_path, 'eval', str(_))
    #     if not os.path.exists(data_path):
    #         os.makedirs(data_path)

    #     idx_list = torch.randperm(num_data)
    #     idx_train = idx_list[0:2000]
    #     idx_test = idx_list[2000:]

    #     torch.save(idx_train, data_path + '/idx_train.pt')
    #     torch.save(idx_test, data_path + '/idx_test.pt')

    #     np.savetxt(data_path + '/idx_train.txt', idx_train.numpy())
    #     np.savetxt(data_path + '/idx_test.txt', idx_test.numpy())

    # # generate tuning splitting
    # for _ in range(args.num_tune):
    #     print('generating eval set: ', _)
    #     data_path = os.path.join(save_path, 'tune', str(_))
    #     if not os.path.exists(data_path):
    #         os.makedirs(data_path)

    #     idx_list = torch.randperm(num_data)
    #     idx_train = idx_list[0:2000]
    #     idx_test = idx_list[2000:]

    #     torch.save(idx_train, data_path + '/idx_train.pt')
    #     torch.save(idx_test, data_path + '/idx_test.pt')

    #     np.savetxt(data_path + '/idx_train.txt', idx_train.numpy())
    #     np.savetxt(data_path + '/idx_test.txt', idx_test.numpy())