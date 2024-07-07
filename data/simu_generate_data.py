###############################################
# Generate the simulated data
###############################################
import os
import numpy as np

from simu_function import simu_data1, simu_data2, simu_data_v3
import argparse
import random
import torch
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate the simulated data')
    parser.add_argument('--save_dir', type=str, default='dataset/simu', help='dir to save generated data')
    # parser.add_argument('--num_eval', type=int, default=5, help='num of dataset for evaluating the methods')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--t_noise_dist', type=str, default='normal', help='type of the noise for t') # NOTE: not used currently
    parser.add_argument('--me_dist', type=str, default='normal', help='distribution of measurement error')
    parser.add_argument('--me_std', type=float, default=0, help='std of the measurement error')

    args = parser.parse_args()
    save_path = args.save_dir

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # for _ in range(args.num_eval):
    print('generating eval set with seed:', args.seed)
    data_path = os.path.join(save_path, str(args.t_noise_dist)+'_'+str(args.me_dist), str(args.me_std), str(args.seed))
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # # Version 2 for generating data
    # train_matrix, test_matrix, t_grid = simu_data2(10000, 500, me_std=args.me_std)
    # data_file = os.path.join(data_path, 'train.txt')
    # np.savetxt(data_file, train_matrix.numpy())
    # data_file = os.path.join(data_path, 'test.txt')
    # np.savetxt(data_file, test_matrix.numpy())
    # data_file = os.path.join(data_path, 't_grid.txt')
    # np.savetxt(data_file, t_grid.numpy())


    # Version 3 for generating data
    train_set, test_set = simu_data_v3(10000, 500, me_std=args.me_std)
    data_file = os.path.join(data_path, 'train_set.pkl')
    with open(data_file, 'wb') as f:
        pickle.dump(train_set, f)
        f.close()
    data_file = os.path.join(data_path, 'test_set.pkl')
    with open(data_file, 'wb') as f:
        pickle.dump(test_set, f)
        f.close()