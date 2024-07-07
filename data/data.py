import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle

class Dataset_from_matrix(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_matrix):
        """
        Args: create a torch dataset from a tensor data_matrix with size n * p
        [treatment, features, outcome]
        """
        self.data_matrix = data_matrix
        self.num_data = data_matrix.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_matrix[idx, :]
        return (sample[0:6], sample[6], sample[7:9])

def get_iter(data_matrix, batch_size, shuffle=True):
    dataset = Dataset_from_matrix(data_matrix)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return iterator


def load_simu_data(load_path, num_obs, batch_size):

    with open(load_path + '/train_set.pkl', 'rb') as f:
        train_set = pickle.load(f)
        f.close()
    train_matrix = torch.cat([train_set['covariates_train'],
                              train_set['y_train'],
                              train_set['t_train'],
                              train_set['t_me_train'],
                              train_set['t_clean_train'],], dim=1)


    # data = pd.read_csv(load_path + '/train.txt', header=None, sep=' ')
    train_matrix = torch.from_numpy(train_matrix.numpy()).float()
    train_matrix = train_matrix[0:num_obs, :]

    # data = pd.read_csv(load_path + '/test.txt', header=None, sep=' ')
    with open(load_path + '/test_set.pkl', 'rb') as f:
        test_set = pickle.load(f)
        f.close()
    test_matrix = torch.cat([test_set['covariates_test'],
                             test_set['y_test'],
                             test_set['t_test'],
                             test_set['t_me_test'],
                             test_set['t_clean_test'],], dim=1)

    test_matrix = torch.from_numpy(test_matrix.numpy()).float()

    idx = (test_matrix[:,7] > -0.3) * (test_matrix[:,7] < 1.3)
    test_matrix = test_matrix[idx]

    t_grid = test_set['t_grid']
    t_grid = torch.from_numpy(t_grid.numpy()).float()

    idx = (t_grid[0, :] > -0.3) * (t_grid[0, :] < 1.3)
    t_grid = t_grid[:, idx]

    train_loader = get_iter(train_matrix, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_matrix, t_grid

############################################
# load ihdp data
############################################

class Dataset_from_matrix_ihdp(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_matrix):
        """
        Args: create a torch dataset from a tensor data_matrix with size n * p
        [treatment, features, outcome]
        """
        self.data_matrix = data_matrix
        self.num_data = data_matrix.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_matrix[idx, :]
        return (sample[0:25], sample[25], sample[26:28])

def get_iter_ihdp(data_matrix, batch_size, shuffle=True):
    dataset = Dataset_from_matrix_ihdp(data_matrix)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return iterator

def load_ihdp_data(load_path, batch_size=100):

    # get data
    with open(load_path + '/data_all.pkl', 'rb') as f:
        data_all = pickle.load(f)
        f.close()
    data_matrix = torch.cat([data_all['covariates_matrix'],
                                data_all['y_matrix'],
                                data_all['t_matrix'],
                                data_all['t_me_matrix'],
                                data_all['t_clean_matrix'],], dim=1)
    
    data_matrix = torch.from_numpy(data_matrix.numpy()).float()
    # data_matrix = data_matrix[0:471, :]

    t_grid_all = data_all['t_grid']
    t_grid_all = torch.from_numpy(t_grid_all.numpy()).float()
    # t_grid_all = torch.load(load_path + '/t_grid.pt')

    idx_train = torch.load(load_path + '/idx_train.pt')
    idx_test = torch.load(load_path + '/idx_test.pt')

    train_matrix = data_matrix[idx_train, :]
    idx = (train_matrix[:,26] > 0) * (train_matrix[:,26] < 1.2)
    train_matrix = train_matrix[idx]
    train_loader = get_iter_ihdp(train_matrix, batch_size=batch_size, shuffle=True)

    test_matrix = data_matrix[idx_test, :]
    t_grid = t_grid_all[:, idx_test]

    idx = (test_matrix[:,26] > 0) * (test_matrix[:,26] < 1.2)
    test_matrix = test_matrix[idx]

    idx = (t_grid[0, :] > 0) * (t_grid[0, :] < 1.2)
    t_grid = t_grid[:, idx]

    return train_loader, test_matrix, t_grid

############################################
# load news data
############################################

class Dataset_from_matrix_news(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_matrix):
        """
        Args: create a torch dataset from a tensor data_matrix with size n * p
        [treatment, features, outcome]
        """
        self.data_matrix = data_matrix
        self.num_data = data_matrix.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_matrix[idx, :]
        return (sample[0:50], sample[50], sample[51:53])

def get_iter_news(data_matrix, batch_size, shuffle=True):
    dataset = Dataset_from_matrix_news(data_matrix)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return iterator

def load_news_data(load_path, batch_size=100):

    # get data
    with open(load_path + '/data_all.pkl', 'rb') as f:
        data_all = pickle.load(f)
        f.close()
    data_matrix = torch.cat([data_all['covariates_matrix'],
                                data_all['y_matrix'],
                                data_all['t_matrix'],
                                data_all['t_me_matrix'],], dim=1)
    
    data_matrix = torch.from_numpy(data_matrix.numpy()).float()
    # data_matrix = data_matrix[0:471, :]

    t_grid_all = data_all['t_grid']
    t_grid_all = torch.from_numpy(t_grid_all.numpy()).float()
    # t_grid_all = torch.load(load_path + '/t_grid.pt')

    idx_train = torch.load(load_path + '/idx_train.pt')
    idx_test = torch.load(load_path + '/idx_test.pt')

    # There is useless for us to pick out the data with t in [0, 1.3] because they are all in this set
    train_matrix = data_matrix[idx_train, :]
    idx = (train_matrix[:,51] > 0) * (train_matrix[:,51] < 1.3)
    train_matrix = train_matrix[idx]
    train_loader = get_iter_news(train_matrix, batch_size=batch_size, shuffle=True)

    test_matrix = data_matrix[idx_test, :]
    t_grid = t_grid_all[:, idx_test]

    idx = (test_matrix[:,51] > 0) * (test_matrix[:,51] < 1.3)
    test_matrix = test_matrix[idx]

    idx = (t_grid[0, :] > 0) * (t_grid[0, :] < 1.3)
    t_grid = t_grid[:, idx]

    return train_loader, test_matrix, t_grid


# ############################################
# # load news data
# ############################################

# class Dataset_from_matrix_news(Dataset):
#     """Face Landmarks dataset."""

#     def __init__(self, data_matrix):
#         """
#         Args: create a torch dataset from a tensor data_matrix with size n * p
#         [treatment, features, outcome]
#         """
#         self.data_matrix = data_matrix
#         self.num_data = data_matrix.shape[0]

#     def __len__(self):
#         return self.num_data

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         sample = self.data_matrix[idx, :]
#         return (sample[0:498], sample[498], sample[499:501])

# def get_iter_news(data_matrix, batch_size, shuffle=True):
#     dataset = Dataset_from_matrix_news(data_matrix)
#     iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#     return iterator

# def load_news_data(load_path, batch_size=100):

#     # get data
#     with open(load_path + '/data_all.pkl', 'rb') as f:
#         data_all = pickle.load(f)
#         f.close()
#     data_matrix = torch.cat([data_all['covariates_matrix'],
#                                 data_all['y_matrix'],
#                                 data_all['t_matrix'],
#                                 data_all['t_me_matrix'],], dim=1)
    
#     data_matrix = torch.from_numpy(data_matrix.numpy()).float()
#     # data_matrix = data_matrix[0:471, :]

#     t_grid_all = data_all['t_grid']
#     t_grid_all = torch.from_numpy(t_grid_all.numpy()).float()
#     # t_grid_all = torch.load(load_path + '/t_grid.pt')

#     idx_train = torch.load(load_path + '/idx_train.pt')
#     idx_test = torch.load(load_path + '/idx_test.pt')

#     # There is useless for us to pick out the data with t in [0, 1.3] because they are all in this set
#     train_matrix = data_matrix[idx_train, :]
#     idx = (train_matrix[:,499] > 0) * (train_matrix[:,499] < 1.3)
#     train_matrix = train_matrix[idx]
#     train_loader = get_iter_news(train_matrix, batch_size=batch_size, shuffle=True)

#     test_matrix = data_matrix[idx_test, :]
#     t_grid = t_grid_all[:, idx_test]

#     idx = (test_matrix[:,499] > 0) * (test_matrix[:,499] < 1.3)
#     test_matrix = test_matrix[idx]

#     idx = (t_grid[0, :] > 0) * (t_grid[0, :] < 1.3)
#     t_grid = t_grid[:, idx]

#     return train_loader, test_matrix, t_grid

############################################
# load simu_s1 data
############################################
class Dataset_from_matrix_s1(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_matrix):
        """ß
        Args: create a torch dataset from a tensor data_matrix with size n * p
        [treatment, features, outcome]
        """
        self.data_matrix = data_matrix
        self.num_data = data_matrix.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_matrix[idx, :]
        return (sample[0:1], sample[1], sample[2:4])

def get_iter_s1(data_matrix, batch_size, shuffle=True):
    dataset = Dataset_from_matrix_s1(data_matrix)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return iterator


def load_simu_s1_data(load_path, num_obs, batch_size):

    with open(load_path + '/train_set.pkl', 'rb') as f:
        train_set = pickle.load(f)
        f.close()
    train_matrix = torch.cat([train_set['covariates_train'],
                              train_set['y_train'],
                              train_set['t_train'],
                              train_set['t_me_train'],
                              train_set['t_clean_train'],], dim=1)


    # data = pd.read_csv(load_path + '/train.txt', header=None, sep=' ')
    train_matrix = torch.from_numpy(train_matrix.numpy()).float()
    train_matrix = train_matrix[0:num_obs, :]

    # data = pd.read_csv(load_path + '/test.txt', header=None, sep=' ')
    with open(load_path + '/test_set.pkl', 'rb') as f:
        test_set = pickle.load(f)
        f.close()
    test_matrix = torch.cat([test_set['covariates_test'],
                             test_set['y_test'],
                             test_set['t_test'],
                             test_set['t_me_test'],
                             test_set['t_clean_test'],], dim=1)

    test_matrix = torch.from_numpy(test_matrix.numpy()).float()

    idx = (test_matrix[:,2] > -0.3) * (test_matrix[:,2] < 1.2)
    test_matrix = test_matrix[idx]

    t_grid = test_set['t_grid']
    t_grid = torch.from_numpy(t_grid.numpy()).float()

    idx = (t_grid[0, :] > -0.3) * (t_grid[0, :] < 1.2)
    t_grid = t_grid[:, idx]

    train_loader = get_iter_s1(train_matrix, batch_size=batch_size, shuffle=True)
    
    return train_loader, train_matrix, test_matrix, t_grid


############################################
# load ihdp data
############################################

class Dataset_from_matrix_tcga(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_matrix):
        """
        Args: create a torch dataset from a tensor data_matrix with size n * p
        [treatment, features, outcome]
        """
        self.data_matrix = data_matrix
        self.num_data = data_matrix.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_matrix[idx, :]
        return (sample[0:4000], sample[4000], sample[4001:4003])

def get_iter_tcga(data_matrix, batch_size, shuffle=True):
    dataset = Dataset_from_matrix_tcga(data_matrix)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return iterator

def load_tcga_data(load_path, batch_size=100):

    # get data
    with open(load_path + '/data_all.pkl', 'rb') as f:
        data_all = pickle.load(f)
        f.close()
    data_matrix = torch.cat([data_all['covariates_matrix'],
                                data_all['y_matrix'],
                                data_all['t_matrix'],
                                data_all['t_me_matrix'],
                                data_all['t_clean_matrix'],], dim=1)
    
    data_matrix = torch.from_numpy(data_matrix.numpy()).float()

    t_grid_all = data_all['t_grid']
    t_grid_all = torch.from_numpy(t_grid_all.numpy()).float()
    # t_grid_all = torch.load(load_path + '/t_grid.pt')

    idx_train = torch.load(load_path + '/idx_train.pt')
    idx_test = torch.load(load_path + '/idx_test.pt')

    t_index = 4001

    train_matrix = data_matrix[idx_train, :]
    idx = (train_matrix[:,t_index] > 0) * (train_matrix[:,t_index] < 1.2)
    train_matrix = train_matrix[idx]
    train_loader = get_iter_tcga(train_matrix, batch_size=batch_size, shuffle=True)

    test_matrix = data_matrix[idx_test, :]
    t_grid = t_grid_all[:, idx_test]

    idx = (test_matrix[:,t_index] > 0) * (test_matrix[:,t_index] < 1.2)
    test_matrix = test_matrix[idx]

    idx = (t_grid[0, :] > 0) * (t_grid[0, :] < 1.2)
    t_grid = t_grid[:, idx]

    return train_loader, test_matrix, t_grid

