import torch
from torch.distributions import Laplace

def x_t(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    t = (10. * torch.sin(max(x1, x2, x3)) + max(x3, x4, x5)**3)/(1. + (x1 + x5)**2) + \
        torch.sin(0.5 * x3) * (1. + torch.exp(x4 - 0.5 * x3)) + x3**2 + 2. * torch.sin(x4) + 2.*x5 - 6.5
    return t

def x_t_link(t):
    return 1. / (1. + torch.exp(-1. * t))

def t_x_y(t, x):
    # only x1, x3, x4 are useful
    x1 = x[0]
    x3 = x[2]
    x4 = x[3]
    x6 = x[5]
    y = torch.cos((t-0.5) * 3.14159 * 2.) * (t**2 + (4.*max(x1, x6)**3)/(1. + 2.*x3**2)*torch.sin(x4))
    return y

def simu_data1(n_train, n_test, me_std=None):
    '''
    Generate the simulated data for each sample. The generation fuction is according to the paper:
    "VCNet and Functional Targeted Regularization For Learning Causal Effects of Continuous Treatments" ICLR 2021.
    There are two main differences from the original paper:
    1. The data is generated with some measurement error.
    2. The data is arranged in a different order.

    Parameters
    ----------
    n_train: int
        number of training samples
    n_test: int
        number of testing samples
    me_var: float/None
        variance of the measurement error

    Returns
    -------
    train_matrix: torch.Tensor, shape (n_train, 7 + 2*num_t), The 1-6 columns are x, the 7th column is y, the last columns are paired t and measured t. (e.g. 8 is paired with 9)
        training data matrix
    test_matrix: torch.Tensor, shape (n_test, 7 + 2*num_t), The 1-6 columns are x, the 7th column is y, the last columns are paired t and measured t. (e.g. 8 is paired with 9)
        testing data matrix
    
    # In this version, we only have one t.
    #TODO: add more t, which means we need to change the shape of t_grid.)

    t_grid: torch.Tensor, shape (2, n_test)
        grid of t for estimating psi(t)
    '''

    train_matrix = torch.zeros(n_train, 9)
    test_matrix = torch.zeros(n_test, 9)

    for _ in range(n_train):

        x = torch.rand(6)
        train_matrix[_, 0:6] = x

        t = x_t(x)
        t += torch.randn(1)[0] * 0.5
        t = x_t_link(t)

        t_me = t + torch.randn(1)[0] * me_std if me_std is not None else 0 # add measurement error
        # train_matrix[_, 0] = t

        y = t_x_y(t, x)
        y += torch.randn(1)[0] * 0.5

        train_matrix[_, 6] = y
        train_matrix[_, 7] = t
        train_matrix[_, 8] = t_me


    for _ in range(n_test):

        x = torch.rand(6)
        test_matrix[_, 0:6] = x

        t = x_t(x)
        t += torch.randn(1)[0] * 0.5
        t = x_t_link(t)

        t_me = t + torch.randn(1)[0] * me_std if me_std is not None else 0 # add measurement error
        # test_matrix[_, 0] = t

        y = t_x_y(t, x)
        y += torch.randn(1)[0] * 0.5

        test_matrix[_, 6] = y
        test_matrix[_, 7] = t
        test_matrix[_, 8] = t_me

    # t_grid records the real response of t for each individual
    t_grid = torch.zeros(2, n_test)
    t_grid[0, :] = test_matrix[:, 7].squeeze()

    for i in range(n_test):
        psi = 0
        t = t_grid[0, i]
        for j in range(n_test):
            x = test_matrix[j, :6]
            psi += t_x_y(t, x)
        psi /= n_test
        t_grid[1, i] = psi

    return train_matrix, test_matrix, t_grid


# New function
def simu_data2(n_train,
               n_test,
               t_noise_dist='normal',
               me_dist='normal',
               me_std=None,
               seed=2023):
    '''
    Generate the simulated data for each sample. The generation fuction is according to the paper:
    "VCNet and Functional Targeted Regularization For Learning Causal Effects of Continuous Treatments" ICLR 2021.
    There are two main differences from the original paper:
    1. The data is generated with some measurement error.
    2. The data is arranged in a different order.

    Parameters
    ----------
    n_train: int
        number of training samples
    n_test: int
        number of testing samples
    t_noise_dist: str
        distribution of the noise for t
    me_dist: str
        distribution of measurement error
    me_var: float/None
        variance of the measurement error

    Returns
    -------
    train_matrix: torch.Tensor, shape (n_train, 7 + 2*num_t), The 1-6 columns are x, the 7th column is y, the last columns are paired t and measured t. (e.g. 8 is paired with 9)
        training data matrix
    test_matrix: torch.Tensor, shape (n_test, 7 + 2*num_t), The 1-6 columns are x, the 7th column is y, the last columns are paired t and measured t. (e.g. 8 is paired with 9)
        testing data matrix
    
    # In this version, we only have one t.
    #TODO: add more t, which means we need to change the shape of t_grid.)

    t_grid: torch.Tensor, shape (2, n_test)
        grid of t for estimating psi(t)
    '''

    train_matrix = torch.zeros(n_train, 10)
    test_matrix = torch.zeros(n_test, 10)

    for _ in range(n_train):

        x = torch.rand(6)
        train_matrix[_, 0:6] = x

        t = x_t(x)
        t_clean = x_t_link(t)
        if t_noise_dist == 'normal':
            t = t_clean + torch.randn(1)[0] * 0.5
        elif t_noise_dist == 'uniform':
            t = t_clean + torch.rand(1)[0] - 0.5
        else:
            raise ValueError("Invalid t noise distribution. Must be 'normal' or 'uniform'.")

        if me_dist == 'normal':
            t_me = t + torch.randn(1)[0] * me_std
        elif me_dist == 'laplace':
            t_me = t + Laplace(0, me_std).sample()
        else:
            raise ValueError("Invalid measurement error distribution. Must be 'normal' or 'laplace'.")

        # t_me = t + torch.randn(1)[0] * me_std if me_std is not None else 0 # add measurement error
        # train_matrix[_, 0] = t

        y = t_x_y(t, x)
        y += torch.randn(1)[0] * 0.5

        train_matrix[_, 6] = y
        train_matrix[_, 7] = t
        train_matrix[_, 8] = t_me
        train_matrix[_, 9] = t_clean


    for _ in range(n_test):

        x = torch.rand(6)
        test_matrix[_, 0:6] = x

        t = x_t(x)
        t_clean = x_t_link(t)
        if t_noise_dist == 'normal':
            t = t_clean + torch.randn(1)[0] * 0.5
        elif t_noise_dist == 'uniform':
            t = t_clean + torch.rand(1)[0] - 0.5
        else:
            raise ValueError("Invalid t noise distribution. Must be 'normal' or 'uniform'.")
        # t = t_clean + torch.randn(1)[0] * 0.5

        if me_dist == 'normal':
            t_me = t + torch.randn(1)[0] * me_std
        elif me_dist == 'laplace':
            t_me = t + Laplace(0, me_std).sample()
        else:
            raise ValueError("Invalid measurement error distribution. Must be 'normal' or 'laplace'.")
        

        # t_me = t + torch.randn(1)[0] * me_std if me_std is not None else 0 # add measurement error
        # test_matrix[_, 0] = t

        y = t_x_y(t, x)
        y += torch.randn(1)[0] * 0.5

        test_matrix[_, 6] = y
        test_matrix[_, 7] = t
        test_matrix[_, 8] = t_me
        test_matrix[_, 9] = t_clean

    # t_grid records the real response of t for each individual
    t_grid = torch.zeros(2, n_test)
    t_grid[0, :] = test_matrix[:, 7].squeeze()

    for i in range(n_test):
        psi = 0
        t = t_grid[0, i]
        for j in range(n_test):
            x = test_matrix[j, :6]
            psi += t_x_y(t, x)
        psi /= n_test
        t_grid[1, i] = psi

    return train_matrix, test_matrix, t_grid


# Insteading of using a matrix to record all data, we use a set to
# contain all data in this version.
def simu_data_v3(n_train,
                 n_test,
                 t_noise_dist='normal',
                 me_dist='normal',
                 me_std=None):
    '''
    Generate the simulated data for each sample. The generation fuction is according to the paper:
    "VCNet and Functional Targeted Regularization For Learning Causal Effects of Continuous Treatments" ICLR 2021.
    There are two main differences from the original paper:
    1. The data is generated with some measurement error.
    2. The data is arranged in a different order.

    Parameters
    ----------
    n_train: int
        number of training samples
    n_test: int
        number of testing samples
    t_noise_dist: str
        distribution of the noise for t
    me_dist: str
        distribution of measurement error
    me_var: float/None
        variance of the measurement error

    Returns
    -------
    train_matrix: torch.Tensor, shape (n_train, 7 + 2*num_t), The 1-6 columns are x, the 7th column is y, the last columns are paired t and measured t. (e.g. 8 is paired with 9)
        training data matrix
    test_matrix: torch.Tensor, shape (n_test, 7 + 2*num_t), The 1-6 columns are x, the 7th column is y, the last columns are paired t and measured t. (e.g. 8 is paired with 9)
        testing data matrix
    
    # In this version, we only have one t.
    #TODO: add more t, which means we need to change the shape of t_grid.)

    t_grid: torch.Tensor, shape (2, n_test)
        grid of t for estimating psi(t)
    '''

    # training data initialization
    covariates_train = torch.zeros(n_train, 6)
    t_clean_train = torch.zeros(n_train, 1)
    t_train = torch.zeros(n_train, 1)
    t_me_train = torch.zeros(n_train, 1)
    y_train = torch.zeros(n_train, 1)

    # testing data initialization
    covariates_test = torch.zeros(n_test, 6)
    t_clean_test = torch.zeros(n_test, 1)
    t_test = torch.zeros(n_test, 1)
    t_me_test = torch.zeros(n_test, 1)
    y_test = torch.zeros(n_test, 1)

    for _ in range(n_train):

        x = torch.rand(6)
        covariates_train[_, :] = x

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

        y_train[_, :] = y
        t_train[_, :] = t
        t_me_train[_, :] = t_me
        t_clean_train[_, :] = t_clean


    for _ in range(n_test):

        x = torch.rand(6)
        covariates_test[_, :] = x

        t = x_t(x)
        t_clean = x_t_link(t)

        # add noise, with std 0.5
        if t_noise_dist == 'normal':
            t = t_clean + torch.randn(1)[0] * 0.5
        elif t_noise_dist == 'uniform':
            t = t_clean + torch.rand(1)[0] - 0.5
        else:
            raise ValueError("Invalid t noise distribution. Must be 'normal' or 'uniform'.")
        # t = t_clean + torch.randn(1)[0] * 0.5

        # add measurement error
        if me_dist == 'normal':
            t_me = t + torch.randn(1)[0] * me_std
        elif me_dist == 'laplace':
            t_me = t + Laplace(0, me_std).sample()
        else:
            raise ValueError("Invalid measurement error distribution. Must be 'normal' or 'laplace'.")

        y = t_x_y(t, x)
        y += torch.randn(1)[0] * 0.5

        y_test[_, :] = y
        t_test[_, :] = t
        t_me_test[_, :] = t_me
        t_clean_test[_, :] = t_clean


    # t_grid records the real response of t for each individual
    t_grid = torch.zeros(2, n_test)
    t_grid[0, :] = t_test.squeeze()

    for i in range(n_test):
        psi = 0
        t = t_grid[0, i]
        for j in range(n_test):
            # x = test_matrix[j, :6]
            x = covariates_test[j, :]
            psi += t_x_y(t, x)
        psi /= n_test
        t_grid[1, i] = psi

    # return train_matrix, test_matrix, t_grid

    return {'covariates_train': covariates_train,
            't_clean_train': t_clean_train,
            't_train': t_train,
            't_me_train': t_me_train,
            'y_train': y_train}, \
              {'covariates_test': covariates_test,
                't_clean_test': t_clean_test,
                't_test': t_test,
                't_me_test': t_me_test,
                'y_test': y_test,
                't_grid': t_grid}




