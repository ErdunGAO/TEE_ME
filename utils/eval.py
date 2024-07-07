import torch
from data.data import get_iter, get_iter_ihdp, get_iter_news, get_iter_tcga

def curve(model, test_matrix, t_grid, dataset_name, device=None):

    n_test = t_grid.shape[1]
    t_grid_hat = torch.zeros(2, n_test)
    t_grid_hat[0, :] = t_grid[0, :]

    if dataset_name == 'simulation':
        test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)
    elif dataset_name in ['ihdp', 'ihdp_add']:
        test_loader = get_iter_ihdp(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)
    elif dataset_name in ['news', 'news_add']:
        test_loader = get_iter_news(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)
    elif dataset_name == 'tcga':
        test_loader = get_iter_tcga(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)
    else:
        raise NotImplementedError

    for _ in range(n_test):
        for idx, (x, y, t_and_s) in enumerate(test_loader):
            t, s = t_and_s[:, 0], t_and_s[:, 1]
            if device is not None:
                x, y, t, s = x.to(device), y.to(device), t.to(device), s.to(device)
            t *= 0
            t += t_grid[0, _]
            break
        out = model.forward(s, t, x, y, mode='test')
        t_grid_hat[1, _] = out['y_pred'].data.squeeze().mean()
    
    mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data

    gt_dosage = t_grid[0, torch.argmax(t_grid[1, :]).item()]
    est_dosage = t_grid_hat[0, torch.argmax(t_grid_hat[1, :]).item()]
    pe = abs(gt_dosage - est_dosage)
    
    return t_grid_hat, mse, pe


