
# from utils.utils import progress_bar
import torch
import torch.nn as nn

class MeasureErrorLoss(nn.Module):
    def __init__(self, error_std, me_dist='normal'):
        super(MeasureErrorLoss, self).__init__()
        self.error_std = error_std
        self.me_dist = me_dist

    def forward(self, est_error):
        if self.me_dist == 'normal':
            # Gaussian distribution
            probability = 0.5 * (est_error / self.error_std) ** 2
        elif self.me_dist == 'laplace':
            # laplace distribution
            probability = torch.abs(est_error) / self.error_std
        else:
            raise ValueError("Invalid noise distribution. Must be 'gaussian' or 'laplace'.")

        # negative log likelihood
        loss = torch.mean(probability)

        return loss


# Training
def train_VF_VCNet(net,
          train_loader,
          optimizer,
          me_dist='gaussian',
          me_std=0.1,
          y_coeff=1,
          alpha=1,
          beta=1,
          epoch=0,
          device=None):

    for _, (x, y, t_and_s) in enumerate(train_loader):
        t, s = t_and_s[:, 0], t_and_s[:, 1]
        if device is not None:
            x, y, t, s = x.to(device), y.to(device), t.to(device), s.to(device)
        optimizer.zero_grad()
        out = net.forward(s, t, x, y, mode='train', device=device)

        # reconstruction loss, including outcome and measurement error terms
        f_mse_loss = nn.MSELoss()
        loss_y = f_mse_loss(out['y_pred'].squeeze(), y)

        f_me_loss = MeasureErrorLoss(error_std=me_std, me_dist=me_dist)
        loss_me = f_me_loss(out['u_est'])

        # KL divergence loss
        kl_loss = -(torch.mean(out['log_prior'] - out['log_posterior']))

        loss = y_coeff * loss_y + alpha * loss_me + beta * kl_loss
        
        loss.backward()
        optimizer.step()


def train_VCNet(model,
            train_loader,
            optimizer,
            alpha=0.5,
            gt=True):

    f_mse_loss = nn.MSELoss(reduce=True, size_average=True)
    for _, (x, y, t_all) in enumerate(train_loader):
        t = t_all[:, 0] if gt else t_all[:, 1]
        optimizer.zero_grad()
        out = model.forward(t, x, mode='train')
        loss_y = f_mse_loss(out['Q'].squeeze(), y)
        loss_t = torch.mean(out['g'])
        loss = loss_y - alpha * loss_t

        loss.backward()
        optimizer.step()
