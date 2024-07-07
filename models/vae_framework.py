import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F


class Truncated_power():
    def __init__(self, degree, knots):
        """
        This class construct the truncated power basis; the data is assumed in [0,1]
        :param degree: int, the degree of truncated basis
        :param knots: list, the knots of the spline basis; two end points (0,1) should not be included
        """
        self.degree = degree
        self.knots = knots
        self.num_of_basis = self.degree + 1 + len(self.knots)
        self.relu = nn.ReLU(inplace=True)

        if self.degree == 0:
            print('Degree should not set to be 0!')
            raise ValueError

        if not isinstance(self.degree, int):
            print('Degree should be int')
            raise ValueError

    def forward(self, x):
        """
        :param x: torch.tensor, batch_size * 1
        :return: the value of each basis given x; batch_size * self.num_of_basis
        """
        x = x.squeeze()
        out = torch.zeros(x.shape[0], self.num_of_basis)
        for _ in range(self.num_of_basis):
            if _ <= self.degree:
                if _ == 0:
                    out[:, _] = 1.
                else:
                    out[:, _] = x**_
            else:
                if self.degree == 1:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree]))
                else:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree - 1])) ** self.degree

        return out # bs, num_of_basis


class Dynamic_FC(nn.Module):
    def __init__(self, ind, outd, degree, knots, act='relu', isbias=1, islastlayer=0):
        super(Dynamic_FC, self).__init__()
        self.ind = ind
        self.outd = outd
        self.degree = degree
        self.knots = knots

        self.islastlayer = islastlayer

        self.isbias = isbias

        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis # num of basis

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd, self.d), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd, self.d), requires_grad=True)
        else:
            self.bias = None

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        # x: batch_size * (treatment, other feature)
        x_feature = x[:, 1:]
        x_treat = x[:, 0]

        x_feature_weight = torch.matmul(self.weight.T, x_feature.T).T # bs, outd, d

        x_treat_basis = self.spb.forward(x_treat) # bs, d
        x_treat_basis_ = torch.unsqueeze(x_treat_basis, 1)

        # x_feature_weight * x_treat_basis; bs, outd, d
        out = torch.sum(x_feature_weight * x_treat_basis_, dim=2) # bs, outd

        if self.isbias:
            out_bias = torch.matmul(self.bias, x_treat_basis.T).T
            out = out + out_bias

        if self.act is not None:
            out = self.act(out)

        # concat the treatment for intermediate layer
        if not self.islastlayer:
            out = torch.cat((torch.unsqueeze(x_treat, 1), out), 1)

        return out


class DensityEstimator(nn.Module):
    def __init__(self, t_dim, ind, isbias=True):
        super(DensityEstimator, self).__init__()
        """
        Assume the conditional prior p(t|x) is a gaussian distribution.
        This module can estimate the mean and variance of the conditional prior distribution.
        """
        self.t_dim = t_dim
        self.ind = ind
        self.isbias = isbias

        self.weight_mean = nn.Parameter(torch.rand(self.ind, self.t_dim), requires_grad=True)
        self.weight_log_sigma = nn.Parameter(torch.rand(self.ind, self.t_dim), requires_grad=True)

        if self.isbias:
            self.bias_mean = nn.Parameter(torch.rand(self.t_dim), requires_grad=True)
            self.bias_log_sigma = nn.Parameter(torch.rand(self.t_dim), requires_grad=True)

    def forward(self, x):
        prior_mean = torch.matmul(x, self.weight_mean)
        prior_log_sigma = torch.matmul(x, self.weight_log_sigma)
        if self.isbias:
            prior_mean += self.bias_mean
            prior_log_sigma += self.bias_log_sigma

        prior_sigma = torch.exp(prior_log_sigma)

        return prior_mean, prior_sigma

class q_phi(nn.Module):
    def __init__(self, layer=3, nodes=8, activ='relu', input_dim=2, output_dim=1):
        super(q_phi, self).__init__()
        """
        This module is used for estimating the posterior distribution of the treatment given the measurement and outcome, i.e., q(t|s,y)
        One can use this module to estimate the mean and variance of the posterior distribution, i.e., q(t|s,y) = N(mu, sigma^2)
        """

        self.layer = layer
        self.nodes = nodes
        self.activ = activ
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, nodes))
        self.layers.append(nn.ReLU())
        
        for _ in range(layer):
            self.layers.append(nn.Linear(nodes, nodes))
            self.layers.append(nn.ReLU())
            
        self.layers.append(nn.Linear(nodes, output_dim))

        # Initialize the weights and biases of the linear layers
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

class New_Density_Block(nn.Module):
    def __init__(self, ind, isbias=1):
        super(New_Density_Block, self).__init__()
        """
        Assume the variable is bounded by [0,1]
        the output grid: 0, 1/B, 2/B, ..., B/B; output dim = B + 1; num_grid = B
        """
        self.ind = ind
        self.isbias = isbias

        self.weight = nn.Parameter(torch.rand(self.ind, 2), requires_grad=True)
        if self.isbias:
            self.bias = nn.Parameter(torch.rand(2), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        out = torch.matmul(x, self.weight)
        if self.isbias:
            out += self.bias
        
        mean, sigma = out[:, 0], torch.exp(out[:, 1])
        mean = torch.unsqueeze(mean, 1)
        sigma = torch.unsqueeze(sigma, 1)

        return mean, sigma

class MD_Network(nn.Module):
    '''
    This is the mixture density network.
    '''
    def __init__(self, ind, n_components, isbias=1):
        super(MD_Network, self).__init__()
        self.ind = ind
        self.n_components = n_components
        self.isbias = isbias

        self.weight = nn.Parameter(torch.rand(self.ind, 2*self.n_components), requires_grad=True)
        self.pi_weight = nn.Parameter(torch.rand(self.ind, self.n_components), requires_grad=True)
        if self.isbias:
            self.bias = nn.Parameter(torch.rand(2*self.n_components), requires_grad=True)

        
    def forward(self, x):
        if self.n_components == 1:
            pi = torch.ones(x.shape[0], 1)
        else:
            pi = F.softmax(torch.matmul(x, self.pi_weight), dim=1)
        
        out = torch.matmul(x, self.weight)
        if self.isbias:
            out += self.bias
        
        mean, sigma = out[:, :self.n_components], torch.exp(out[:, self.n_components:])
        
        if self.n_components == 1:
            mean = torch.unsqueeze(mean, 1)
            sigma = torch.unsqueeze(sigma, 1)

        return pi, mean, sigma


class MLP_simple(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        
        # Use super __init__ to inherit from parent nn.Module class
        super(MLP_simple, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, hidden_features)
        self.fc4 = nn.Linear(hidden_features, out_features)
                
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x) # ReLU activation
        x = self.fc2(x)
        x = F.relu(x) # ReLU activation
        x = self.fc3(x)
        x = F.relu(x) # ReLU activation
        x = self.fc4(x)
        
        return x
        

class VF_VCNet(nn.Module):
    def __init__(self, x_dim, t_dim=1, n_components=3):
        super(VF_VCNet, self).__init__()
        """
        cfg_density: cfg for the density estimator; [(ind1, outd1, isbias1), 'act', ....]; the cfg for density estimator head is not included
        num_grid: how many grid used for the density estimator head
        """

        self.t_dim = t_dim
        self.hidden_dim = 64
        self.n_components = n_components

        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]
        self.cfg_density = [(x_dim, 50, 1, 'relu'), (50, 50, 1, 'relu')]
        self.cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
        self.degree = 2
        self.knots = [0.33, 0.66]

        # construct the density estimator
        density_blocks = []
        density_hidden_dim = -1
        for layer_idx, layer_cfg in enumerate(self.cfg_density):
            # fc layer
            if layer_idx == 0:
                # weight connected to feature
                self.feature_weight = nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2])
                density_blocks.append(self.feature_weight)
            else:
                density_blocks.append(nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2]))
            density_hidden_dim = layer_cfg[1]
            if layer_cfg[3] == 'relu':
                density_blocks.append(nn.ReLU(inplace=True))
            elif layer_cfg[3] == 'tanh':
                density_blocks.append(nn.Tanh())
            elif layer_cfg[3] == 'sigmoid':
                density_blocks.append(nn.Sigmoid())
            else:
                print('No activation')

        self.hidden_features = nn.Sequential(*density_blocks)

        # construct the density estimator head
        self.density_hidden_dim = density_hidden_dim
        self.density_estimator_head = MD_Network(density_hidden_dim, n_components=self.n_components, isbias=1)


        self.q_phi_mu = MLP_simple(in_features=self.t_dim+1, hidden_features=self.hidden_dim, out_features=self.t_dim)
        self.q_phi_log_sigma = MLP_simple(in_features=self.t_dim+1, hidden_features=self.hidden_dim, out_features=self.t_dim)

        # construct the dynamics network
        blocks = []
        for layer_idx, layer_cfg in enumerate(self.cfg):
            if layer_idx == len(self.cfg)-1: # last layer
                last_layer = Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=1)
            else:
                blocks.append(
                    Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=0))
        blocks.append(last_layer)
        self.Q = nn.Sequential(*blocks)

    def forward(self, s, t, x, y, mode='train', device=None):
        hidden = self.hidden_features(x)
        if mode == 'train':
            # estimating the mean and variance of the posterior distribution q(t|s,y)
            s = torch.unsqueeze(s, 1) if len(s.shape) == 1 else s
            s_and_y = torch.cat((s, torch.unsqueeze(y, 1)), 1)
            mu = self.q_phi_mu(s_and_y)
            log_sigma = self.q_phi_log_sigma(s_and_y)
            sigma = torch.exp(log_sigma)

            t_repara = self.reparameterize(mu, sigma, noise=True, device=device)

            # check the square distance between t and mu estimated from s and y
            t_error_posterior = torch.mean((t - mu) ** 2).mean().data

            prior_pi, prior_mean, prior_sigma = self.density_estimator_head(hidden)
            for i in range(self.n_components):
                # prior_normal = dist.normal.Normal(torch.unsqueeze(prior_mean[:, i],1), torch.unsqueeze(prior_sigma[:, i], 1))
                prior_normal = dist.normal.Normal(prior_mean[:, i], prior_sigma[:, i]) # 9.18 test new code
                if i == 0:
                    prior = torch.unsqueeze(prior_pi[:, i],1) * torch.exp(prior_normal.log_prob(t_repara))
                else:
                    prior += torch.unsqueeze(prior_pi[:, i],1) * torch.exp(prior_normal.log_prob(t_repara))

            log_prior = torch.log(prior)


            # calculating the log_posterior probability
            t_normal = dist.normal.Normal(mu, sigma)
            log_posterior = t_normal.log_prob(t_repara)

            # estimating the measurement error
            u_est = s - t_repara

            # estimating the outcome
            t_hidden = torch.cat((t_repara, hidden), 1)
            y_pred = self.Q(t_hidden)

            return {'y_pred': y_pred,
                    'log_prior': log_prior,
                    'log_posterior': log_posterior,
                    'u_est': u_est,
                    't_error_posterior': t_error_posterior,
                    'posterior_sigma': sigma,
                    't_repara': t_repara,
                    'prior': prior}
        
        elif mode == 'pretrain':
            s_hidden = torch.cat((torch.unsqueeze(s, 1), hidden), 1)
            y_pred = self.Q(s_hidden)
            mean, sigma = self.density_estimator_head(hidden)
            prior_normal = dist.normal.Normal(mean, sigma)
            log_prior = prior_normal.log_prob(s)

            return {'y_pred': y_pred,
                    'log_prior': log_prior}
        
        elif mode == 'u_est':

            s = torch.unsqueeze(s, 1) if len(s.shape) == 1 else s
            s_and_y = torch.cat((s, torch.unsqueeze(y, 1)), 1)
            mu = self.q_phi_mu(s_and_y)
            log_sigma = self.q_phi_log_sigma(s_and_y)
            sigma = torch.exp(log_sigma)
            t_repara = self.reparameterize(mu, sigma, noise=True, device=device)

            u_gt = s - torch.unsqueeze(t, 1)
            u_est = s - t_repara

            return u_gt, u_est


        elif mode == 't_est':

            _, prior_mean, prior_sigma = self.density_estimator_head(hidden)
            return prior_mean, prior_sigma
        
        elif mode == 't_est_2':

            pi, prior_mean, prior_sigma = self.density_estimator_head(hidden)
            return pi, prior_mean, prior_sigma

        else:
            t = torch.unsqueeze(t, 1) if len(t.shape) == 1 else t
            t_hidden = torch.cat((t, hidden), 1)
            y_pred = self.Q(t_hidden)

            return {'y_pred': y_pred}
    
    def reparameterize(self, mu, sigma, noise=True, device=None):
        """
        :param mu: mean of the distribution
        :param sigma: std of the distribution
        :param noise: whether to add noise
        :param device: device
        :return: reparameterized sample
        """
        zero_mean = torch.zeros(mu.shape[1]).to(device) if device is not None else torch.zeros(mu.shape[1])
        ones_sigma = torch.ones(sigma.shape[1]).to(device) if device is not None else torch.ones(sigma.shape[1])
        normal = dist.normal.Normal(zero_mean, ones_sigma)
        eps = normal.sample([sigma.shape[0]]).to(device) if device is not None else normal.sample([sigma.shape[0]])

        return mu + eps * sigma if noise else mu

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, Dynamic_FC):
                m.weight.data.normal_(0, 1.)
                if m.isbias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            # elif isinstance(m, New_Density_Block):
            #     m.weight.data.normal_(0, 0.01)
            #     if m.isbias:
            #         m.bias.data.zero_()
            elif isinstance(m, MLP_simple):
                m.fc1.weight.data.normal_(0, 0.01)
                m.fc2.weight.data.normal_(0, 0.01)
                m.fc3.weight.data.normal_(0, 0.01)
                m.fc4.weight.data.normal_(0, 0.01)
                if m.fc1.bias is not None:
                    m.fc1.bias.data.zero_()
                if m.fc2.bias is not None:
                    m.fc2.bias.data.zero_()
                if m.fc3.bias is not None:
                    m.fc3.bias.data.zero_()
                if m.fc4.bias is not None:
                    m.fc4.bias.data.zero_()
    