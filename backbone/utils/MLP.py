"""Author: https://github.com/GitGyun/multi_task_neural_processes/blob/main/model/mlp.py"""
import torch
import torch.nn as nn
from backbone import xavier

@torch.jit.script
def soft_clamp5(x: torch.Tensor):
    return x.div(5.).tanh_().mul(5.)

class FFB(nn.Module):
    def __init__(self, dim_in, dim_out, act_fn, ln):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.LayerNorm(dim_out) if ln else nn.Identity(),
            act_fn(),
        )

    def forward(self, x):
        return self.layers(x)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, n_layers=2, act_fn='relu', ln=True, xavier_init=True):
        super().__init__()
        assert n_layers >= 1
        act_fn = nn.GELU if act_fn == 'gelu' else nn.ReLU

        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out

        layers = []
        for l_idx in range(n_layers):
            di = dim_in if l_idx == 0 else dim_hidden
            do = dim_out if l_idx == n_layers - 1 else dim_hidden
            layers.append(FFB(di, do, act_fn, ln))

        self.layers = nn.Sequential(*layers)
        if xavier_init:
            self.reset_parameters()

    def forward(self, x):
        x = self.layers(x)

        return x

    def reset_parameters(self):
        self.layers.apply(xavier)

class LatentMLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, n_layers=2, act_fn='relu', ln=True,
                 epsilon=0.1, sigma=True, sigma_act=torch.sigmoid, xavier_init=True):
        super().__init__()

        self.epsilon = epsilon
        self.sigma = sigma

        assert n_layers >= 1
        if n_layers >= 2:
            self.mlp = MLP(dim_in, dim_hidden, dim_hidden, n_layers - 1, act_fn, ln, xavier_init=xavier_init)
        else:
            dim_hidden = dim_in
            self.mlp = None

        self.hidden_to_mu = nn.Linear(dim_hidden, dim_out)
        if self.sigma:
            self.hidden_to_log_sigma = nn.Linear(dim_hidden, dim_out)
            self.sigma_act = sigma_act

        if xavier_init:
            self.net = nn.Sequential(self.hidden_to_mu, self.hidden_to_log_sigma)
            self.reset_parameters()

    def reset_parameters(self):
        self.net.apply(xavier)

    def forward(self, x):
        hidden = self.mlp(x) if self.mlp is not None else x

        mu = self.hidden_to_mu(hidden)
        if self.sigma:
            log_sigma = self.hidden_to_log_sigma(hidden)
            sigma = torch.exp(log_sigma) + 1e-2
            sigma = torch.clamp(sigma, min=0.01) #, max=10.0)
            # sigma = 0.1 + 0.9 * nn.functional.softplus(log_sigma)

            return mu, sigma
        else:
            return mu