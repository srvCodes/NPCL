from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
import torch
import numpy as np

def compute_kl_div(new_mu, new_sigma, old_mu, old_sigma, residual_normal=False):
    if residual_normal:
        """
        Source: https://github.com/NVlabs/NVAE/blob/9fc1a288fb831c87d93a4e2663bc30ccf9225b29/distributions.py#L46
        """
        term1 = (new_mu - old_mu) / old_sigma
        term2 = new_sigma / old_sigma
        loss = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
        return loss.sum()
    else:
        new_dist = Normal(new_mu, new_sigma)
        old_dist = Normal(old_mu, old_sigma)
        return kl_divergence(new_dist, old_dist).sum()

def compute_js_div(new_mu, new_sigma, old_mu, old_sigma, skew=False, alpha=0.8):
    if skew:
        avg_mu = alpha * old_mu + (1-alpha) * new_mu
        avg_var = alpha * old_sigma + (1-alpha) * new_sigma
    else:
        alpha = 0.5
        avg_mu = alpha * (new_mu + old_mu)
        avg_var = alpha * (new_sigma + old_sigma)

    rev_kl = (1-alpha)  * kl_divergence(Normal(new_mu, new_sigma), Normal(avg_mu, avg_var)).sum()
    kl = alpha * kl_divergence(Normal(old_mu, old_sigma), Normal(avg_mu, avg_var)).sum()

    return  kl + rev_kl

def kl_coeff(step, total_step, constant_step, min_kl_coeff, max_kl_val=1.0):
    # return max(min((step - constant_step) / total_step, max_kl_val), min_kl_coeff)
    return max(min((step - constant_step) / total_step, 0.1), min_kl_coeff)

# print(kl_coeff(10, 100, 0.001, 0.001, 0.1))
# def kl_balancer_coeff(num_scales, groups_per_scale, fun, device=None):
#     if fun == 'equal':
#         coeff = [1 for i in range(num_scales)]
#     elif fun == 'linear':
#         coeff = [(2 ** i) * 1 for i in range(num_scales)]
#     elif fun == 'sqrt':
#         coeff = [np.sqrt(2 ** i) * 1 for i in range(num_scales)]
#     elif fun == 'square':
#         coeff = [np.square(2 ** i) / 1 * 1 for i in range(num_scales)]
#     else:
#         raise NotImplementedError
#     # convert min to 1.
#     coeff /= np.min(coeff)
#     return coeff
def kl_balancer_coeff(num_scales, groups_per_scale, fun):
    if fun == 'equal':
        coeff = torch.cat([torch.ones(groups_per_scale[num_scales - i - 1]) for i in range(num_scales)], dim=0).cuda()
    elif fun == 'linear':
        coeff = torch.cat([(2 ** i) * torch.ones(groups_per_scale[num_scales - i - 1]) for i in range(num_scales)], dim=0).cuda()
    elif fun == 'sqrt':
        coeff = torch.cat([np.sqrt(2 ** i) * torch.ones(groups_per_scale[num_scales - i - 1]) for i in range(num_scales)], dim=0).cuda()
    elif fun == 'square':
        coeff = torch.cat([np.square(2 ** i) / groups_per_scale[num_scales - i - 1] * torch.ones(groups_per_scale[num_scales - i - 1]) for i in range(num_scales)], dim=0).cuda()
    else:
        raise NotImplementedError
    # convert min to 1.
    coeff /= torch.min(coeff)
    return coeff

def groups_per_scale(num_scales, num_groups_per_scale, is_adaptive, divider=2, minimum_groups=1):
    g = []
    n = num_groups_per_scale
    for s in range(num_scales):
        assert n >= 1
        g.append(n)
        if is_adaptive:
            n = n // divider
            n = max(minimum_groups, n)
    return g

# groups = groups_per_scale(1, 10, False)
# print(groups)
# print(kl_balancer_coeff(1, groups, 'equal'))
# print(kl_balancer_coeff(1, groups, 'linear'))
# print(kl_balancer_coeff(1, groups, 'sqrt'))
# print(kl_balancer_coeff(1, groups, 'square'))