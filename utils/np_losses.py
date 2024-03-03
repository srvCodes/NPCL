import torch
import torch.nn.functional as F

def ce_loss_np(logits, targets_onehot):
    assert targets_onehot.dim() == 2, "Targets must be one hot encoded"
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
        sample_T=1
    else:
        sample_T= logits.size(0)
    pred = F.softmax(logits, dim=-1)

    B = pred.size(1)
    targets_onehot_expand = targets_onehot.unsqueeze(0).expand(sample_T, -1, -1)
    loss = torch.sum(-targets_onehot_expand * pred.log())

    return loss / (B * sample_T)

# def linear_schedule_rate(step_num, max_val, warmup_step=4000, optimizer=None):
#     # lr = 0.0001 * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
#     rate = min(float(step_num) / float(max(1, warmup_step)) *  max_val, max_val)
#     if optimizer is not None:
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = rate
#     else:
#         return rate

def linear_schedule_rate(initial_val, max_val, cur_step, total_steps):
    rate = min(initial_val + cur_step * (max_val - initial_val) / total_steps, max_val)
    return rate

# for i in range(200):
#     print(i, linear_schedule_rate(0.00005, 0.1, i, 100))
#     from distributions import kl_coeff
#     print(i, kl_coeff(i, 100,
#                               0.00005 * 200, 0.00005))