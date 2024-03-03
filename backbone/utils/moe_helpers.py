import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import xavier


class Softmax_Net(nn.Module):
    def __init__(self,
                 dim_xz,
                 experts_in_gates,
                 dim_logit_h,
                 num_logit_layers,
                 num_experts,
                 xavier_init=False,
                 ):
        super().__init__()
        self.dim_xz = dim_xz
        self.experts_in_gates = experts_in_gates
        self.dim_logit_h = dim_logit_h
        self.num_logit_layers = num_logit_layers
        self.num_experts = num_experts

        self.logit_modules = []
        if self.experts_in_gates:
            self.logit_modules.append(nn.Linear(self.dim_xz, self.dim_logit_h))
            for i in range(self.num_logit_layers):
                self.logit_modules.append(nn.ReLU())
                self.logit_modules.append(nn.Linear(self.dim_logit_h, self.dim_logit_h))
            self.logit_modules.append(nn.ReLU())
            self.logit_modules.append(nn.Linear(self.dim_logit_h, 1))
        else:
            self.logit_modules.append(nn.Linear(self.dim_xz, self.dim_logit_h))
            for i in range(self.num_logit_layers):
                self.logit_modules.append(nn.ReLU())
                self.logit_modules.append(nn.Linear(self.dim_logit_h, self.dim_logit_h))
            self.logit_modules.append(nn.ReLU())
            self.logit_modules.append(nn.Linear(self.dim_logit_h, self.num_experts))
        self.logit_net = nn.Sequential(*self.logit_modules)
        if xavier_init:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        pass
        # self.logit_net.apply(xavier)

    def forward(self, x_z, temperature=1, gumbel_max=False, return_label=False):
        if self.experts_in_gates:
            logit_output = self.logit_net(x_z) # e.g., [32, 3, 640] -->> [32, 3, 1]
        else:
            logit_output = self.logit_net(x_z) # e.g. [32, 512] -->> [32, 3]
            logit_output = logit_output.unsqueeze(-1) # e.g., [32, 3] --> [32, 3, 1]

        if gumbel_max:
            logit_output = logit_output + sample_gumbel(logit_output.size(), device=logit_output.device)

        softmax_y = F.softmax(logit_output / temperature, dim=-2)
        softmax_y = softmax_y.squeeze(-1)
        shape = softmax_y.size()
        _, ind = softmax_y.max(dim=-1)
        y_hard = torch.zeros_like(softmax_y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - softmax_y).detach() + softmax_y
        softmax_y, y_hard = softmax_y.unsqueeze(-1), y_hard.unsqueeze(-1)
        return softmax_y, y_hard


def sample_gumbel(shape, eps=1e-20, device=None):
    U = torch.rand(shape)
    if device:
        U = U.to(device)

    return -torch.log(-torch.log(U + eps) + eps)

def get_moe_outputs(outputs, alpha, y_hard, hard=True, training=True, task_dist_ids=None):
    if task_dist_ids is not None:
        alpha = torch.index_select(alpha, 1, torch.tensor(task_dist_ids).to(alpha.device))

    assert outputs.size(0) == alpha.size(
        1), f"Either the shapes of alpha and outputs should match, or enter the valid task ids to filter alpha from! Outputs: {outputs.shape}, alpha: {alpha.shape}"
    outputs = outputs.permute(2, 1, 0, 3)
    # make y_hard from available alpha as Y_hard might contain 1s for future unseen task heads too
    alpha_max_idcs = torch.argmax(alpha, dim=1, keepdim=True)
    alpha_one_hot = torch.zeros_like(alpha).scatter_(1, alpha_max_idcs, 1.)
    # alpha_one_hot = alpha_one_hot.unsqueeze(1).expand(-1, outputs.size(1), -1, -1) # (40, 5, num_experts, 1)
    alpha = alpha.unsqueeze(1).expand(-1, outputs.size(1), -1, -1) # (40, 5, num_experts, 1)
    weighted_y_pred = None
    if hard == False:
        # try:
        weighted_y_pred = torch.mul(outputs, alpha)
        weighted_y_pred = torch.sum(weighted_y_pred, dim=-2)
        # except RuntimeError as e:
        #     print(e)
    else:
        raise NotImplementedError("Not implemented for moe hard = False !")

    return weighted_y_pred.permute(1, 0, 2)

if __name__ == '__main__':
    snet = Softmax_Net(128, True, 64, 2, 3).cuda()
    l = torch.rand((32, 128)).cuda()
    out = snet(l, gumbel_max=True)
    print(out[0].shape, out[1].shape)
    out = snet(l, gumbel_max=False)
    print(out[0].shape, out[1].shape); exit(1)