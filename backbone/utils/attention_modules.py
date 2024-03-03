import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from backbone import xavier

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, xavier_init=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

        if xavier_init:
            self.net = nn.Sequential(self.fc_q, self.fc_k, self.fc_v, self.fc_o)
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self.net.apply(xavier)

    def forward(self, K, V, Q):
        two_dim=False
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(V)
        if K.dim() == 2:
            two_dim = True
            K, V, Q = K.unsqueeze(0),  V.unsqueeze(0), Q.unsqueeze(0)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2) # attention values
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O) # LayerNorm(X + Multihead(X, Y, Y ; ω)) where X = Q
        O = O + F.relu(self.fc_o(O)) # H + rFF(H)
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        if two_dim:
            O = O.squeeze(0)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(X, X, self.I.repeat(X.size(0), 1, 1))
        return self.mab1(H, H, X)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X, self.S.repeat(X.size(0), 1, 1))


class SetTransformer(nn.Module):
    """
    # what doesnt help much: (a) removing SAB layers from decoder (b)
    # what does help is:  (a) reducing I from 32 to 16 and then further to 8
    """
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=8, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        two_dim = False
        if X.dim() == 2:
            two_dim = True
            X = X.unsqueeze(0)
        outs = self.dec(self.enc(X))
        if two_dim:
            outs = outs.squeeze(0)
        return outs