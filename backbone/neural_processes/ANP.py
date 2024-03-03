import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from backbone import xavier, num_flat_features
import math
import numpy as np
from backbone.utils.MLP import MLP, LatentMLP
from backbone.utils.attention_modules import MAB as Attention
from backbone.neural_processes.NP_Head_Base import NP_HEAD

class DetEncoder(nn.Module):
    def __init__(self,  input_dim, num_classes, latent_dim, num_layers=2, num_attn_heads=1, label_embedder=False, xavier_init=False):
        super(DetEncoder, self).__init__()
        set_encoding_dim = input_dim + num_classes if not label_embedder else input_dim
        self.set_encoder = MLP(set_encoding_dim, latent_dim, latent_dim, num_layers, xavier_init=xavier_init)
        self.attentions = nn.ModuleList([Attention(latent_dim, latent_dim, latent_dim, num_attn_heads, xavier_init=xavier_init) for _ in range(num_layers)])
        self.context_projection = MLP(input_dim, latent_dim, latent_dim, 1, xavier_init=xavier_init)
        self.target_projection = MLP(input_dim, latent_dim, latent_dim, 1, xavier_init=xavier_init)
        self.cross_attentions = nn.ModuleList([Attention(latent_dim, latent_dim, latent_dim, num_attn_heads, xavier_init=xavier_init) for _ in range(num_layers)])
        self.label_embedder = label_embedder
        if label_embedder:
            self.label_emb = nn.Embedding(num_classes, input_dim)

    def forward(self, x, y, x_target):
        if self.label_embedder:
            d = x + self.label_emb(y)
        else:
            d = torch.cat((x, y), -1)
        s = self.set_encoder(d)
        # add attention here optionally
        for attention in self.attentions:
            s = attention(s, s, s)
        x = self.context_projection(x)
        x_target = self.target_projection(x_target)
        for attention in self.cross_attentions:
            x_target = attention(x, s, x_target)
        return x_target

class LatentEncoder(nn.Module):
    def __init__(self, input_dim,  num_classes, latent_dim, num_layers=2, num_attn_heads=1, label_embedder=False, xavier_init=False):
        super(LatentEncoder, self).__init__()
        set_encoding_dim = input_dim + num_classes if not label_embedder else input_dim
        self.set_encoder = MLP(set_encoding_dim, latent_dim, latent_dim, num_layers, xavier_init=xavier_init)
        self.attentions = nn.ModuleList([Attention(latent_dim, latent_dim, latent_dim, num_attn_heads, xavier_init=xavier_init) for _ in range(num_layers)])
        self.label_embedder = label_embedder
        if label_embedder:
            self.label_emb = nn.Embedding(num_classes, input_dim)
        self.global_amortizer = LatentMLP(latent_dim, latent_dim, latent_dim,
                                        num_layers, xavier_init=xavier_init)
    def forward(self, x, y):
        if self.label_embedder:
            d = x + self.label_emb(y)
        else:
            d = torch.cat((x, y), -1)
        s = self.set_encoder(d)
        for attention in self.attentions:
            s = attention(s, s, s)
        q =self.global_amortizer(s.mean(0))
        return q

class Decoder(nn.Module):
    def __init__(self,  decoder_input_dim, latent_dim, num_layers=2, xavier_init=False):
        super(Decoder, self).__init__()
        self.decoder = MLP(decoder_input_dim, latent_dim, latent_dim, num_layers, xavier_init=xavier_init)

    def forward(self, x_in, r_det=None, z_lat=None):
        decoder_in = torch.cat((x_in, r_det), -1) if r_det is not None else x_in
        if z_lat is not None:
            decoder_in = torch.cat((decoder_in, z_lat), -1)
        decoder_out = self.decoder(decoder_in)
        return decoder_out

class ANP_HEAD(NP_HEAD):
    def __init__(self, input_dim,
                 latent_dim,
                 num_classes,
                 n_tasks,
                 cls_per_task,
                 num_layers=2,
                 xavier_init=False,
                 num_attn_heads=4,
                 label_embedder=False,
                 context_task_labels=None,
                 target_task_labels=None,
                 test_oracle=False,
                 use_deterministic=True,
                 hierarchy=False
                 ):
        super().__init__(input_dim, latent_dim, num_classes, n_tasks, cls_per_task, num_layers, xavier_init)

        self.det_encoder = DetEncoder(input_dim, num_classes, latent_dim, num_layers=num_layers,
                                      num_attn_heads=num_attn_heads, label_embedder=label_embedder,
                                      xavier_init=xavier_init) if use_deterministic else None
        self.latent_encoder = LatentEncoder(input_dim, num_classes, latent_dim, num_layers=num_layers,
                                            num_attn_heads=num_attn_heads, label_embedder=label_embedder,
                                            xavier_init=xavier_init)
        self.fc_decoder = Decoder(self.decoder_input_dim, input_dim, num_layers=num_layers, xavier_init=xavier_init)
        self.classifier = nn.Linear(input_dim, num_classes, bias=True)
        self.use_deterministic = use_deterministic
        if xavier_init:
            self.net = nn.Sequential(self.classifier)
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self.net.apply(xavier)

    def forward(self, x_context_in, labels_context_in, x_target_in, labels_target_in=None,
                phase_train=True, forward_times=1, epoch=None, cur_test_task=None,
                clnp_stochasticity='all_global', context_task_labels=None, target_task_labels=None,
                task_to_labels=None, top_k_decode=1, x_percent=10, prev_task_distr = None) -> tuple:
        B = x_target_in.size(0)
        context_representation_deterministic_expand = None
        if phase_train:
            q_target = self.latent_encoder(x_target_in, labels_target_in)
            q_context = self.latent_encoder(x_context_in, labels_context_in)
            latent_z_target = None
            for i in range(0, forward_times):
                z =Normal(q_target[0], q_target[1]).rsample()
                z = z.unsqueeze(0)
                if i == 0:
                    latent_z_target = z
                else:
                    latent_z_target = torch.cat((latent_z_target, z))
            latent_z_target_expand = latent_z_target.unsqueeze(1).expand(-1, B, -1)
            x_target_in_expand = x_target_in.unsqueeze(0).expand(forward_times, -1, -1)
            if self.use_deterministic:
                x_representation_deterministic = self.det_encoder(x_context_in, labels_context_in, x_target_in)
                context_representation_deterministic_expand = x_representation_deterministic.unsqueeze(0).expand(
                    forward_times, -1, -1)

            ################## decoder ####
            ##############
            output_function = self.fc_decoder(x_target_in_expand, context_representation_deterministic_expand,
                                              latent_z_target_expand)
            output = self.classifier(output_function)
            return output, ((q_context, None), (q_target, None)), None
        else:
            q = self.latent_encoder(x_context_in, labels_context_in)
            latent_z_target = None

            for i in range(0, forward_times):
                z = Normal(q[0], q[1]).rsample()
                z = z.unsqueeze(0)
                if i == 0:
                    latent_z_target = z
                else:
                    latent_z_target = torch.cat((latent_z_target, z))
            latent_z_target_expand = latent_z_target.unsqueeze(1).expand(-1, B, -1)

            x_target_in_expand = x_target_in.unsqueeze(0).expand(forward_times, -1, -1)

            if self.use_deterministic:
                x_representation_deterministic = self.det_encoder(x_context_in, labels_context_in, x_target_in)
                context_representation_deterministic_expand = x_representation_deterministic.unsqueeze(0).expand(
                    forward_times, -1, -1)

            output_function = self.fc_decoder(x_target_in_expand, context_representation_deterministic_expand,
                                              latent_z_target_expand)
            output = self.classifier(output_function)

            return output, None


