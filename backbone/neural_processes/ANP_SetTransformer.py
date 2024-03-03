import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from backbone import xavier, num_flat_features
import math
import numpy as np
from backbone.utils.MLP import MLP
from backbone.utils.attention_modules import MAB as Attention
from backbone.neural_processes.NP_Head_Base import NP_HEAD
from backbone.utils.attention_modules import SetTransformer


class DetEncoder(nn.Module):
    def __init__(self,  input_dim, num_classes, latent_dim, num_layers=2, num_attn_heads=1, label_embedder=False):
        super(DetEncoder, self).__init__()
        set_encoding_dim = input_dim + num_classes if not label_embedder else input_dim
        self.set_encoder = MLP(set_encoding_dim, latent_dim, latent_dim, num_layers)
        self.attentions = nn.ModuleList([Attention(latent_dim, latent_dim, latent_dim, num_attn_heads) for _ in range(num_layers)])
        self.context_projection = MLP(input_dim, latent_dim, latent_dim, 1)
        self.target_projection = MLP(input_dim, latent_dim, latent_dim, 1)
        self.cross_attentions = nn.ModuleList([Attention(latent_dim, latent_dim, latent_dim, num_attn_heads) for _ in range(num_layers)])
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

class DetEncoder_v0(nn.Module):
    def __init__(self,  input_dim, num_classes, latent_dim, num_layers=2, num_attn_heads=1, label_embedder=False, seeds=1):
        super(DetEncoder_v0, self).__init__()
        set_encoding_dim = input_dim + num_classes if not label_embedder else input_dim
        self.label_encoder = MLP(set_encoding_dim, latent_dim, latent_dim, num_layers)
        self.set_encoder = SetTransformer(latent_dim, seeds, latent_dim, num_heads=num_attn_heads)


    def forward(self, x, y, x_target):
        d = torch.cat((x, y), -1)
        d = self.label_encoder(d)
        s = self.set_encoder(d)
        return s.mean(0)

class LatentEncoder(nn.Module):
    def __init__(self, input_dim,  num_classes, latent_dim, num_layers=2, num_attn_heads=1, label_embedder=False, seeds=1):
        super(LatentEncoder, self).__init__()
        set_encoding_dim = input_dim + num_classes if not label_embedder else input_dim
        self.label_encoder = MLP(set_encoding_dim, latent_dim, latent_dim, num_layers)
        self.set_encoder = SetTransformer(latent_dim, seeds, latent_dim, num_heads=num_attn_heads)

    def forward(self, x, y):
        d = torch.cat((x, y), -1)
        d = self.label_encoder(d)
        s = self.set_encoder(d)
        return s.mean(0)

class Decoder(nn.Module):
    def __init__(self,  decoder_input_dim, latent_dim, num_layers=2):
        super(Decoder, self).__init__()
        self.decoder = MLP(decoder_input_dim, latent_dim, latent_dim, num_layers)

    def forward(self, x_in, r_det, z_lat=None):
        decoder_in = torch.cat((x_in, r_det), -1)
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
                 det_set_transformer=False,
                 decoder_seeds=1):
        super().__init__(input_dim, latent_dim, num_classes, n_tasks, cls_per_task, num_layers, xavier_init)
        self.det_set_transformer = det_set_transformer
        if det_set_transformer:
            self.det_encoder = DetEncoder_v0(input_dim, num_classes, latent_dim, num_layers=num_layers,
                                             num_attn_heads=num_attn_heads,
                                             label_embedder=label_embedder,
                                             seeds=decoder_seeds)
        else:
            self.det_encoder = DetEncoder(input_dim, num_classes, latent_dim, num_layers=num_layers, num_attn_heads=num_attn_heads, label_embedder=label_embedder)
        self.latent_encoder = LatentEncoder(input_dim, num_classes, latent_dim, num_layers=num_layers,
                                            num_attn_heads=num_attn_heads,
                                            label_embedder=label_embedder,
                                            seeds=decoder_seeds)
        self.mean_net = nn.Linear(latent_dim, latent_dim)
        self.log_var_net = nn.Linear(latent_dim, latent_dim)

        self.fc_decoder = Decoder(self.decoder_input_dim, input_dim, num_layers=num_layers)
        self.classifier = nn.Linear(input_dim, num_classes, bias=True)


    def forward(self, x_context_in, labels_context_in, x_target_in, labels_target_in =None,
                    phase_train=True, forward_times=1, epoch=None, cur_task=None, task_labels=None) -> tuple:
        B = x_target_in.size(0)

        if phase_train:
            x_representation_deterministic = self.det_encoder(x_context_in, labels_context_in, x_target_in)
            x_representation_latent = self.latent_encoder(x_target_in, labels_target_in)
            x_representation_latent_context = self.latent_encoder(x_context_in, labels_context_in)

            mean_context_x = self.mean_net(x_representation_latent_context)
            log_var_context_x = self.log_var_net(x_representation_latent_context)
            sigma_context_x = 0.1 + 0.9 * F.softplus(log_var_context_x)
            q_context = Normal(mean_context_x, sigma_context_x)

            mean_x = self.mean_net(x_representation_latent)
            log_var_x = self.log_var_net(x_representation_latent)
            sigma_x = 0.1 + 0.9 * F.softplus(log_var_x)
            q_target = Normal(mean_x, sigma_x)

            latent_z_target = None
            for i in range(0, forward_times):
                z = q_target.rsample()
                z = z.unsqueeze(0)
                if i == 0:
                    latent_z_target = z
                else:
                    latent_z_target = torch.cat((latent_z_target, z))
            latent_z_target_expand = latent_z_target.unsqueeze(1).expand(-1, B, -1)



            x_target_in_expand = x_target_in.unsqueeze(0).expand(forward_times, -1, -1)
            if x_representation_deterministic.size(0) == x_target_in.size(0):
                context_representation_deterministic_expand = x_representation_deterministic.unsqueeze(0).expand(
                    forward_times, -1, -1)
            else:
                context_representation_deterministic_expand = x_representation_deterministic.unsqueeze(0).unsqueeze(1).expand(
                    forward_times, B, -1)
            ################## decoder ####
            ##############
            output_function = self.fc_decoder(x_target_in_expand, context_representation_deterministic_expand,
                                              latent_z_target_expand)
            output = self.classifier(output_function)
            return output, (q_context, q_target)
        else:
            latent_z_target_expand = None
            x_representation_latent = self.latent_encoder(x_context_in, labels_context_in)
            mean = self.mean_net(x_representation_latent)
            log_var = self.log_var_net(x_representation_latent)
            sigma = 0.1 + 0.9 * F.softplus(log_var)
            q = Normal(mean, sigma)
            latent_z_target = None

            for i in range(0, forward_times):
                z = q.rsample()
                z = z.unsqueeze(0)
                if i == 0:
                    latent_z_target = z
                else:
                    latent_z_target = torch.cat((latent_z_target, z))
            latent_z_target_expand = latent_z_target.unsqueeze(1).expand(-1, B, -1)
            x_target_in_expand = x_target_in.unsqueeze(0).expand(forward_times, -1, -1)

            x_representation_deterministic = self.det_encoder(x_context_in, labels_context_in, x_target_in)
            if x_representation_deterministic.size(0) == x_target_in.size(0):
                context_representation_deterministic_expand = x_representation_deterministic.unsqueeze(0).expand(
                    forward_times, -1, -1)
            else:
                context_representation_deterministic_expand = x_representation_deterministic.unsqueeze(0).unsqueeze(1).expand(
                    forward_times, B, -1)

            output_function = self.fc_decoder(x_target_in_expand, context_representation_deterministic_expand,
                                              latent_z_target_expand)
            output = self.classifier(output_function)

            return output, None


