import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from backbone import MammothBackbone, xavier, num_flat_features
import math
import numpy as np
from backbone.utils.MLP import MLP
from backbone.utils.attention_modules import MAB as Attention

class NP_HEAD(nn.Module):
    def __init__(self,input_dim,
                 latent_dim,
                 num_classes,
                 n_tasks,
                 cls_per_task,
                 num_layers=2,
                 xavier_init=False,
                 num_attn_heads=4,
                 label_embedder=False,
                 task_to_classes_map=None,
                 use_deterministic=True
                 ):
        super(NP_HEAD, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.latent_encoder = None
        self.num_layers = num_layers
        self.decoder_input_dim = input_dim + latent_dim*2
        if not use_deterministic:
            self.decoder_input_dim -= latent_dim
        self.xavier_init = xavier_init
        self.n_tasks = n_tasks
        self.cls_per_task = cls_per_task

    def forward(self, *args, **kwargs) -> tuple:
        raise NotImplementedError