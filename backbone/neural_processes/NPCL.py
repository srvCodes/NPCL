import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from backbone import xavier, num_flat_features
from backbone.utils.MLP import MLP, LatentMLP
from backbone.utils.attention_modules import MAB as Attention
from backbone.neural_processes.NP_Head_Base import NP_HEAD
from utils.uncertainty_quantifiers import compute_shannon_entropy
import numpy as np

class DetEncoder(nn.Module):
    def __init__(self, input_dim, num_classes, latent_dim, num_layers=2, num_attn_heads=1, label_embedder=False,
                 xavier_init=False, num_tasks=None):
        super(DetEncoder, self).__init__()
        set_encoding_dim = input_dim + num_classes if not label_embedder else input_dim
        self.set_encoder = MLP(set_encoding_dim, latent_dim, latent_dim, num_layers, xavier_init=xavier_init)
        # self.per_task_attentions = nn.ModuleList([nn.ModuleList(
        #     [Attention(latent_dim, latent_dim, latent_dim, num_attn_heads, xavier_init=xavier_init) for _ in
        #      range(num_layers)]) for _ in range(num_tasks)])
        # self.cross_task_attentions = nn.ModuleList(
        #     [Attention(latent_dim, latent_dim, latent_dim, num_attn_heads, xavier_init=xavier_init) for _ in
        #      range(num_layers)])
        self.context_projection = MLP(input_dim, latent_dim, latent_dim, 1, xavier_init=xavier_init)
        self.target_projection = MLP(input_dim, latent_dim, latent_dim, 1, xavier_init=xavier_init)
        self.cross_attentions = nn.ModuleList(
            [Attention(latent_dim, latent_dim, latent_dim, num_attn_heads, xavier_init=xavier_init) for _ in
             range(num_layers)])
        self.label_embedder = label_embedder
        if label_embedder:
            self.label_emb = nn.Embedding(num_classes, input_dim)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x, y, x_target, task_labels=None):
        if self.label_embedder:
            d = x + self.label_emb(y)
        else:
            d = torch.cat((x, y), -1)
        s = self.set_encoder(d)
        # temp = torch.zeros_like(s)
        # # per task attention
        # for label in torch.unique(task_labels):
        #     idcs = (task_labels == label).nonzero(as_tuple=True)[0]
        #     s_local = s[idcs]
        #     for attention in self.per_task_attentions[label]:
        #         s_local = attention(s_local, s_local, s_local)
        #     temp[idcs] = s_local
        # s = temp
        # add attention here optionally
        # for attention in self.cross_task_attentions:
        #     s = attention(s, s, s)
        x = self.context_projection(x)
        x = self.dropout(x)
        x_target = self.target_projection(x_target)
        x_target = self.dropout(x_target)
        for attention in self.cross_attentions:
            x_target = attention(x, s, x_target)
        return x_target


class TaskEncoder(nn.Module):
    def __init__(self, dim_hidden, hierarchical=True, num_layers=2, xavier_init=False):
        super().__init__()
        self.hierarchical = hierarchical
        self.task_amortizer = LatentMLP(dim_hidden * (1 + int(hierarchical)), dim_hidden, dim_hidden,
                                        num_layers, xavier_init=xavier_init)

    def forward(self, s, z=None):
        # hierarchical conditioning
        if self.hierarchical:
            assert z is not None
            s = torch.cat((s, z), -1)

        # task latent distribution
        q_T = self.task_amortizer(s)
        return q_T


class LatentEncoder(nn.Module):
    def __init__(self, input_dim, num_classes, latent_dim, num_layers=2, num_attn_heads=1, label_embedder=False,
                 xavier_init=False, num_tasks=None, hierarchy=True):
        super(LatentEncoder, self).__init__()
        self.set_encoder = MLP(input_dim + num_classes, latent_dim, latent_dim, num_layers, xavier_init=xavier_init)
        self.hierarchy = hierarchy
        # self.task_embeddings = nn.Embedding(num_tasks, latent_dim)
        self.per_task_attentions = nn.ModuleList([nn.ModuleList(
            [Attention(latent_dim, latent_dim, latent_dim, num_attn_heads, xavier_init=xavier_init) for _ in
             range(num_layers)]) for _ in range(num_tasks)])
        self.attentions = nn.ModuleList(
            [Attention(latent_dim, latent_dim, latent_dim, num_attn_heads, xavier_init=xavier_init) for _ in
             range(num_layers)])
        if self.hierarchy:
            self.global_amortizer = LatentMLP(latent_dim, latent_dim, latent_dim,
                                              num_layers, xavier_init=xavier_init)

    def forward(self, x, y, task_labels=None):
        d = torch.cat((x, y), -1)
        s = self.set_encoder(d)
        # task_labels_embedded = self.task_embeddings(task_labels)
        # s = s + task_labels_embedded
        temp = torch.zeros_like(s)
        # per task attention
        for label in torch.unique(task_labels):
            idcs = (task_labels == label).nonzero(as_tuple=True)[0]
            s_local = s[idcs]
            for attention in self.per_task_attentions[label]:
                s_local = attention(s_local, s_local, s_local)
            temp[idcs] = s_local
        s_local = temp
        # across task attention
        for attention in self.attentions:
            temp = attention(temp, temp, temp)

        q_target = self.global_amortizer(temp.mean(0)) if self.hierarchy else None
        return q_target, s_local, temp


class Decoder(nn.Module):
    def __init__(self, decoder_input_dim, latent_dim, num_layers=2, xavier_init=False):
        super(Decoder, self).__init__()
        self.decoder = MLP(decoder_input_dim, latent_dim, latent_dim, num_layers, xavier_init=xavier_init)

    def forward(self, x_in, r_det, vs=None, task_labels=None):
        decoder_in = torch.cat((x_in, r_det), -1)
        if task_labels is not None:
            decoder_in_temp = decoder_in.clone().transpose(0, 1)
            assert x_in.size(1) == task_labels.size(0) and decoder_in_temp.size(0) == task_labels.size(
                0), "Task labels size mismatch in decoder!"
            decoder_in = torch.stack(
                [torch.cat((x, vs[task_labels[idx].item()]), -1) for idx, x in enumerate(decoder_in_temp)]).transpose(0,
                                                                                                                      1)
        else:
            # no labels during test time
            # so concatenate all available z to each test sample
            vs = torch.stack(
                list(vs.values()))  # each value in v is repeated forward_times and v has 't' such task values
            vs_expand = vs.unsqueeze(2).expand(-1, -1, x_in.size(-2), -1)
            if x_in.dim() == vs_expand.dim() - 1:
                x_in = x_in.unsqueeze(0).expand(len(vs), -1, -1, -1)
            if r_det.dim() == vs_expand.dim() - 1:
                r_det = r_det.unsqueeze(0).expand(len(vs), -1, -1, -1)
            decoder_in = torch.cat((x_in, r_det), -1)
            decoder_in = torch.cat((decoder_in, vs_expand), -1)
        decoder_out = self.decoder(decoder_in)
        return decoder_out


class NPCL(NP_HEAD):
    def __init__(self, input_dim,
                 latent_dim,
                 num_classes,
                 n_tasks,
                 cls_per_task,
                 num_layers=2,
                 xavier_init=False,
                 num_attn_heads=4,
                 label_embedder=False,
                 task_to_classes_map=None,
                 test_oracle=False,
                 hierarchy=True):
        super().__init__(input_dim, latent_dim, num_classes, n_tasks, cls_per_task, num_layers, xavier_init)
        self.task_to_classes_map = task_to_classes_map
        self.test_oracle = test_oracle
        self.hierarchy = hierarchy
        self.det_encoder = DetEncoder(input_dim, num_classes, latent_dim, num_layers=num_layers, \
                                      num_attn_heads=num_attn_heads, label_embedder=label_embedder,
                                      xavier_init=xavier_init, num_tasks=self.n_tasks)
        self.latent_encoder = LatentEncoder(input_dim, num_classes, latent_dim, num_layers=num_layers,
                                            num_attn_heads=num_attn_heads, label_embedder=label_embedder,
                                            xavier_init=xavier_init, num_tasks=self.n_tasks, hierarchy=hierarchy)
        self.task_encoder = nn.ModuleList(
            [TaskEncoder(latent_dim, hierarchical=hierarchy, num_layers=num_layers, xavier_init=xavier_init) for task in
             range(self.n_tasks)])

        self.fc_decoder = Decoder(self.decoder_input_dim, input_dim, num_layers=num_layers, xavier_init=xavier_init)
        self.classifier = nn.Linear(input_dim, num_classes, bias=True)
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
        if phase_train:
            x_representation_deterministic = self.det_encoder(x_context_in, labels_context_in, x_target_in, context_task_labels)
            q_target, s_D, _ = self.latent_encoder(x_target_in, labels_target_in, target_task_labels)
            q_context, s_C, _ = self.latent_encoder(x_context_in, labels_context_in, context_task_labels)
            latent_z_target = None
            vs, q_target_taskwise = self.get_local_latents(q_target, clnp_stochasticity,
                                                           forward_times, s_D,
                                                           target_task_labels,
                                                           phase_train, context=False,
                                                           task_to_labels=task_to_labels,
                                                           hierarchy=self.hierarchy)
            _, q_context_taskwise = self.get_local_latents(q_context, clnp_stochasticity,
                                                           forward_times, s_C,
                                                           context_task_labels,
                                                           phase_train, context=True,
                                                           task_to_labels=task_to_labels,
                                                           hierarchy=self.hierarchy)

            x_target_in_expand = x_target_in.unsqueeze(0).expand(forward_times, -1, -1)
            x_representation_deterministic_expand = x_representation_deterministic.unsqueeze(0).expand(
                forward_times, -1, -1)

            ################## decoder ####
            ##############
            output_function = self.fc_decoder(x_target_in_expand, x_representation_deterministic_expand,
                                              vs, target_task_labels)
            output = self.classifier(output_function)
            return output, ((q_context, q_context_taskwise), (q_target, q_target_taskwise)), (None, None)
        else:
            q_context, s_C, _ = self.latent_encoder(x_context_in, labels_context_in, context_task_labels)
            vs, _ = self.get_local_latents(q_context, clnp_stochasticity,
                                                            forward_times, s_C,
                                                            context_task_labels,
                                                            phase_train, context=True,
                                                            task_to_labels=task_to_labels,
                                                            hierarchy=self.hierarchy,)
#                                                            prev_task_dist=prev_task_distr)

            x_target_in_expand = x_target_in.unsqueeze(0).unsqueeze(1).expand(len(vs), forward_times, -1, -1)
            x_representation_deterministic = self.det_encoder(x_context_in, labels_context_in, x_target_in, context_task_labels)
            x_representation_deterministic_expand = x_representation_deterministic.unsqueeze(0).unsqueeze(1) \
                .expand(len(vs), forward_times, -1, -1)

            output_function = self.fc_decoder(x_target_in_expand, x_representation_deterministic_expand,
                                              vs)
            output = self.classifier(output_function)
            decoded_output = self.decode_outputs(output, len(vs), cur_test_task, x_target_in.size(0), voting=False, top_k_decode=top_k_decode, x_percent=x_percent)

            return decoded_output, output

    def decode_outputs(self, output, num_tasks, cur_test_task, x_target_in_len, voting=False, top_k_decode=1, x_percent=10):
        if self.test_oracle:
            try:
                outputs = torch.stack([output[cur_test_task, :, sample_idx, :]
                                       for sample_idx in range(x_target_in_len)
                                       ]).transpose(1, 0)
            except IndexError:
                outputs = output.mean(0)
        else:
            if num_tasks > 1:
                if voting:
                    outputs = decode_by_voting(output)

                else:
                    outputs, uncs = get_k_nearest_by_uncertainty(logits=output, k=top_k_decode)
                    # outputs, _ = get_k_nearest_by_energy_score(logits=output, k=top_k_decode, x_percent=x_percent)
                    # outputs, _ = get_k_nearest_by_energy_score(logits=output, k=top_k_decode)
                    # outputs = get_k_nearest_by_variance(output, k=top_k_decode, return_indices=False, metric='softmax', x_percent=x_percent)
                    # outputs =
                    # _, _, outputs = get_modulewise_num_unique_preds(output)
                    # _, _, outputs = get_modulewise_top_two_prob_diff(output, return_id=False)
                    # _, _, outputs = get_modulewise_top_prob_score(output, return_id=False)
            else:
                outputs = output.mean(0)
        return outputs

    def encode_task(self, s, zs=None, task_labels=None, task_to_labels=None):
        # task-specific latent in across-task inference of latent path
        assert s.size(0) == task_labels.size(
            0), "Task encoder err :: Size mismatch between task labels and set encoding!"
        q_T = {}
        unique_tasks = torch.unique(task_labels, sorted=True)
        for i, task_id in enumerate(unique_tasks):
            if task_id.item() not in task_to_labels:
                valid_labels = [i for i in
                                range(task_id * self.cls_per_task, (task_id * self.cls_per_task) + self.cls_per_task)]
                task_to_labels[task_id.item()] = valid_labels

            task_id = task_id.item()
            idcs = (task_labels == task_id).nonzero(as_tuple=True)[0]
            if len(idcs) > 0:
                s_t = s[idcs]
                s_t = s_t.mean(0)
                if zs is None:
                    q_T[task_id] = self.task_encoder[task_id](s_t)
                else:
                    z = zs[i] if zs.dim() == 2 else zs
                    q_T[task_id] = self.task_encoder[task_id](s_t, z)
        return q_T

    def get_local_latents(self, q, clnp_stochasticity, forward_times, s, task_labels, phase_train=False, context=False,
                          task_to_labels=None, hierarchy=False, prev_task_dist= None):
        vs = None
        q_taskwise = None
        if prev_task_dist is not None:
            q_taskwise = prev_task_dist
            clnp_stochasticity = 'all_local'

        if not hierarchy:
            clnp_stochasticity = 'all_local'
        if clnp_stochasticity == 'all_global':
            for i in range(0, forward_times):
                z = Normal(q[0], q[1]).rsample()
                q_taskwise = self.encode_task(s, z, task_labels, task_to_labels)
                if phase_train and context:
                    pass
                else:
                    if vs is None:
                        vs = {task_id: Normal(*stats).rsample().unsqueeze(0) for task_id, stats in
                              q_taskwise.items()}
                    else:
                        vs.update(
                            {task_id: torch.cat((vs[task_id], Normal(*stats).rsample().unsqueeze(0))) for task_id, stats
                             in
                             q_taskwise.items()})
        elif clnp_stochasticity == 'all_global_unique':
            for i in range(0, forward_times):
                zs = []
                global_dist = Normal(q[0], q[1])
                for _ in torch.unique(task_labels):
                    z = global_dist.rsample()
                    zs.append(z)
                zs = torch.stack(zs)
                q_taskwise = self.encode_task(s, zs, task_labels, task_to_labels)
                if phase_train and context:
                    pass
                else:
                    if vs is None:
                        vs = {task_id: Normal(*stats).rsample().unsqueeze(0) for task_id, stats in
                              q_taskwise.items()}
                    else:
                        vs.update(
                            {task_id: torch.cat((vs[task_id], Normal(*stats).rsample().unsqueeze(0))) for task_id, stats
                             in
                             q_taskwise.items()})
        elif clnp_stochasticity == 'all_local':
            if q_taskwise is None:
                z = None
                if hierarchy:
                    z = Normal(q[0], q[1]).rsample()

                q_taskwise = self.encode_task(s, z, task_labels, task_to_labels)
            if phase_train and context:
                pass
            else:
                for i in range(0, forward_times):
                    if vs is None:
                        vs = {task_id: Normal(*stats).rsample().unsqueeze(0) for task_id, stats in
                              q_taskwise.items()}
                    else:
                        vs.update(
                            {task_id: torch.cat((vs[task_id], Normal(*stats).rsample().unsqueeze(0))) for task_id, stats
                             in
                             q_taskwise.items()})
        elif clnp_stochasticity == 'mix':
            global_samples = forward_times // 2
            local_samples = forward_times - global_samples
            for i in range(0, global_samples):
                z = Normal(q[0], q[1]).rsample()
                q_taskwise = self.encode_task(s, z, task_labels, task_to_labels)
                if phase_train and context:
                    pass
                else:
                    for i in range(0, local_samples):
                        if vs is None:
                            vs = {task_id: Normal(*stats).rsample().unsqueeze(0) for task_id, stats in
                                  q_taskwise.items()}
                        else:
                            vs.update(
                                {task_id: torch.cat((vs[task_id], Normal(*stats).rsample().unsqueeze(0))) for
                                 task_id, stats
                                 in
                                 q_taskwise.items()})
        else:
            raise NotImplementedError
        return vs, q_taskwise

def decode_by_voting(output):
    min_entropy_indices = get_k_nearest_by_uncertainty(output, return_indices=True)
    min_var_entropy_indices = get_k_nearest_by_variance(logits=output, k=1, metric='entropy',
                                                        return_indices=True)
    min_var_softmax_indices = get_k_nearest_by_variance(logits=output, k=1, metric='softmax',
                                                        return_indices=True)
    max_prob_diff_indices, _, _ = get_modulewise_top_two_prob_diff(output, return_id=True)
    majority_voted_indices = torch.stack((min_var_entropy_indices, max_prob_diff_indices)).mode(0).values
    decoded_logits = []
    for sample_idx in range(output.size(2)):
        logit = output[majority_voted_indices[sample_idx], :, sample_idx, :]
        decoded_logits.append(logit)
    return torch.stack(decoded_logits).transpose(0, 1)


def get_k_nearest_by_uncertainty(logits, k=1, return_indices=False, x_percent=10):
    x_percent = 100
    k = min(k, logits.size(0))
    taskwise_uncertainties = []
    for task_id, _ in enumerate(logits):
        uncertainties = compute_shannon_entropy(logits[task_id])
        taskwise_uncertainties.append(uncertainties)
    samplewise_uncertainties = torch.stack(taskwise_uncertainties).transpose(0, 1)
    min_uncertainties = samplewise_uncertainties.topk(dim=1, largest=False, k=k)
    min_entropy_wts, min_entropy_idcs = min_uncertainties
    if return_indices:
        if k == 1:
            return min_entropy_idcs.squeeze(0).flatten()
        else:
            raise NotImplementedError("Return indices not implemented for k > 1.")
    decoded_logits = []
    for sample_idx in range(logits.size(2)):
        # print(min_variances_wts.shape, max_variance_wts.shape, sample_idx)
        # print(min_variances_wts[sample_idx, 0])
        # print(max_variance_wts[sample_idx, 0])
        min_var, max_var = min_entropy_idcs[sample_idx, 0], min_entropy_wts[sample_idx, 0]
        # x_percent = x_percent / 100
        this_logit = []
        for top_k in range(k):
            k_to_choose = top_k
            logit = logits[min_entropy_idcs[sample_idx, k_to_choose], :, sample_idx, :]
            if top_k == 0:
                this_logit.extend(logit)
            else:
                percentage_diff = check_percentage_in_range(min_entropy_idcs[sample_idx, k_to_choose], min_var, max_var)
                if percentage_diff * 100. < x_percent:
                    scale = percentage_diff
                    # logit = logit * (scale + 1.)
                    this_logit = torch.stack(this_logit) * 0.5 + logit * 0.5
                    this_logit.extend(logit)
        if type(this_logit) is list:
            this_logit = torch.stack(this_logit).squeeze(0)
        decoded_logits.append(this_logit)
    return torch.stack(decoded_logits).transpose(0, 1),  samplewise_uncertainties

def  get_k_nearest_by_energy_score(logits, k, x_percent=10):
    x_percent = 15
    logits_energy = -torch.logsumexp(logits, dim=-1).mean(1)
    samplewise_energies = logits_energy.permute(1, 0)
    min_energies = samplewise_energies.topk(dim=1, largest=False, k=k)
    min_energy_wts, min_energy_idcs = min_energies
    decoded_logits = []
    for sample_idx in range(logits.size(2)):
        # print(min_variances_wts.shape, max_variance_wts.shape, sample_idx)
        # print(min_variances_wts[sample_idx, 0])
        # print(max_variance_wts[sample_idx, 0])
        min_var, max_var = min_energy_idcs[sample_idx, 0], min_energy_wts[sample_idx, 0]
        # x_percent = x_percent / 100
        this_logit = []
        for top_k in range(k):
            k_to_choose = top_k
            logit = logits[min_energy_idcs[sample_idx, k_to_choose], :, sample_idx, :]
            if top_k == 0:
                this_logit.extend(logit)
            else:
                percentage_diff = check_percentage_in_range(min_energy_idcs[sample_idx, k_to_choose], min_var, max_var)
                if percentage_diff * 100. < x_percent:
                #     scale = percentage_diff
                #     # logit = logit * (scale + 1.)
                    this_logit = torch.stack(this_logit) * 0.5 + logit * 0.5
                # this_logit.extend(logit)
        if type(this_logit) is list:
            this_logit = torch.stack(this_logit).squeeze(0)
        decoded_logits.append(this_logit)
    return torch.stack(decoded_logits).transpose(0, 1), samplewise_energies


def  get_k_nearest_by_variance(logits, k=1, metric='softmax', return_indices=False, x_percent=10):
    x_percent = 10
    k = min(k, logits.size(0))
    if metric == 'softmax':
        logits_softmax = logits.softmax(-1)
        logits_var = logits_softmax.var(1).mean(-1)
    elif metric == 'entropy':
        logits_entropy = compute_shannon_entropy(logits, return_mean=False)
        logits_var = logits_entropy.var(1)
    else:
        raise NotImplementedError("Please provide a valid metric for variance calculation !")
    min_variances_wts, min_variances_idcs = logits_var.topk(dim=0, largest=False, k=k)
    if return_indices:
        if k == 1:
            min_variances_idcs = min_variances_idcs.squeeze(0)
            return min_variances_idcs
        else:
            raise NotImplementedError("Return indices not implemented for k > 1.")
    decoded_logits = []
    min_variances_idcs = min_variances_idcs.transpose(0, 1)
    min_variances_wts = min_variances_wts.transpose(0, 1)
    max_variance_wts, _ = logits_var.topk(dim = 0, largest=True, k = 1)
    max_variance_wts = max_variance_wts.transpose(0, 1)
    for sample_idx in range(logits.size(2)):
        # print(min_variances_wts.shape, max_variance_wts.shape, sample_idx)
        # print(min_variances_wts[sample_idx, 0])
        # print(max_variance_wts[sample_idx, 0])
        min_var, max_var = min_variances_wts[sample_idx, 0], max_variance_wts[sample_idx, 0]
        # x_percent = x_percent / 100
        this_logit = []
        for top_k in range(k):
            k_to_choose = top_k
            logit = logits[min_variances_idcs[sample_idx, k_to_choose], :, sample_idx, :]
            if top_k == 0:
                this_logit.extend(logit)
            if top_k == 1:
                percentage_diff = check_percentage_in_range(min_variances_wts[sample_idx, k_to_choose], min_var, max_var)
                # if percentage_diff * 100. < x_percent and min_variances_idcs[sample_idx, 0] == logits.size(0):
                #     scale = percentage_diff
                    # logit = logit * (scale + 1.)
                # this_logit = torch.stack(this_logit) * 0.5 + logit * 0.5
                this_logit.extend(logit)
        if type(this_logit) is list:
            this_logit = torch.stack(this_logit).squeeze(0)
        decoded_logits.append(this_logit)
    # print(torch.stack(decoded_logits).shape); exit(1)
    return torch.stack(decoded_logits).transpose(0, 1)

def check_percentage_in_range(z, x, y):
    return (z - x) / (y - x)

def get_modulewise_num_unique_preds(outputs, return_id=True):
    outputs = outputs.permute(2, 0, 1, 3)
    _, pred = torch.max(outputs.data, -1)
    num_unique_by_modules = []
    decoded_logits = []
    least_label_change_module_id = []
    for sample_idx in range(pred.size(0)):
        unique_by_modules = []
        for module_out in range(pred[sample_idx].size(0)):
            modulewise_unique_num = torch.unique(pred[sample_idx][module_out])
            unique_by_modules.append(modulewise_unique_num.size(0))
        unique_by_modules = torch.tensor(unique_by_modules, dtype=torch.int8)
        num_unique_by_modules.append(unique_by_modules)
        correct_id, _ = torch.min(unique_by_modules, 0)
        if return_id:
            least_label_change_module_id.append(correct_id)
        decoded_logits.append(outputs[sample_idx, correct_id, :, :])
    num_unique_by_modules = torch.stack(num_unique_by_modules)
    return least_label_change_module_id, num_unique_by_modules, torch.stack(decoded_logits).transpose(0,1)


def get_modulewise_top_two_prob_diff(outputs, return_id = True):
    outputs = outputs.permute(2, 0, 1, 3)
    output_probs = torch.softmax(outputs, dim=-1)
    top_two_largest, _ = output_probs.topk(dim = -1, largest=True, k = 2)
    top_two_largest_diff = top_two_largest[:, :, :, 0] - top_two_largest[:, :, :, 1]
    top_two_largest_diff = top_two_largest_diff.mean(-1)
    top_two_largest_diff_module_id = None
    decoded_logits = []
    for sample_idx in range(outputs.size(0)):
        _, largest_diff_module_id = torch.max(top_two_largest_diff[sample_idx], 0)
        decoded_logits.append(outputs[sample_idx, largest_diff_module_id, :, :])
    if return_id:
        _, top_two_largest_diff_module_id = top_two_largest_diff.topk(dim=1, largest=True, k=1)
        top_two_largest_diff_module_id = top_two_largest_diff_module_id.squeeze()
    return top_two_largest_diff_module_id, top_two_largest_diff, torch.stack(decoded_logits).transpose(0, 1)

def get_modulewise_top_prob_score(outputs, return_id = True):
    outputs = outputs.permute(2, 0, 1, 3)
    output_probs = torch.softmax(outputs, dim=-1)
    decoded_logits = []
    decoded_logits_id = []
    largest_prob_values = []
    for sample_idx in range(outputs.size(0)):
        max_probs, _ = torch.max(output_probs[sample_idx].mean(-1), -1)
        max_prob_val, max_prob_module_id = torch.max(max_probs, 0)
        decoded_logits.append(outputs[sample_idx, max_prob_module_id, :, :])
        largest_prob_values.append(max_prob_val)
        if return_id:
            decoded_logits_id.append(max_prob_module_id)

    if return_id:
        decoded_logits_id = torch.stack(decoded_logits_id)
    return decoded_logits_id, torch.stack(largest_prob_values), torch.stack(decoded_logits).transpose(0, 1)