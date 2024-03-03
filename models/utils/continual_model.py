# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from argparse import Namespace
from contextlib import suppress
from typing import List
import numpy as np

import torch
import torch.nn as nn
from torch.optim import SGD

from utils.conf import get_device
from utils.magic import persistent_locals
import torch.nn.functional as F
from utils.distributions import compute_kl_div, compute_js_div
from utils.np_losses import linear_schedule_rate
from utils.distributions import kl_coeff

with suppress(ImportError):
    import wandb


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME: str
    COMPATIBILITY: List[str]

    def __init__(self, np_head: nn.Module, backbone: nn.Module, loss: nn.Module,
                 args: Namespace, transform: nn.Module) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.np_head = np_head
        self.loss = loss
        self.args = args
        self.transform = transform
        all_params = self.net.parameters() if np_head is None else list(self.net.parameters()) + list(self.np_head.parameters())
        self.opt = SGD(all_params, lr=self.args.lr)
        self.device = get_device()
        self.previous_time_to_task_distributions = {}
        self.previous_time_to_global_distributions = {}
        self.task_to_epoch_to_kl = {}
        self.prev_batch_epoch = 0
        self.prev_batch_task = 0
        if not self.NAME or not self.COMPATIBILITY:
            raise NotImplementedError('Please specify the name and the compatibility of the model.')

    def forward(self, x: torch.Tensor, x_context_in = None, context_labels_one_hot=None, context_task_labels=None, task_to_labels=None, cur_test_task=None) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        if not self.args.np_type:
            return self.net(x)
        else:
            feats = self.net(x, returnt='features')
            context_feats = self.net(x_context_in, returnt='features')
            outputs, all_outputs = self.np_head(context_feats, context_labels_one_hot,
                                      feats,
                                      phase_train=False,
                                      forward_times=self.args.forward_times_test,
                                      context_task_labels=context_task_labels,
                                      task_to_labels=task_to_labels,
                                      clnp_stochasticity=self.args.clnp_stochasticity, top_k_decode = self.args.top_k_decode,
                                                cur_test_task = cur_test_task, x_percent=self.args.top_k_decode_cutoff,
                                                missing_dists = self.previous_time_to_task_distributions if len(self.previous_time_to_task_distributions) else None)
            return outputs.mean(0), all_outputs

    def meta_observe(self, *args, **kwargs):
        if 'wandb' in sys.modules and not self.args.nowand:
            pl = persistent_locals(self.observe)
            ret = pl(*args, **kwargs)
            self.autolog_wandb(pl.locals)
        else:
            ret = self.observe(*args, **kwargs)
        return ret

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor, m: int, task_to_labels=None) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        raise NotImplementedError

    def autolog_wandb(self, locals):
        """
        All variables starting with "_wandb_" or "loss" in the observe function
        are automatically logged to wandb upon return if wandb is installed.
        """
        if not self.args.nowand and not self.args.debug_mode:
            wandb.log({k: (v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v)
                      for k, v in locals.items() if k.startswith('_wandb_') or k.startswith('loss')})


    def get_context_indices(self, labels, m, random=False):
        if random:
            buf_context_indices = torch.randperm(labels.size(0))[:m]
        else:
            # guarantees that the context set contains at least one sample from each task
            unique, idx, counts = torch.unique(labels, dim=0, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat((torch.tensor([0]).to(cum_sum.device), cum_sum[:-1]))
            buf_context_indices = ind_sorted[cum_sum]
            if buf_context_indices.size(0) < m:
                diff = m - buf_context_indices.size(0)
                buf_context_indices_permuted = torch.randperm(labels.size(0)).to(buf_context_indices.device)
                for val in buf_context_indices:
                    buf_context_indices_permuted = buf_context_indices_permuted[buf_context_indices_permuted != val]
                buf_context_indices = torch.cat((buf_context_indices, buf_context_indices_permuted[:diff]))
        return buf_context_indices

    @torch.no_grad()
    def observe_dist(self, inputs, labels, m, cur_task=None, task_to_labels=None) -> tuple:
        real_batch_size = inputs[m:].shape[0]
        target_task_labels = torch.tensor([cur_task] * inputs.size(0), dtype=torch.int64).to(self.device)

        if not self.buffer.is_empty():
            buf_retrieval = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            if 'der' in self.args.model:
                buf_inputs, buf_labels, _, buf_target_task_labels = buf_retrieval
            else:
                buf_inputs, buf_labels, buf_target_task_labels = buf_retrieval
            buf_context_indices = self.get_context_indices(buf_labels, m=8, random=True)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))
            target_task_labels = torch.cat((target_task_labels, buf_target_task_labels))

        feats = self.net(inputs, returnt='features')
        context_outputs = torch.cat((feats[:m], feats[m + real_batch_size:][buf_context_indices]))
        context_labels = torch.cat((labels[:m], labels[m + real_batch_size:][buf_context_indices]))
        context_task_labels = torch.cat((target_task_labels[:m], target_task_labels[m + real_batch_size:][buf_context_indices]))

        context_labels_one_hot = F.one_hot(context_labels.view(-1), num_classes=self.np_head.num_classes)
        labels_one_hot = F.one_hot(labels.view(-1), num_classes=self.np_head.num_classes)

        context_labels = context_labels_one_hot if not self.args.label_embed else context_labels
        target_labels = labels_one_hot if not self.args.label_embed else labels

        _, (q_context, q_target), _ = self.np_head(context_outputs, context_labels,
                                                      feats,
                                                      target_labels,
                                                      forward_times=self.args.forward_times_train,
                                                      context_task_labels=context_task_labels,
                                                      target_task_labels=target_task_labels,
                                                      task_to_labels=task_to_labels,
                                                    clnp_stochasticity=self.args.clnp_stochasticity
                                                    )

        return q_target[0], q_target[1]

    def get_averaged_task_to_epoch_to_kl(self, cur_task):
        for latent_layer, val in self.task_to_epoch_to_kl[cur_task][self.prev_batch_epoch].items():
            if type(val) is list:
                self.task_to_epoch_to_kl[cur_task][self.prev_batch_epoch].update({latent_layer: np.mean(val)})
            elif type(val) is dict:
                for task_id, iter_val in val.items():
                    self.task_to_epoch_to_kl[cur_task][self.prev_batch_epoch][latent_layer].update(
                        {task_id: np.mean(iter_val)})

    def get_dist_losses(self, q_target_global=None, q_context_global=None,
                        q_target_taskwise=None, q_context_taskwise=None,
                        cur_task=None, kd_choice='kl', compute_kd=True,
                         epoch_num=None,
                        global_step=None, num_total_iter= None):
        if self.args.visualize_latent and epoch_num is not None:
            # check if new task has arrived
            if self.prev_batch_task != cur_task:
                self.get_averaged_task_to_epoch_to_kl(self.prev_batch_task)
                self.prev_batch_epoch = 0
            # if it is the old task, then check if a new training epoch has been triggered
            if epoch_num != self.prev_batch_epoch:
                self.get_averaged_task_to_epoch_to_kl(cur_task)
            self.prev_batch_epoch = epoch_num
            self.prev_batch_task = cur_task

        loss = 0.
        kd_loss = compute_js_div if kd_choice == 'js' else compute_kl_div
        kl_per_layer_per_group = {}
        if q_target_global is not None:
            if self.args.kl_g > 0:
                klg = self.args.kl_g
                if self.args.kl_warmup:
                    # kl_coef = kl_coeff(global_step, self.args.kl_anneal_portion * num_total_iter,
                    #                    self.args.kl_const_portion * num_total_iter, self.args.kl_const_coeff)
                    klg_coef = linear_schedule_rate(self.args.kl_const_coeff, self.args.kl_g, global_step,
                                                   int(self.args.kl_anneal_portion * num_total_iter))
                    klg = klg_coef

                kl = compute_kl_div(q_target_global[0], q_target_global[1],
                                       q_context_global[0], q_context_global[1], self.args.residual_normal_kl)
                if self.args.min_info_constraint and kl < self.args.kl_cutoff:
                    kl = torch.full_like(kl, self.args.kl_cutoff)
                loss += kl * klg
                if self.args.visualize_latent and epoch_num is not None:
                    if cur_task not in self.task_to_epoch_to_kl:
                        self.task_to_epoch_to_kl[cur_task] = {epoch_num:  {1: [kl.detach().item()]}}
                    else:
                        if epoch_num not in self.task_to_epoch_to_kl[cur_task]:
                            self.task_to_epoch_to_kl[cur_task].update({epoch_num:  {1: [kl.detach().item()]}})
                        else:
                            self.task_to_epoch_to_kl[cur_task][epoch_num][1].append(kl.detach().item())

            if compute_kd and self.args.kd_gr > 0 and cur_task:
                for t_id in range(cur_task):
                    loss += kd_loss(q_target_global[0], q_target_global[1],
                                    self.previous_time_to_global_distributions[t_id][0],
                                    self.previous_time_to_global_distributions[t_id][1]) * self.args.kd_gr
                if self.args.kd_context:
                    loss += kd_loss(q_context_global[0], q_context_global[1],
                                self.previous_time_to_global_distributions[cur_task-1][0],
                                self.previous_time_to_global_distributions[cur_task-1][1]) * self.args.kd_gr

        if q_target_taskwise is not None:
            kl_taskwise, kd_taskwise = [], []
            for task_id in q_target_taskwise:
                if self.args.kl_t > 0 and task_id in q_context_taskwise:
                    kl = compute_kl_div(q_target_taskwise[task_id][0], q_target_taskwise[task_id][1],
                                                  q_context_taskwise[task_id][0], q_context_taskwise[task_id][1],
                                                    self.args.residual_normal_kl)
                    if self.args.min_info_constraint and kl < self.args.kl_cutoff:
                        kl = torch.full_like(kl, self.args.kl_cutoff)
                    kl_taskwise.append(kl)
                    if self.args.visualize_latent and epoch_num is not None:
                        if cur_task not in self.task_to_epoch_to_kl:
                            self.task_to_epoch_to_kl[cur_task] = {epoch_num: {2: {task_id: [kl.detach().item()]}}}
                        else:
                            if epoch_num not in self.task_to_epoch_to_kl[cur_task]:
                                self.task_to_epoch_to_kl[cur_task].update({epoch_num:  {2: { task_id: [kl.detach().item()]}}})
                            else:
                                if 2 not in self.task_to_epoch_to_kl[cur_task][epoch_num]:
                                    self.task_to_epoch_to_kl[cur_task][epoch_num][2] = {task_id: [kl.detach().item()]}
                                else:
                                    if task_id not in self.task_to_epoch_to_kl[cur_task][epoch_num][2]:
                                        self.task_to_epoch_to_kl[cur_task][epoch_num][2].update({task_id: [kl.detach().item()]})
                                    else:
                                        self.task_to_epoch_to_kl[cur_task][epoch_num][2][task_id].append(kl.detach().item())
                if compute_kd and self.args.kd_tr > 0  and task_id in self.previous_time_to_task_distributions and cur_task:
                    kd_taskwise.append(kd_loss(q_target_taskwise[task_id][0], q_target_taskwise[task_id][1],
                                           self.previous_time_to_task_distributions[task_id][0],
                                           self.previous_time_to_task_distributions[task_id][1]))
                    if self.args.kd_context and task_id in q_context_taskwise:
                        kd_taskwise.append(kd_loss(q_context_taskwise[task_id][0], q_context_taskwise[task_id][1],
                                           self.previous_time_to_task_distributions[task_id][0],
                                           self.previous_time_to_task_distributions[task_id][1]))
            if self.args.kl_t > 0 and len(kl_taskwise) > 0:
                klt = self.args.kl_t
                if self.args.kl_warmup:
                    klt_coef = linear_schedule_rate(self.args.kl_const_coeff, self.args.kl_t, global_step,
                                                   int(self.args.kl_anneal_portion * num_total_iter))
                    klt = klt_coef
                # if global_step % 200 == 0:
                #     print(f"kl-t coefficient: {klt}")
                loss += torch.stack(kl_taskwise).sum(0) * klt
            if self.args.kd_tr > 0 and len(kd_taskwise) > 0:
                loss += torch.stack(kd_taskwise).sum(0) * self.args.kd_tr

        return loss, kl_per_layer_per_group

    def kl_div_cat(self, softmax_y, cat_dim=2):
        log_ratio = torch.log(softmax_y * cat_dim + 1e-20)
        kld = torch.sum(softmax_y * log_ratio, dim=-1).mean()
        return kld