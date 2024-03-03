# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args, add_np_args
from utils.buffer import Buffer
import torch
from backbone.utils.moe_helpers import get_moe_outputs

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_np_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    return parser


class Der(ContinualModel):
    NAME = 'der'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, np_head, backbone, loss, args, transform):
        super(Der, self).__init__(np_head, backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe_base(self, inputs, labels):

        outputs, _ = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs, _ = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

        return loss, outputs

    def observe(self, inputs, labels, not_aug_inputs, m, cur_task=None, task_to_labels=None, global_step=None, num_total_iter=None, epoch_num=None):
        m = 0 if m is None else m
        real_batch_size = inputs.shape[0]
        self.opt.zero_grad()
        target_task_labels = None
        if self.args.np_type == '':
            loss, outputs = self.observe_base(inputs, labels)
        else:
            target_task_labels = torch.tensor([cur_task] * inputs.size(0), dtype=torch.int64).to(self.device)
            feats = self.net(inputs, returnt='features')
            context_outputs, context_labels, context_task_labels = feats[:m], labels[:m], target_task_labels[:m]
            context_labels_one_hot = F.one_hot(context_labels.view(-1), num_classes=self.net.num_classes)
            labels_one_hot = F.one_hot(labels.view(-1), num_classes=self.net.num_classes)
            context_labels = context_labels_one_hot if not self.args.label_embed else context_labels
            target_labels = labels_one_hot if not self.args.label_embed else labels
            outputs, (q_context, q_target), (alpha, y_hard) = self.np_head(context_outputs, context_labels,
                                                          feats,
                                                          target_labels,
                                                          forward_times=self.args.forward_times_train,
                                                          context_task_labels=context_task_labels,
                                                          target_task_labels=target_task_labels,
                                                          task_to_labels=task_to_labels,
                                                          clnp_stochasticity=self.args.clnp_stochasticity
                                                          )

            loss = self.loss(outputs, labels_one_hot)
            dist_loss, _ = self.get_dist_losses(q_target[0], q_context[0], q_target[1], q_context[1],
                                                cur_task=cur_task, compute_kd=False,
                                                global_step=global_step, num_total_iter=num_total_iter)
            loss += dist_loss
            if not self.buffer.is_empty():
                buf_inputs, buf_labels, buf_logits, buf_target_task_labels = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform)
                buf_feats = self.net(buf_inputs, returnt='features')
                buf_context_indices = self.get_context_indices(buf_labels, m=m, random=False)
                buf_context_outputs, buf_context_labels, buf_context_task_labels = \
                    buf_feats[buf_context_indices], buf_labels[buf_context_indices], buf_target_task_labels[buf_context_indices]
                # buf_context_outputs, buf_context_labels, buf_context_task_labels = buf_feats[:m], buf_labels[:m], buf_target_task_labels[:m]
                buf_context_labels_one_hot = F.one_hot(buf_context_labels.view(-1),
                                                       num_classes=self.np_head.num_classes)
                buf_labels_one_hot = F.one_hot(buf_labels.view(-1), num_classes=self.np_head.num_classes)
                context_labels = buf_context_labels_one_hot if not self.args.label_embed else buf_context_labels
                target_labels = buf_labels_one_hot if not self.args.label_embed else buf_labels
                buf_logits_target_out, (buf_q_context, buf_q_target), (alpha, y_hard) = self.np_head(
                    buf_context_outputs,
                    context_labels,
                    buf_feats,
                    target_labels,
                    forward_times=self.args.forward_times_train,
                    context_task_labels=buf_context_task_labels,
                    target_task_labels=buf_target_task_labels,
                    task_to_labels=task_to_labels
                )
                c_kld = 0
                if alpha is not None:
                    buf_logits_target_out = get_moe_outputs(buf_logits_target_out, alpha, y_hard, hard=False)#, task_dist_ids=list(buf_q_target[1].keys()))
                    if alpha.size(1) > 1:
                        try:
                            c_kld = F.cross_entropy(alpha.squeeze(), buf_target_task_labels)
                        except RuntimeError:
                            print(alpha.shape, buf_target_task_labels.shape, alpha.squeeze()[:5], buf_target_task_labels[:5], cur_task); exit(1)
                loss += c_kld
                loss += self.args.alpha * F.mse_loss(buf_logits_target_out.mean(0), buf_logits)
                dist_loss, kl_per_dist = self.get_dist_losses(buf_q_target[0], buf_q_context[0],
                                                              buf_q_target[1], buf_q_context[1],
                                                              cur_task=cur_task, epoch_num=epoch_num,
                                                              global_step=global_step, num_total_iter=num_total_iter
                                                              )
                loss += dist_loss

            outputs = outputs.mean(0)
        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[m:real_batch_size],
                             logits=outputs[m:real_batch_size].data,
                             task_labels=target_task_labels[m:real_batch_size] if target_task_labels is not None else None,
                             )

        return loss.item()