# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')

    parser.add_argument('--n_epochs', type=int,
                        help='Batch size.')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size.')

    parser.add_argument('--distributed', type=str, default='no', choices=['no', 'dp', 'ddp'])

def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--non_verbose', default=0, choices=[0, 1], type=int, help='Make progress bars non verbose')
    parser.add_argument('--disable_log', default=0, choices=[0, 1], type=int, help='Enable csv logging')

    parser.add_argument('--validation', default=0, choices=[0, 1], type=int,
                        help='Test on the validation set')
    parser.add_argument('--ignore_other_metrics', default=0, choices=[0, 1], type=int,
                        help='disable additional metrics')
    parser.add_argument('--debug_mode', type=int, default=0, help='Run only a few forward steps per epoch')
    parser.add_argument('--nowand', default=1, choices=[0, 1], type=int, help='Inhibit wandb logging')
    parser.add_argument('--wandb_entity', type=str, default='regaz', help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, default='mammoth', help='Wandb project name')

def add_np_args(parser: ArgumentParser) -> None:
    parser.add_argument('--num_labels', type=int, default=5, required=False, help="Total no of training points per class to be included in the context dataset")
    parser.add_argument('--context-batch-factor', type=float, default=0.25, help="proportion of the batch_size to use as context batch size")
    parser.add_argument('--use_context', action='store_true', help='Use context data for training')
    parser.add_argument('--np_type', type=str, required=False, default='', help='specify np type: "anp", "npcl", "npcl-no-hierarchy", "npcl-moe')
    parser.add_argument('--forward_times_train', type=int, default=5, required=False, help='num of Monte Carlo samples for training')
    parser.add_argument('--forward_times_test', type=int, default=5, required=False, help='num of Monte Carlo samples for testing')
    parser.add_argument('--label_embed', action='store_true', help='learn label embeddings instead of concatenating')
    parser.add_argument('--det_set_transformer', action='store_true', help='use set transformer along det path')
    parser.add_argument('--set_transformer_seeds', type=int, default=1, required=False, help='num of seed vectors that are outputs of set transformer decoder')
    parser.add_argument('--clnp_stochasticity', type=str, default='all_global_unique', help="'all_global', 'all_global_unique', 'all_local', 'mix'")
    parser.add_argument('--warmup-portion', type=float, default=0.5, help='portion of warmup iterations out of total training iterations')
    parser.add_argument('--test-oracle-npcl', action='store_true', help="Use test time oracle for NPCL")
    parser.add_argument('--kl-t', type=float, default=0., help='taskwise kl div weight')
    parser.add_argument('--kl-g', type=float, default=0., help='global kl div weight')
    parser.add_argument('--kd-tr', type=float, default=0., help='taskwise kd weight')
    parser.add_argument('--kd-gr', type=float, default=0., help='global kd weight')
    parser.add_argument('--kd-context', action='store_true', help='use context set as well to compute KD losses')
    parser.add_argument('--kl-warmup', action='store_true', help='use KL warmup')
    parser.add_argument('--viz-modulewise-pred', action='store_true', help='visualize task to module uncertainty, variance,  etc.')
    parser.add_argument('--kl_anneal_portion', type=float, default=0.3, help='The portions epochs that KL is annealed')
    parser.add_argument('--kl_const_portion', type=float, default=0.0001, help='The portions epochs that KL is constant at kl_const_coeff')
    parser.add_argument('--kl_const_coeff', type=float, default=0.0001, help='The constant value used for min KL coeff')
    parser.add_argument('--residual-normal-kl', action='store_true', help='replace KL by Residual Normal distribution from the NVAE paper')
    parser.add_argument('--visualize-latent',  action='store_true', help='visualize latent representation: epochwise and layerwise KL, and tsne of taskwise zs')
    parser.add_argument('--min-info-constraint', action='store_true', help='apply kl-cutoff value that removes the effect of the KL term when it is below a certain threshold')
    parser.add_argument('--kl-cutoff', type=float, default=0.25, help='The nats of information per latent variable subset')
    parser.add_argument('--eval-ood', action='store_true', help='evaluate incremntally trained model on ood datasets')
    parser.add_argument('--ood-dataset', type=str, required=False,
                        choices=['cifar10', 'cifar100'],
                        help='Which dataset to eval ood on.')
    parser.add_argument('--top-k-decode', type=int, default=1, help='top-k task modules to decode logits from during inference')
    parser.add_argument('--top-k-decode-cutoff', type=float, default=10, help='percentage cutoff to consider top-k task module outputs')
    parser.add_argument('--load-checkpoint', action='store_true', help= "task checkpoints to be loaded instead of training" )

def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int,
                        help='The batch size of the memory buffer.')
