# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from typing import Tuple

import numpy as np
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.sampler import BatchSampler
import random

class ContinualDataset:
    """
    Continual learning evaluation setting.
    """
    NAME: str
    SETTING: str
    N_CLASSES_PER_TASK: int
    N_TASKS: int
    DET_SET_TRANSFORMER: bool
    SET_TRANSFORMER_SEEDS: int
    TASK_TO_LABELS: dict

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args

        if not all((self.NAME, self.SETTING, self.N_CLASSES_PER_TASK, self.N_TASKS)):
            raise NotImplementedError('The dataset must be initialized with all the required fields.')

        ContinualDataset.DET_SET_TRANSFORMER = args.det_set_transformer
        ContinualDataset.SET_TRANSFORMER_SEEDS = args.set_transformer_seeds

    def get_task_to_labels(self):
        return self.task_to_labels

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training, test and context loaders
        """
        raise NotImplementedError

    @staticmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_transform() -> nn.Module:
        """
        Returns the transform to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_loss(np_name: str) -> nn.Module:
        """
        Returns the loss to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_normalization_transform() -> nn.Module:
        """
        Returns the transform used for normalizing the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_denormalization_transform() -> nn.Module:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_scheduler(model, args: Namespace) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Returns the scheduler to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    def get_epochs():
        raise NotImplementedError

    @staticmethod
    def get_batch_size():
        raise NotImplementedError

    @staticmethod
    def get_minibatch_size():
        raise NotImplementedError

    @staticmethod
    def get_np_head(np_name: str):
        raise NotImplementedError

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def sample_context_data(context_dataset, num_labels=50):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    samples_per_class = num_labels

    lb_data = []
    lbs = []
    lb_idx = []
    for c in np.unique(context_dataset.targets):
        idx = np.where(context_dataset.targets == c)[0]
        idx = np.random.choice(idx, samples_per_class, False)
        lb_idx.extend(idx)

        lb_data.extend(context_dataset.data[idx])
        lbs.extend(context_dataset.targets[idx])
    # from collections import Counter
    # print(Counter(lbs)); exit(1)
    context_dataset.data = lb_data
    context_dataset.targets = lbs
    return context_dataset, np.array(lb_idx)

def get_context_loader(context_dataset, batch_size,
                       num_iters=None,
                       replacement=True,
                       drop_last=True,
                       n_epochs=None,
                       fix_context=False,
                       seed_worker=None,
                       g=None,
                       len_train_loader=None):
    if fix_context:
        batch_size = 1000 # upper limit
        data_sampler = SequentialSampler(context_dataset)
    else:
        if len_train_loader is not None:
            num_samples = batch_size * len_train_loader
        elif n_epochs is not None:
            num_samples = len(context_dataset) * n_epochs
        elif num_iters is not None:
            num_samples = batch_size * num_iters
        else:
            raise Exception("Please provide either the total number of training iterations or the number of training epochs for the context loader to be functional!")
        data_sampler = RandomSampler(context_dataset, replacement, num_samples)
    batch_sampler = BatchSampler(data_sampler, batch_size, drop_last)
    return DataLoader(context_dataset, batch_sampler=batch_sampler,
                          num_workers=4, worker_init_fn=seed_worker,
                        generator=g,)

# def get_context_loader(context_dataset, batch_size, num_iters=None, replacement=True, drop_last=True, n_epochs=None, n_classes=None, fix_context=False,
#                        seed_worker=None, g=None):
#
#     if num_iters is not None:
#         num_samples = batch_size * num_iters
#     elif (n_epochs is not None) and (num_iters is None):
#         num_samples = len(context_dataset) * n_epochs
#     # num_samples = 1
#     if fix_context:
#         batch_size = 1000 # upper limit
#         data_sampler = SequentialSampler(context_dataset)
#     else:
#         data_sampler = RandomSampler(context_dataset, replacement, num_samples)
#     batch_sampler = BatchSampler(data_sampler, batch_size, drop_last)
#     return DataLoader(context_dataset, batch_sampler=batch_sampler,
#                           num_workers=4, worker_init_fn=seed_worker,
#                         generator=g,)

def store_masked_loaders(train_dataset: Dataset, test_dataset: Dataset,
                         setting: ContinualDataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i,
                                np.array(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
    test_mask = np.logical_and(np.array(test_dataset.targets) >= setting.i,
                               np.array(test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    test_dataset.data = test_dataset.data[test_mask]

    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    # make dataloader deterministic
    g = torch.Generator()
    g.manual_seed(0)

    from copy import deepcopy
    context_dataset = deepcopy(train_dataset)
    context_dataset, context_idcs = sample_context_data(context_dataset,
                                                        num_labels=setting.args.num_labels)

    train_loader = DataLoader(train_dataset,
                              batch_size=setting.args.batch_size, shuffle=True, num_workers=4,
                              worker_init_fn=seed_worker,
                              generator=g,
                              )
    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.batch_size, shuffle=False, num_workers=4,
                             worker_init_fn=seed_worker,
                             generator=g,
                             )
    context_loader = get_context_loader(context_dataset,
                                        int(setting.args.batch_size * setting.args.context_batch_factor),
                                        drop_last=False,
                                        num_iters=len(train_loader) * setting.args.n_epochs,
                                        seed_worker=seed_worker, g=g,
                                        n_epochs=None,#setting.args.n_epochs,
                                        len_train_loader=None)#len(train_loader))
    # context_loader = get_context_loader(context_dataset,
    #                                     int(setting.args.batch_size * setting.args.context_batch_factor),
    #                                     drop_last=False,
    #                                     n_classes=setting.N_CLASSES_PER_TASK,
    #                                     num_iters=len(train_loader) * setting.args.n_epochs,
    #                                     seed_worker=seed_worker, g=g)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CLASSES_PER_TASK
    setting.TASK_TO_LABELS[int(setting.i/setting.N_CLASSES_PER_TASK) - 1] = np.unique(context_dataset.targets)
    return train_loader, test_loader, context_loader


def get_previous_train_loader(train_dataset: Dataset, batch_size: int,
                              setting: ContinualDataset) -> DataLoader:
    """
    Creates a dataloader for the previous task.
    :param train_dataset: the entire training set
    :param batch_size: the desired batch size
    :param setting: the continual dataset at hand
    :return: a dataloader
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >=
                                setting.i - setting.N_CLASSES_PER_TASK, np.array(train_dataset.targets)
                                < setting.i - setting.N_CLASSES_PER_TASK + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
