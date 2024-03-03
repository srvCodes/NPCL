# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pickle
import math
import sys
from argparse import Namespace
from typing import Tuple

import torch
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel

from utils.ood_manager import get_ood_test_loader
from utils.loggers import *
from utils.status import ProgressBar
from utils.evaluation import evaluate, evaluate_ood
from utils.np_losses import linear_schedule_rate
try:
    import wandb
except ImportError:
    wandb = None

def save_checkpoint(model, optimizer, save_path, epoch, dump_dir="wts_dump"):
    save_path = f"{dump_dir}/{save_path}"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)

def save_pickle(obj, fname, dump_dir="wts_dump"):
    pickle.dump(obj, open(f"{dump_dir}/{fname}.obj","wb"))

def load_checkpoint(model, np_head, fname, optimizer=None, dump_dir = "wts_dump"):
    if os.path.isdir(dump_dir):
        for f in os.listdir(dump_dir):
            if f.startswith(fname):
                load_path = f"{dump_dir}/{fname}"
                model_checkpoint = torch.load(load_path + "_backbone.pt")
                np_checkpoint = torch.load(load_path + "_np.pt")
                model.load_state_dict(model_checkpoint['model_state_dict'])
                np_head.load_state_dict(np_checkpoint['model_state_dict'])
                epoch=None
                if optimizer is not None:
                    optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
                    epoch = model_checkpoint['epoch']
                model.eval()
                np_head.eval()
                return model, np_head, optimizer
    return None, None, None

def load_pickle(fname,  dump_dir="wts_dump", train_flag=False):
    full_path = f"{dump_dir}/{fname}.obj"
    obj = None
    try:
        with open(full_path, "rb") as f:
            obj = pickle.load(f)
            f.close()
    except FileNotFoundError:
        train_flag = True
    return obj, train_flag

def get_total_warmup_steps(train_loader_len, n_epochs, warmup_portion):
    total_training_steps = train_loader_len * n_epochs
    total_warmup_steps = warmup_portion * total_training_steps
    return total_warmup_steps, total_training_steps

def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    print(args)
    if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        args.wandb_url = wandb.run.get_url()

    model.net.train()
    model.net.to(model.device)
    if model.np_head is not None:
        model.np_head.train()
        model.np_head.to(model.device)
    results, results_mask_classes = [], []

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    if not args.ignore_other_metrics:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            _, _, _ = dataset_copy.get_data_loaders()
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            random_results_class, random_results_task = evaluate(model, dataset_copy, dummy_test=True)

    ood_test_loader = get_ood_test_loader(args) if args.eval_ood else None
    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):

        train_loader, test_loader, context_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t and not args.ignore_other_metrics:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

        scheduler = dataset.get_scheduler(model, args)
        global_step = 0
        total_warmup_steps, total_training_steps = get_total_warmup_steps(len(train_loader),
                                                                          model.args.n_epochs,
                                                                          model.args.warmup_portion)
        context_iter = iter(context_loader)

        train_flag = False
        if model.args.load_checkpoint and model.np_head is not None:
            filename = f'{args.dataset}_task_{t}_{dataset.N_TASKS}_tasks_{dataset.N_CLASSES_PER_TASK}_cls_per_task'
            pickled_obj, train_flag = load_pickle(filename+'_buffer', train_flag=train_flag)
            if not train_flag:
                model.buffer.num_seen_examples, model.buffer.examples, model.buffer.labels, model.buffer.task_labels = pickled_obj
                model.net, model.np_head,  model.opt = load_checkpoint(model.net, model.np_head, filename, model.opt)
                model.previous_time_to_task_distributions, _ = load_pickle(filename+'_tdist')
                model.previous_time_to_global_distributions, _ = load_pickle(filename+'_gdist')
                print(f"\nSuccessfully loaded pretrained model !")
        else:
            train_flag = True

        if train_flag:
            for epoch in range(model.args.n_epochs):
                if args.model == 'joint':
                    continue
                for i, data in enumerate(train_loader):
                    if global_step < total_warmup_steps:
                        learning_rate = linear_schedule_rate(0.00001, model.args.lr, global_step, total_warmup_steps)
                        for param_group in model.opt.param_groups:
                            param_group['lr'] = learning_rate
                    # for param_group in model.opt.param_groups:
                    #     print(param_group['lr'], global_step)
                    if args.debug_mode and i > 3:
                        break
                    if hasattr(dataset.train_loader.dataset, 'logits'):
                        inputs, labels, not_aug_inputs, logits = data
                        inputs = inputs.to(model.device)
                        labels = labels.to(model.device)
                        not_aug_inputs = not_aug_inputs.to(model.device)
                        logits = logits.to(model.device)
                        loss = model.meta_observe(inputs, labels, not_aug_inputs, logits, dataset.TASK_TO_LABELS)
                    else:
                        inputs, labels, not_aug_inputs = data
                        m = None
                        if args.use_context:
                            context_indices = model.get_context_indices(labels,
                                                                        m=math.ceil(args.context_batch_factor * inputs.size(0)), random=False)
                            context_inputs, context_labels = inputs[context_indices], labels[context_indices]
                            # try:
                            #     context_data = next(context_iter)
                            # except StopIteration:
                            #     context_iter = iter(context_loader)
                            #     context_data = next(context_iter)
                            # context_inputs, context_labels, _ = context_data
                            m = context_inputs.size(0)
                            inputs = torch.cat((context_inputs, inputs))
                            labels = torch.cat((context_labels, labels))
                        inputs, labels = inputs.to(model.device), labels.to(
                            model.device)
                        not_aug_inputs = not_aug_inputs.to(model.device)
                        loss = model.meta_observe(inputs, labels, not_aug_inputs, m, t, dataset.TASK_TO_LABELS,  global_step, total_training_steps, epoch)

                    assert not math.isnan(loss)
                    progress_bar.prog(i, len(train_loader), epoch, t, loss)
                    global_step += 1

                if scheduler is not None:
                    scheduler.step()
                # record current and global task distributions
                if epoch == args.n_epochs - 1 and 'np' in args.np_type:
                    observe_and_update_distributions(args, train_loader, context_loader, model, dataset, t, context_iter)

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        accs = evaluate(model, dataset)
        if model.args.eval_ood:
            _ = evaluate_ood(model, dataset, ood_test_loader=ood_test_loader)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        if not args.disable_log:
            logger.log(mean_acc)
            logger.log_fullacc(accs)

        if not args.nowand:
            d2={'RESULT_class_mean_accs': mean_acc[0], 'RESULT_task_mean_accs': mean_acc[1],
                **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs[0])},
                **{f'RESULT_task_acc_{i}': a for i, a in enumerate(accs[1])}}

            wandb.log(d2)

        # if train_flag:
        #     filename = f'{args.dataset}_task_{t}_{dataset.N_TASKS}_tasks_{dataset.N_CLASSES_PER_TASK}_cls_per_task'
        #     save_checkpoint(model.net, model.opt, filename + '_backbone.pt', epoch)
        #     save_checkpoint(model.np_head, model.opt, filename + '_np.pt', epoch)
        #     save_pickle((model.buffer.num_seen_examples, model.buffer.examples, model.buffer.labels,
        #                  model.buffer.task_labels), filename + '_buffer')
        #     save_pickle(model.previous_time_to_global_distributions, filename + '_gdist')
        #     save_pickle(model.previous_time_to_task_distributions, filename + '_tdist')


    if model.args.visualize_latent:
        model.get_averaged_task_to_epoch_to_kl(dataset.N_TASKS-1)
        pickle.dump(model.task_to_epoch_to_kl, open(f"{args.np_type}{'_residual_' if args.residual_normal_kl else ''}_"
                                                    f"task_to_epoch_to_kl_klt_{args.kl_t}_klg_{args.kl_g}_"
                                                    f"{'warmup' if args.kl_warmup else ''}"
                                                    f"{'cutoff_' + str(args.kl_cutoff) if args.min_info_constraint else ''}.pkl", "wb"))

    if not args.disable_log and not args.ignore_other_metrics:
        logger.add_bwt(results, results_mask_classes)
        logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            logger.add_fwt(results, random_results_class,
                    results_mask_classes, random_results_task)

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()

def observe_and_update_distributions(args, train_loader, context_loader, model, dataset, t, context_iter):
    all_batches_dist_task = []
    all_batches_dist_global = []
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            inputs, labels, _ = data
            if args.use_context:
                # context_indices = model.get_context_indices(labels, m=math.ceil(args.context_batch_factor * inputs.size(0)), random=True)
                # context_inputs, context_labels = inputs[context_indices], labels[context_indices]
                try:
                    context_data = next(context_iter)
                except StopIteration:
                    context_iter = iter(context_loader)
                    context_data = next(context_iter)
            context_inputs, context_labels, _ = context_data
            m = context_inputs.size(0)
            inputs = torch.cat((context_inputs, inputs))
            labels = torch.cat((context_labels, labels))
            inputs, labels = inputs.to(model.device), labels.to(
                model.device)

            global_task_dist_, cur_task_dist_ = model.observe_dist(inputs, labels,
                                                                   m, t,
                                                                   dataset.TASK_TO_LABELS)
            if global_task_dist_ is not None:
                if len(all_batches_dist_global) == 0:
                    all_batches_dist_global = [global_task_dist_[0].detach(), global_task_dist_[1].detach()]
                else:
                    all_batches_dist_global[0] += global_task_dist_[0].detach()
                    all_batches_dist_global[1] += global_task_dist_[1].detach()
            if cur_task_dist_ is not None:
                if len(all_batches_dist_task) == 0:
                    all_batches_dist_task = [cur_task_dist_[t][0].detach(), cur_task_dist_[t][1].detach()]
                else:
                    all_batches_dist_task[0] += cur_task_dist_[t][0].detach()
                    all_batches_dist_task[1] += cur_task_dist_[t][1].detach()

        divisor = len(train_loader)
        if len(all_batches_dist_task) > 0:
            model.previous_time_to_task_distributions.update(
                {t: (torch.div(all_batches_dist_task[0], divisor), torch.div(all_batches_dist_task[1], divisor))})
        if len(all_batches_dist_global) > 0:
            model.previous_time_to_global_distributions.update(
                {t: (torch.div(all_batches_dist_global[0], divisor), torch.div(all_batches_dist_global[1], divisor))})