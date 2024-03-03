import torch
import os
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from typing import Tuple
import torch.nn.functional as F
from utils.uncertainty_quantifiers import compute_shannon_entropy
import numpy as np
from utils.visualize_helper import store_dict_as_df
from backbone.neural_processes.NPCL_robust import get_k_nearest_by_variance, get_k_nearest_by_uncertainty
from utils.ood_manager import get_measures
from timeit import default_timer as timer

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False, dummy_test=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    if model.np_head is not None:
        model.np_head.eval()
    accs, accs_mask_classes = [], []

    buf_inputs, buf_labels, buf_labels_one_hot, buf_task_labels = None, None, None, None
    if model.args.np_type and not dummy_test:
        buf_retrieval = model.buffer.get_all_data(transform=model.transform)
        if 'der' in dataset.args.model:
            buf_inputs, buf_labels, _, buf_task_labels = buf_retrieval
        else:
            buf_inputs, buf_labels, buf_task_labels = buf_retrieval
        buf_labels_one_hot = F.one_hot(buf_labels.view(-1), num_classes=model.net.num_classes)
        # print(buf_inputs.shape, buf_labels.shape, buf_labels_one_hot.shape)

    task_to_module_entropies, task_to_module_var_softmax, task_to_module_var_entropy, task_to_module_ranking, task_to_module_energy = {}, {}, {}, {}, {}
    num_matching_indices, num_matching_indices_a, num_matching_indices_b , num_matching_indices_c = 0, 0 , 0, 0
    start = timer()

    for k, test_loader in enumerate(dataset.test_loaders):
        # print(f"Test loader for {k}-th task")
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        all_outputs = None
        batch_to_module_entropies = {j: [] for j in range(len(dataset.test_loaders))}
        batch_to_module_var_softmax = {j: [] for j in range(len(dataset.test_loaders))}
        batch_to_module_var_entropy = {j: [] for j in range(len(dataset.test_loaders))}
        batch_to_module_ranking = {j: [] for j in range(len(dataset.test_loaders))}
        batch_to_module_energy = {j: [] for j in range(len(dataset.test_loaders))}
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if model.args.np_type and dummy_test:
                    buf_inputs = torch.rand_like(inputs)
                    buf_labels = torch.randint(0, model.net.num_classes, size=(buf_inputs.size(0),)).to(model.device)
                    buf_labels_one_hot = F.one_hot(buf_labels.view(-1), num_classes=model.net.num_classes)
                    buf_task_labels = torch.randint(0, len(dataset.test_loaders), size=(buf_inputs.size(0),)).to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    context_labels = buf_labels_one_hot if not model.args.label_embed else buf_labels
                    outputs, all_outputs = model(inputs, buf_inputs, context_labels, buf_task_labels, dataset.TASK_TO_LABELS, cur_test_task=k)
                    # ############ Match indices ##############
                    # min_entropy_indices = get_k_nearest_by_uncertainty(all_outputs, return_indices=True)
                    # min_var_entropy_indices = get_k_nearest_by_variance(logits=all_outputs, k=1, metric='entropy', return_indices=True)
                    # min_var_softmax_indices = get_k_nearest_by_variance(logits=all_outputs, k=1, metric='softmax', return_indices=True)
                    # a = min_entropy_indices == min_var_entropy_indices
                    # b = min_entropy_indices == min_var_softmax_indices
                    # c = min_var_softmax_indices == min_var_entropy_indices
                    # num_matching_indices += torch.sum(torch.logical_and(torch.logical_and(a, b), c))
                    # num_matching_indices_a += torch.sum(a)
                    # num_matching_indices_b += torch.sum(b)
                    # num_matching_indices_c += torch.sum(c)

                _, pred = torch.max(outputs.data, 1)
                corr_mask = pred == labels
                correct += torch.sum(corr_mask).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

                if all_outputs is not None and model.args.viz_modulewise_pred and not dummy_test and len(dataset.test_loaders) > 1 and not last:
                    entropies, variance_softmax, variance_entropy, energies = compute_uncertainties(all_outputs, metrics={'entropy', 'variance', 'energy'})
                    incorrect_entropies = entropies.transpose(0,1)[corr_mask == False].transpose(0,1)
                    entropies = entropies.transpose(0,1)[corr_mask == False].transpose(0,1)
                    # variance_softmax = variance_softmax.transpose(0,1)[corr_mask == False].transpose(0,1)
                    # variance_entropy = variance_entropy.transpose(0,1)[corr_mask == False].transpose(0,1)

                    for each in range(entropies.size(0)):
                        batch_to_module_entropies[each].extend(entropies[each].tolist())
                        batch_to_module_var_softmax[each].extend(variance_softmax[each].tolist())
                        batch_to_module_var_entropy[each].extend(variance_entropy[each].tolist())
                        batch_to_module_ranking[each].extend(incorrect_entropies[each].tolist())
                        batch_to_module_energy[each].extend(energies[each].tolist())

        if model.args.viz_modulewise_pred and not dummy_test  and len(dataset.test_loaders) > 1 and not last:
            task_to_module_entropies[k] = {key_: np.mean(val_) for key_, val_ in batch_to_module_entropies.items()}
            task_to_module_var_entropy[k] = {key_: np.mean(val_) for key_, val_ in batch_to_module_var_entropy.items()}
            task_to_module_var_softmax[k] = {key_: np.mean(val_) for key_, val_ in batch_to_module_var_softmax.items()}
            task_to_module_ranking[k] = {key_: np.mean(val_) for key_, val_ in batch_to_module_ranking.items()}
            task_to_module_energy[k] = {key_: np.mean(val_) for key_, val_ in batch_to_module_energy.items()}

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
    end = timer()
    if not dummy_test and not last:
        print(f"Inference time for context size {buf_inputs.size(0) if buf_inputs is not None else ''}: ", end - start)
    model.net.train(status)
    if model.np_head is not None:
        model.np_head.train(status)
    if model.args.viz_modulewise_pred and not dummy_test  and len(dataset.test_loaders) > 1 and not last:
        # store_dict_as_df(task_to_module_entropies,  incremental_step=len(dataset.test_loaders)-1, store_dir='./task_to_module_uncertainty', dataset=model.args.dataset)
        # store_dict_as_df(task_to_module_ranking, metric='Module ranking',  incremental_step=len(dataset.test_loaders)-1, store_dir='./task_to_module_uncertainty', dataset=model.args.dataset)
        store_dict_as_df(task_to_module_energy, metric='Energy', incremental_step=len(dataset.test_loaders)-1, store_dir='./task_to_module_uncertainty', dataset=model.args.dataset)
        # store_dict_as_df(task_to_module_var_entropy, metric='Variance of entropy', incremental_step=len(dataset.test_loaders)-1, store_dir='./task_to_module_uncertainty', dataset=model.args.dataset)
        # store_dict_as_df(task_to_module_var_softmax, metric='Variance of softmax', incremental_step=len(dataset.test_loaders)-1, store_dir='./task_to_module_uncertainty', dataset=model.args.dataset)

    # print(f"\nMatch between entropy and entropy var: {num_matching_indices_a}, between entropy and softmax var: {num_matching_indices_b}, between entropy var and softmax var: {num_matching_indices_c}")
    # print(f"\nTotal matching indices = {num_matching_indices} out of {total}, i.e., {num_matching_indices / (total*len(dataset.test_loaders)) * 100}")
    return accs, accs_mask_classes,

def get_module_ranking(entropies):
    ascending_sorted_indices = 1. + torch.argsort(torch.argsort(entropies, dim=1), dim=1)
    return ascending_sorted_indices.transpose(0,1)

def evaluate_ood(model: ContinualModel, dataset: ContinualDataset, ood_test_loader: torch.utils.data.DataLoader, last=False, dummy_test=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    get_ood_scores(ood_test_loader, model, dataset, in_dist=True)
    get_ood_scores(ood_test_loader, model, dataset, in_dist=False)

def get_ood_scores_helper(out_conf_score, in_conf_score, _score, _right_score, _wrong_score, in_dist, use_xent, model, test_loader, dataset):
    for data in test_loader:
        with torch.no_grad():
            inputs, labels = data
            inputs = inputs.to(model.device)
            if model.args.np_type:
                buf_inputs = torch.rand_like(inputs)
                buf_labels = torch.randint(0, model.net.num_classes, size=(buf_inputs.size(0),)).to(model.device)
                buf_labels_one_hot = F.one_hot(buf_labels.view(-1), num_classes=model.net.num_classes)
                buf_task_labels = torch.randint(0, len(dataset.test_loaders), size=(buf_inputs.size(0),)).to(
                    model.device)

            context_labels = buf_labels_one_hot if not model.args.label_embed else buf_labels
            output, _ = model(inputs, buf_inputs, context_labels, buf_task_labels, dataset.TASK_TO_LABELS)

            smax = to_np(F.softmax(output, dim=1))

            if use_xent:
                _score.append(
                    to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            else:
                _score.append(-np.max(smax, axis=1))
                out_conf_score.append(np.max(smax, axis=1))

            if in_dist:
                in_conf_score.append(np.max(smax, axis=1))
                preds = np.argmax(smax, axis=1)
                targets = labels.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                if use_xent:
                    _right_score.append(
                        to_np((output.mean(1) -
                               torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(
                        to_np((output.mean(1) -
                               torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(in_conf_score).copy(), concat(_score).copy(), concat(_right_score).copy(), concat(
            _wrong_score).copy()
    else:
        return concat(out_conf_score).copy(), concat(_score).copy()

def get_ood_scores(loader, model, dataset, in_dist=False, use_xent=False):
    status = model.net.training
    model.net.eval()
    if model.np_head is not None:
        model.np_head.eval()
    accs, accs_mask_classes = [], []

    buf_inputs, buf_labels, buf_labels_one_hot, buf_task_labels = None, None, None, None
    if model.args.np_type:
        buf_data = model.buffer.get_all_data(transform=model.transform)
        if 'der' in dataset.args.model:
            buf_inputs, buf_labels, _, buf_task_labels = buf_data
        else:
            buf_inputs, buf_labels, buf_task_labels = buf_data
        buf_labels_one_hot = F.one_hot(buf_labels.view(-1), num_classes=model.net.num_classes)

    task_to_module_entropies, task_to_module_var_softmax, task_to_module_var_entropy = {}, {}, {}
    num_matching_indices, num_matching_indices_a, num_matching_indices_b , num_matching_indices_c = 0, 0 , 0, 0
    _score = []
    out_conf_score = []
    in_conf_score = []
    _right_score = []
    _wrong_score = []
    if in_dist:
        for k, test_loader in enumerate(dataset.test_loaders):
            if k < len(dataset.test_loaders) - 1:
                in_conf_score, _score, _right_score, _wrong_score = get_ood_scores_helper(out_conf_score,
                                                                                          in_conf_score,
                                                                                          _score,
                                                                                          _right_score,
                                                                                          _wrong_score,
                                                                                          in_dist,
                                                                                          use_xent,
                                                                                          model,
                                                                                          test_loader,
                                                                                          dataset)
        # num_right = len(_right_score)
        # num_wrong = len(_wrong_score)
        # print('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))

    else:
        out_conf_score, _score = get_ood_scores_helper(out_conf_score,
                                                          in_conf_score,
                                                          _score,
                                                          _right_score,
                                                          _wrong_score,
                                                          in_dist,
                                                          use_xent,
                                                          model,
                                                          loader,
                                                          dataset)

        return  out_conf_score, _score

def compute_uncertainties(all_outputs, metrics={'entropy', 'variance', 'energy'}):
    entropies = None
    variances_softmax = None
    variances_entropy = None
    energies = None
    if 'entropy' in metrics:
        entropies = compute_entropy(all_outputs)
    if 'variance' in metrics:
        variances_softmax, variances_entropy = compute_variance(all_outputs, entropies=entropies)
    if 'energy' in metrics:
        energies = -torch.logsumexp(all_outputs, dim=-1).mean(1)
    return entropies.mean(2), variances_softmax, variances_entropy, energies


def compute_entropy(logits):
    taskwise_uncertainties = []
    for task_id, _ in enumerate(logits):
        uncertainties = compute_shannon_entropy(logits[task_id], return_mean=False)
        taskwise_uncertainties.append(uncertainties)
    samplewise_uncertainties = torch.stack(taskwise_uncertainties).permute((0, 2, 1))
    return samplewise_uncertainties

def compute_variance(logits, entropies=None):
    logits = logits.softmax(dim=-1)
    softmax_var, entropy_var = [], []
    for sample_idx in range(logits.size(2)):
        task_wise_preds = logits[:, :, sample_idx, :]
        task_wise_vars = torch.var(task_wise_preds, dim=1)
        # print(logits.shape, task_wise_preds.shape, task_wise_vars.shape); exit(1)
        softmax_var.append(task_wise_vars.mean(1))
        if entropies is not None:
            taskwise_entropy = entropies[:, sample_idx, :]
            taskwise_entropy_var = taskwise_entropy.var(-1)
            entropy_var.append(taskwise_entropy_var)
    softmax_var = torch.stack(softmax_var).transpose(0, 1)
    entropy_var = torch.stack(entropy_var).transpose(0, 1)
    return softmax_var, entropy_var

