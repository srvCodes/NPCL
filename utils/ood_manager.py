import torch
import numpy as np
from datasets.seq_cifar10 import TCIFAR10
from datasets.seq_cifar100 import TCIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.conf import base_path_dataset as base_path
from datasets.utils.continual_dataset import seed_worker
import matplotlib.pyplot as plt
import sklearn.metrics as sk

recall_level_default = 0.95


def plot_image(test_loader, test_dataset):
    images = test_dataset.data
    labels = test_dataset.targets
    for idx, img in enumerate(images):
        # dataiter = iter(test_loader)
        # images, labels = dataiter.next()
        print(idx, labels[idx], img.shape)
        plt.imshow(img)
        plt.show()

    print(images.shape, labels.shape);
    exit(1)


def get_ood_test_loader(args):
    ood_dataset = args.ood_dataset
    if ood_dataset == 'cifar10':
        normalization_transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                       (0.2470, 0.2435, 0.2615))
        cls_upper, cls_lower = 10, 0
        test_dset = TCIFAR10
    elif ood_dataset == 'cifar100':
        normalization_transform = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                       (0.2675, 0.2565, 0.2761))
        cls_upper, cls_lower = 100, 0
        test_dset = TCIFAR100
    else:
        raise NotImplementedError("ood dataset can be either cifar100 or cifar10")

    test_transform = transforms.Compose(
        [transforms.ToTensor(), normalization_transform])

    test_dataset = test_dset(base_path() + ood_dataset.upper(), train=False,
                             download=True, transform=test_transform)
    test_mask = np.logical_and(np.array(test_dataset.targets) >= cls_lower,
                               np.array(test_dataset.targets) < cls_upper)
    test_dataset.data = test_dataset.data[test_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]
    # plot_image(test_dataset)
    g = torch.Generator()
    g.manual_seed(0)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size, shuffle=False, num_workers=4,
                             worker_init_fn=seed_worker,
                             generator=g,
                             )
    return test_loader

"""
Source: https://github.com/nazim1021/OOD-detection-using-OECC/blob/master/utils/display_results.py
"""
def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def get_measures(_pos, _neg, recall_level=recall_level_default):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1
    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def print_measures(auroc, aupr, fpr, f, method_name='Ours', recall_level=recall_level_default):

    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))
    f.write('\nFPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    f.write('\nAUROC: \t\t\t{:.2f}'.format(100 * auroc))
    f.write('\nAUPR:  \t\t\t{:.2f}'.format(100 * aupr))


def print_measures_with_std(aurocs, auprs, fprs, f, method_name='Ours', recall_level=recall_level_default):

    print('FPR{:d}:\t\t\t{:.2f}\t+/- {:.2f}'.format(int(100 * recall_level), 100 * np.mean(fprs), 100 * np.std(fprs)))
    print('AUROC: \t\t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(aurocs), 100 * np.std(aurocs)))
    print('AUPR:  \t\t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(auprs), 100 * np.std(auprs)))
    f.write('FPR{:d}:\t\t\t{:.2f}\t+/- {:.2f}'.format(int(100 * recall_level), 100 * np.mean(fprs), 100 * np.std(fprs)))
    f.write('AUROC: \t\t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(aurocs), 100 * np.std(aurocs)))
    f.write('AUPR:  \t\t\t{:.2f}\t+/- {:.2f}'.format(100 * np.mean(auprs), 100 * np.std(auprs)))

