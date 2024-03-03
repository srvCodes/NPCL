import numpy as np
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker

plt.rcParams.update({'font.size': 15})

def store_dict_as_df(task_to_module_unc, metric='Entropy', incremental_step=1, store_dir='./task_to_module_uncertainty', dataset='seq-cifar-100'):
    if os.path.exists(store_dir) and os.path.isdir(store_dir):
        pass
    else:
        os.mkdir(store_dir)

    df = pd.DataFrame.from_dict(task_to_module_unc).T
    if metric == "Module ranking":
        df = np.argsort(np.argsort(df, axis=1), axis=1) + 1
    plt.figure(figsize=(4+(incremental_step*2.1), 4+(incremental_step*0.9)))
    cmap='Blues'
    sns.heatmap(df, annot=True, fmt='.3f' if metric in ["Entropy"] else '.0f' if metric in ["Module ranking"] else '.2e', cmap=cmap)
    plt.xlabel("Module ID")
    plt.ylabel("Test set ID")
    plt.axis("tight")  # gets rid of white border
    plt.title(metric)
    plt.savefig(f'{store_dir}/{"_".join(metric.split(" "))}_incorrect_incr_task_{incremental_step}_{dataset}',pad_inches = 0, bbox_inches='tight')
    # plt.show()
    plt.close()

def evolving_kl_per_latent_unit(task_to_epoch_to_kl_dict, fname, to_save_path=f"train_time_kl/"):
    fig, axs = plt.subplots(
        nrows=2, ncols=len(task_to_epoch_to_kl_dict), figsize=(16, 5), facecolor="white", constrained_layout=True,
        gridspec_kw={'height_ratios': [1, 4]}
    )
    for i, task_id in enumerate(task_to_epoch_to_kl_dict):
        # if task_id != 4:
        #     continue
        for j, layer in enumerate([1, 2]):

            layer_to_group_to_epoch_dict = {}
            for epoch, epochwise_dict in task_to_epoch_to_kl_dict[task_id].items():
                if layer not in epochwise_dict:
                    continue
                layer_dict = epochwise_dict[layer]
                if type(layer_dict) is dict:
                    for t, kl in layer_dict.items():
                        if t not in layer_to_group_to_epoch_dict:
                            layer_to_group_to_epoch_dict[t] = {epoch: np.log(kl)}
                        else:
                            layer_to_group_to_epoch_dict[t].update({epoch: np.log(kl)})
                else:
                    layer_to_group_to_epoch_dict[epoch] = np.log(layer_dict)

            if len(layer_to_group_to_epoch_dict) == 0:
                continue
            cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
            try:
                df = pd.DataFrame(layer_to_group_to_epoch_dict).T
            except ValueError:
                df = pd.DataFrame(layer_to_group_to_epoch_dict, index=[0])
            s = sns.heatmap(df,  ax=axs[j, i], cmap=cmap, xticklabels = 10)
            s.set(xlabel='Train epochs', ylabel=f'L = {j}')
            # tick_locator = ticker.MaxNLocator(5)
            # axs[j,i].xaxis.set_major_locator(tick_locator)
    plt.axis("tight")  # gets rid of white border
    plt.ylabel("Layer")
    plt.savefig(f'{to_save_path}/evolving_{fname}.png', pad_inches=0)
    plt.show()
    plt.close()

def layerwise_latent_unit_viz(task_to_epoch_to_kl_dict, fname, to_save_path=f"train_time_kl/"):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), facecolor="white", constrained_layout=True,)
    layerwise_kl = {}
    for i, task_id in enumerate(task_to_epoch_to_kl_dict):
        for j, layer in enumerate([1, 2]):
            for epoch, epochwise_dict in task_to_epoch_to_kl_dict[task_id].items():

                if epoch == len(epochwise_dict) - 1:
                    if layer in epochwise_dict:
                        layer_dict = epochwise_dict[layer]
                        if type(layer_dict) is dict:
                            for t, kl in layer_dict.items():
                                if task_id in layerwise_kl:
                                    layerwise_kl[task_id].update({f'L={layer},T={t}' : np.log(kl)})
                                else:
                                    layerwise_kl[task_id] = {f'L={layer},T={t}': np.log(kl)}

                        else:
                            layerwise_kl[task_id] = {f'L={layer},G' : np.log(layer_dict)}
    df = pd.DataFrame(layerwise_kl)
    df.plot(ax=ax)
    plt.legend(title="Task timeline", prop={'size': 11})
    plt.axis("tight")  # gets rid of white border
    plt.ylabel("log KL[q|p]")
    plt.xlabel(f"Layer: L, Task: T, Global: G")
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.savefig(f'{to_save_path}/kl_by_layer_{fname}.png', pad_inches=0)
    plt.show()
    plt.close()

if __name__ == "__main__":
    fname = "npcl_task_to_epoch_to_kl_klt_0.1_klg_0.1_cutoff_2.0.pkl"
    task_to_epoch_to_kl_dict = pickle.load(open(f"train_time_kl/{fname}", "rb"))
    evolving_kl_per_latent_unit(task_to_epoch_to_kl_dict, fname)
    layerwise_latent_unit_viz(task_to_epoch_to_kl_dict, fname)