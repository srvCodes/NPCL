import numpy as np
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from torch.distributions import Normal
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

plt.rcParams.update({'font.size': 15})

def load_pickle(fname,  dump_dir="../wts_dump"):
    full_path = f"{dump_dir}/{fname}.obj"

    with open(full_path, "rb") as f:
        obj = pickle.load(f)
        f.close()

    return obj

def sample_taskwise_dists(taskwise_dists, n_samples):
    task_to_sample_dict = {t: [] for t in taskwise_dists}
    for t, (mu, sigma) in taskwise_dists.items():
        dist = Normal(mu, sigma)
        for _ in range(n_samples):
            z = dist.rsample()
            task_to_sample_dict[t].append(z.cpu().numpy())
    return task_to_sample_dict

def viz_tsne(taskwise_dists, nsamples=800, plot_dim=3):
    task_to_samples = sample_taskwise_dists(taskwise_dists, nsamples)
    all_zs =  []
    for t, samples in task_to_samples.items():
        all_zs.extend(samples)
    all_zs = np.array(all_zs)
    # result = TSNE(n_components=plot_dim, learning_rate='auto', early_exaggeration=12).fit_transform(all_zs)
    result =  PCA(n_components=plot_dim).fit_transform(all_zs)
    col_labels = [[t] * nsamples for t in task_to_samples]
    col_labels = [label for each in col_labels for label in each]
    X = {
        'x': result[:, 0],
        'y': result[:, 1],
        'Task': col_labels
    }
    if plot_dim == 3:
        X['z'] = result[:, 2]
    data = pd.DataFrame(data=X)

    # plt.figure()
    sns.set(style="white", color_codes=True)
    if plot_dim == 2:
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="x", y="y",
            hue="Task",
            palette=sns.color_palette("hls", 10),
            data=data,
            legend="full",
            alpha=0.8
        )
    elif plot_dim == 3:
        ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
        sc = ax.scatter(
            xs=data.loc[:]["x"],
            ys=data.loc[:]["y"],
            zs=data[:]["z"],
            c=data.loc[:]["Task"],
            cmap='tab10',
        )
        ax.legend([each for each in taskwise_dists])
        plt.legend(*sc.legend_elements(), loc=2, bbox_to_anchor=(1.05, 1), ncol=1)
    else:
        raise NotImplementedError("Select correct tsne plot dim - 2d or 3d!")
    plt.savefig(f'tsne_viz/{plot_dim}D_plot_kl_warmup_residual.png', pad_inches=0)
    plt.show()
    plt.close()


if __name__ == "__main__":
    taskwise_dists = load_pickle("seq-cifar100_task_9_10_tasks_10_cls_per_task_tdist")
    viz_tsne(taskwise_dists, plot_dim=2)
    pass