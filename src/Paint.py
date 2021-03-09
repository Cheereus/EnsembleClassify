import joblib
import matplotlib.pyplot as plt
from Utils import get_color
import pandas as pd
import numpy as np


def paint_scatter_together(dataset, n_clusters, value_type='_UMAP_'):
    from DimensionReduction import get_umap, t_SNE, get_pca
    dim_data_original = joblib.load('datasets/' + dataset + '.pkl')
    # dim_data_original = joblib.load('dim_data/' + dataset + '/' + value_type + '.pkl')
    dim_data_incidence = joblib.load('rel_mat/' + dataset + '/' + '_DNN_Pred_Proba.pkl')
    label_true = joblib.load('datasets/' + dataset + '_labels.pkl')
    label_pred = joblib.load('labels_pred/' + dataset + '/' + 'Final_Pred.pkl')

    label_true = get_color(label_true, range(n_clusters))
    label_pred = get_color(label_pred, range(n_clusters))

    if value_type == '_UMAP_':
        dim_data_original = get_umap(dim_data_original, 2)
        dim_data_incidence = get_umap(dim_data_incidence, 2)
    if value_type == '_tSNE_':
        dim_data_original = t_SNE(dim_data_original, 2, with_normalize=True)
        dim_data_incidence = t_SNE(dim_data_incidence, 2)
    if value_type == '_PCA_':
        dim_data_original, _1, _2 = get_pca(dim_data_original, 2, with_normalize=True)
        dim_data_incidence, _1, _2 = get_pca(dim_data_incidence, 2)

    fig = plt.figure()

    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.subplot(2, 2, 1)
    plt.scatter(dim_data_original[:, 0], dim_data_original[:, 1], c=label_true, cmap='rainbow', s=1)
    plt.gca().set_aspect('equal', 'datalim')
    plt.ylabel('Original Data')

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.subplot(2, 2, 2)
    plt.scatter(dim_data_original[:, 0], dim_data_original[:, 1], c=label_pred, cmap='rainbow', s=1)
    plt.gca().set_aspect('equal', 'datalim')

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.subplot(2, 2, 3)
    plt.scatter(dim_data_incidence[:, 0], dim_data_incidence[:, 1], c=label_true, cmap='rainbow', s=1)
    plt.gca().set_aspect('equal', 'datalim')
    plt.xlabel('True Label')
    plt.ylabel('Incidence Matrix')

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.subplot(2, 2, 4)
    plt.scatter(dim_data_incidence[:, 0], dim_data_incidence[:, 1], c=label_pred, cmap='rainbow', s=1)
    plt.gca().set_aspect('equal', 'datalim')
    plt.xlabel('Predict Label')
    plt.show()

    fig.savefig('images/' + dataset + value_type + 'all_together.svg', dpi=600, format='svg', bbox_inches='tight')


def paint_bars_together(dataset, ensemble=None):
    df = [1, 2]
    colors = ['#DC143C', '#800080', '#0000FF', '#00BFFF',
              '#7FFFAA', '#D2691E', '#FF4500', '#000000']

    source = 'evaluation/'
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    bar_width = 0.1
    x = np.arange(4)
    if ensemble is None:
        ensemble = [0.0, 0.0]

    fig = plt.figure()
    for i in range(2):
        plt.subplot(2, 1, (i % 2) + 1)
        if i == 0:
            path = (source + dataset + '_ARI_matrix.csv')
            plt.ylabel("ARI")
        else:
            path = (source + dataset + '_NMI_matrix.csv')
            plt.ylabel("NMI")
        df[i] = pd.read_csv(path, header=None)
        df[i] = pd.DataFrame(df[i])
        df[i].insert(7, '7', ensemble[i])
        pd.DataFrame(df[i])
        df[i].columns = ['tSNE', 'PCA', 'FA', 'UMAP', 'LLE', 'MDS', 'Isomap', 'SCEC']
        df[i].index = ['K-Means', 'AGNES', 'GMM', 'Spectral Clustering']
        for j in range(df[i].shape[1]):  # 每个df的列
            plt.ylim(0, 1)
            plt.bar(x + bar_width * j, df[i][df[i].columns[j]], bar_width,
                    align="center", color=colors[j], label=df[i].columns[j], alpha=0.7)
        plt.grid(axis='y', linewidth=0.3, linestyle='dashed')
        plt.xticks(x + bar_width * 7 / 2, df[i].index)
        if i == 0:
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
            plt.subplots_adjust(hspace=0.1)
        else:
            plt.legend(loc=2, bbox_to_anchor=(1.05, 2.1), borderaxespad=0.)
            plt.show()

    # 文章中需要用到矢量图
    fig.savefig('images/' + dataset + '_bars_together.svg', dpi=600, format='svg', bbox_inches='tight')
    # 普通图片
    fig.savefig('images/' + dataset + '_bars_together.png', bbox_inches='tight')


if __name__ == '__main__':
    # paint_scatter_together('Chu_cell_type', 7, '_tSNE_')
    paint_bars_together(dataset='Chu_cell_type', ensemble=[0.8466, 0.8993])
    # paint_bars_together(dataset='PBMC', ensemble=[0.8640, 0.7430])
