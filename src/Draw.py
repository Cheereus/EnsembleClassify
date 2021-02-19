import joblib
import numpy as np
from Utils import get_color, draw_scatter
import matplotlib
import matplotlib.pyplot as plt
from Config import dimension_reduction_methods, cluster_methods
from DimensionReduction import get_umap, t_SNE


def draw_scatter_2d_from_pred_rel(dataset):
    default_colors = ['b', 'g', 'r', 'm', 'y', 'k', 'c']
    data = joblib.load('rel_mat/' + dataset + '/' + '_DNN_Pred_Proba.pkl')
    # dim_data = t_SNE(data, 2)
    dim_data = get_umap(data, 2)
    x = [i[0] for i in dim_data]
    y = [i[1] for i in dim_data]
    labels = joblib.load('datasets/' + dataset + '_labels.pkl')
    colors = get_color(labels, default_colors)
    draw_scatter(x, y, labels, colors, title='True Label with pred rel, ' + dataset, xlabel='UMAP-1', ylabel='UMAP-2')

    labels_pred = joblib.load('labels_pred/' + dataset + '/' + 'Final_Pred.pkl')
    colors = get_color(labels_pred, default_colors)
    draw_scatter(x, y, labels_pred, colors, title='Predict Label with pred rel, ' + dataset, xlabel='UMAP-1', ylabel='UMAP-2')


def draw_scatter_2d_from_tSNE(dataset):
    default_colors = ['b', 'g', 'r', 'm', 'y', 'k', 'c']
    dim_data = joblib.load('dim_data/' + dataset + '/_tSNE_.pkl')
    x = [i[0] for i in dim_data]
    y = [i[1] for i in dim_data]

    labels = joblib.load('datasets/' + dataset + '_labels.pkl')
    colors = get_color(labels, default_colors)
    draw_scatter(x, y, labels, colors, title='True Label with t-SNE, ' + dataset, xlabel='tSNE-1', ylabel='tSNE-2')

    labels_pred = joblib.load('labels_pred/' + dataset + '/' + 'Final_Pred.pkl')
    colors = get_color(labels_pred, default_colors)
    draw_scatter(x, y, labels_pred, colors, title='Predict Label with t-SNE, ' + dataset, xlabel='tSNE-1', ylabel='tSNE-2')


def draw_scatter_2d_from_UMAP(dataset):
    default_colors = ['b', 'g', 'r', 'm', 'y', 'k', 'c']
    dim_data = joblib.load('dim_data/' + dataset + '/_UMAP_.pkl')
    x = [i[0] for i in dim_data]
    y = [i[1] for i in dim_data]

    labels = joblib.load('datasets/' + dataset + '_labels.pkl')
    colors = get_color(labels, default_colors)
    draw_scatter(x, y, labels, colors, title='True Label with UMAP, ' + dataset, xlabel='UMAP-1', ylabel='UMAP-2')

    labels_pred = joblib.load('labels_pred/' + dataset + '/' + 'Final_Pred.pkl')
    colors = get_color(labels_pred, default_colors)
    draw_scatter(x, y, labels_pred, colors, title='Predict Label with UMAP, ' + dataset, xlabel='UMAP-1', ylabel='UMAP-2')


def draw_bars(dataset, metric='ARI', ensemble=0.0):
    colors = [[0, 0.8, 1], [0, 0.5, 0.5], [0.2, 0.8, 0.8], [0.6, 0.8, 1], [1, 0.6, 0.8], [0.8, 0.6, 1], [1, 0.8, 0.6], [0.2, 0.4, 1]]
    data = joblib.load('evaluation/' + dataset + '_' + metric + '_matrix.pkl')
    n_cm, n_dr = data.shape
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    fig = plt.figure()
    for i in range(n_cm):
        plt.rcParams['xtick.direction'] = 'in'  # 将x轴的刻度线方向设置向内
        plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
        plt.subplot(2, 2, i + 1)
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, 1, 0.1))
        plt.grid(axis='y')
        plt.bar([i.replace('_', '') for i in dimension_reduction_methods] + ['SCEC'], list(np.ndarray.tolist(data[i])) + [ensemble], width=1, color=colors)
        plt.tick_params(labelsize=12)
        plt.legend(title=cluster_methods[i], frameon=False, fontsize=40, loc='upper left')

    plt.show()
    # 文章中需要用到矢量图
    fig.savefig('images/' + dataset + '_' + metric + '.svg', dpi=600, format='svg', bbox_inches='tight')
    # 普通图片
    fig.savefig('images/' + dataset + '_' + metric + '.png', bbox_inches='tight')


if __name__ == '__main__':
    dataset_name = 'PBMC'
    cluster_methods = ['k-means', 'AGNES', 'GMM', 'Spectral Clustering']
    # draw_scatter_2d_from_pred_rel(dataset_name)
    # draw_scatter_2d_from_tSNE(dataset_name)
    draw_scatter_2d_from_UMAP(dataset_name)
    # draw_bars(dataset_name, metric='ARI', ensemble=0.8161)
    # draw_bars(dataset_name, metric='NMI', ensemble=0.8623)
