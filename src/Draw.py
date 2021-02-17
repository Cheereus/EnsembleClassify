import joblib
import numpy as np
from Utils import get_color, draw_scatter
import matplotlib
import matplotlib.pyplot as plt
from Config import dimension_reduction_methods, cluster_methods


def draw_scatter_2d_from_tSNE(dataset):
    default_colors = ['b', 'g', 'r', 'm', 'y', 'k', 'c']
    dim_data = joblib.load('dim_data/' + dataset + '/_tSNE_.pkl')
    x = [i[0] for i in dim_data]
    y = [i[1] for i in dim_data]

    labels = joblib.load('datasets/' + dataset + '_labels.pkl')
    colors = get_color(labels, default_colors)
    draw_scatter(x, y, labels, colors, title='True Label with t-SNE, ' + dataset)

    labels_pred = joblib.load('labels_pred/' + dataset + '/' + 'Final_Pred.pkl')
    colors = get_color(labels_pred, default_colors)
    draw_scatter(x, y, labels_pred, colors, title='Predict Label with t-SNE, ' + dataset)


def draw_scatter_2d_from_UMAP(dataset):
    default_colors = ['b', 'g', 'r', 'm', 'y', 'k', 'c']
    dim_data = joblib.load('dim_data/' + dataset + '/_UMAP_.pkl')
    x = [i[0] for i in dim_data]
    y = [i[1] for i in dim_data]

    labels = joblib.load('datasets/' + dataset + '_labels.pkl')
    colors = get_color(labels, default_colors)
    draw_scatter(x, y, labels, colors, title='True Label with UMAP, ' + dataset)

    labels_pred = joblib.load('labels_pred/' + dataset + '/' + 'Final_Pred.pkl')
    colors = get_color(labels_pred, default_colors)
    draw_scatter(x, y, labels_pred, colors, title='Predict Label with UMAP, ' + dataset)


def draw_bars(dataset, metric='ARI', ensemble=0.0):
    colors = [[0, 0.8, 1], [0, 0.5, 0.5], [0.2, 0.8, 0.8], [0.6, 0.8, 1], [1, 0.6, 0.8], [0.8, 0.6, 1], [1, 0.8, 0.6], [0.2, 0.4, 1]]
    data = joblib.load('evaluation/' + dataset + '_' + metric + '_matrix.pkl')
    n_cm, n_dr = data.shape
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    fig = plt.figure()
    for i in range(n_cm):
        plt.subplot(2, 2, i + 1)
        plt.bar([i.replace('_', '') for i in dimension_reduction_methods] + ['Ensemble'], list(np.ndarray.tolist(data[i])) + [ensemble], width=1, color=colors, alpha=0.9)
        plt.tick_params(labelsize=12)
        plt.legend(title=cluster_methods[i], frameon=False, fontsize=20)

    plt.show()
    # 文章中需要用到矢量图
    fig.savefig('images/' + dataset + '_' + metric + '.eps', dpi=600, format='eps', bbox_inches='tight')
    # 普通图片
    fig.savefig('images/' + dataset + '_' + metric + '.png', bbox_inches='tight')


if __name__ == '__main__':
    dataset_name = 'PBMC'
    cluster_methods = ['k-means', 'AGNES', 'GMM', 'Spectral Clustering']
    draw_scatter_2d_from_UMAP(dataset_name)
    # draw_bars(dataset_name, metric='ARI', ensemble=0.8640)
