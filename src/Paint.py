import joblib
import matplotlib.pyplot as plt
from DimensionReduction import get_umap, t_SNE
from Utils import get_color


def paint_together(dataset, n_clusters, value_type='_UMAP_'):

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


if __name__ == '__main__':
    paint_together('Chu_cell_type', 7, '_tSNE_')
