import joblib
import numpy as np
from Clustering import k_means
from Clustering import hca, hca_dendrogram, hca_labels
from Metrics import ARI, NMI


def cluster_from_pred_rel(dataset):

    rel_pred = joblib.load('rel_mat/' + dataset + '/' + '_DNN_Pred_Proba.pkl')
    labels_true = joblib.load('datasets/' + dataset + '_labels.pkl')

    labels_pred = k_means(rel_pred, 6)

    # hca training and predict
    # model = hca(rel_pred)
    # labels_pred = hca_labels(model, 6)

    print('Final Clustering Finished')
    print(ARI(labels_true, labels_pred))
    print(NMI(labels_true, labels_pred))


if __name__ == '__main__':
    dataset_name = 'Chu_cell_type'
    cluster_from_pred_rel(dataset_name)
