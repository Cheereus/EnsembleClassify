import joblib
import numpy as np
from Clustering import k_means
from Clustering import hca, hca_dendrogram, hca_labels
from Metrics import ARI, NMI


def cluster_from_pred_rel(dataset, n_clusters):

    rel_pred = joblib.load('rel_mat/' + dataset + '/' + '_DNN_Pred_Proba.pkl')
    labels_true = joblib.load('datasets/' + dataset + '_labels.pkl')

    labels_pred = k_means(rel_pred, n_clusters)

    joblib.dump(labels_pred, 'labels_pred/' + dataset + '/' + 'Final_Pred.pkl')

    print('Final Clustering Finished')
    print(ARI(labels_true, labels_pred))
    print(NMI(labels_true, labels_pred))


if __name__ == '__main__':
    dataset_name = 'PBMC'
    cluster_from_pred_rel(dataset_name, 6)
