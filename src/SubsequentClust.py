import joblib
import numpy as np
from Clustering import k_means
from Clustering import hca, hca_dendrogram, hca_labels
from Metrics import ARI


dataset_name = 'PBMC'

rel_pred = joblib.load('rel_mat/' + dataset_name + '/' + dataset_name + '_Rel_Pred_Proba.pkl')
labels_true = joblib.load('datasets/' + dataset_name + '_labels.pkl')

labels_pred = k_means(rel_pred, 6)

# hca training and predict
# model = hca(rel_pred)
# labels_pred = hca_labels(model, 6)

print(ARI(labels_true, labels_pred))
