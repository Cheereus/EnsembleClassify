import joblib
import numpy as np
from Distance import RelevanceMatrix
from Utils import get_color, draw_scatter
from Clustering import hca, hca_dendrogram, hca_labels
from Metrics import ARI

dataset_name = 'PBMC'

data = joblib.load('ae_output/ae_dim_data_99.pkl')
# data = joblib.load('outputs/' + dataset_name + '_pca_20.pkl')
labels_true = joblib.load('datasets/' + dataset_name + '_labels.pkl')

# hca training and predict
model = hca(data)
labels_pred = hca_labels(model, 6)

rel_mat = RelevanceMatrix(labels_pred)
joblib.dump(rel_mat, 'rel_mat/' + dataset_name + '_ae_HCA.pkl')
print('ARI:', ARI(labels_true, labels_pred))
# hca_dendrogram(model)

# draw
x = [i[0] for i in data]
y = [i[1] for i in data]
default_colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k']
colors = get_color(labels_pred, default_colors)
draw_scatter(x, y, labels_pred, colors)
