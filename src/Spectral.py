import joblib
import numpy as np
from Distance import RelevanceMatrix
from Utils import get_color, draw_scatter
from Metrics import ARI
from sklearn.cluster import SpectralClustering

dataset_name = 'PBMC'

# data = joblib.load('ae_output/ae_dim_data_99.pkl')
# data = joblib.load('outputs/' + dataset_name + '_pca_20.pkl')
data = joblib.load('outputs/' + dataset_name + '_tSNE_2.pkl')
labels_true = joblib.load('datasets/' + dataset_name + '_labels.pkl')

SC_cluster = SpectralClustering(n_clusters=6).fit(data)

labels_pred = SC_cluster.labels_

rel_mat = RelevanceMatrix(labels_pred)
joblib.dump(rel_mat, 'rel_mat/' + dataset_name + '_tSNE_Spectral.pkl')
print('ARI:', ARI(labels_true, labels_pred))

# draw
x = [i[0] for i in data]
y = [i[1] for i in data]
default_colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k']
colors = get_color(labels_pred, default_colors)
draw_scatter(x, y, labels_pred, colors)
