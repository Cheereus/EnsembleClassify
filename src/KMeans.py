import joblib
import numpy as np
from Distance import RelevanceMatrix
from Clustering import k_means
from Metrics import ARI
from Utils import get_color, draw_scatter

dataset_name = 'PBMC'

data = joblib.load('outputs/' + dataset_name + '_FA_20.pkl')
labels_true = joblib.load('datasets/' + dataset_name + '_labels.pkl')
# data = joblib.load('ae_output/ae_dim_data_99.pkl')

labels_pred = k_means(data, 6)

print(ARI(labels_true, labels_pred))
print(labels_true, labels_pred)

rel_mat = RelevanceMatrix(labels_pred)
joblib.dump(rel_mat, 'rel_mat/' + dataset_name + '_FA_kmeans.pkl')

# get color list based on labels
colors = get_color(labels_pred)
print(colors)

# get two coordinates
x = [i[0] for i in data]
y = [i[1] for i in data]
# draw
draw_scatter(x, y, labels_pred, colors)
