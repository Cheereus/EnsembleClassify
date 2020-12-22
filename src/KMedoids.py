import joblib
import numpy as np
from Distance import GetDistanceMatrix, RelevanceMatrix
from Clustering import kMedoids
from Metrics import ARI
from Utils import get_color, draw_scatter

dataset_name = 'PBMC'

data = joblib.load('outputs/' + dataset_name + '_pca_20.pkl')
labels_true = joblib.load('datasets/' + dataset_name + '_labels.pkl')
# data = joblib.load('ae_output/ae_dim_data_99.pkl')
dis = GetDistanceMatrix(data, 'euclidean')
print(len(labels_true))

M, C = kMedoids(dis, 6, tmax=1000)

print('clustering result:')
labels_pred = np.zeros((len(data)))
for label in C:
    for point_idx in C[label]:
        # print('label {0}:　{1}'.format(label, point_idx))
        labels_pred[point_idx] = label

print(ARI(labels_true, labels_pred))
print(labels_true, labels_pred)

rel_mat = RelevanceMatrix(labels_pred)
joblib.dump(rel_mat, 'rel_mat/' + dataset_name + '_AE_euc_kmedoids.pkl')

# get color list based on labels
colors = get_color(labels_pred)
print(colors)

# get two coordinates
x = [i[0] for i in data]
y = [i[1] for i in data]
# draw
draw_scatter(x, y, labels_pred, colors)
