import joblib
from DimensionReduction import *
from Clustering import k_means, hca, hca_labels
from Metrics import ARI
from Decorator import time_indicator
from Utils import get_color, draw_scatter

dataset_name = 'PBMC'


X = joblib.load('datasets/' + dataset_name + '.pkl')
labels = joblib.load('datasets/' + dataset_name + '_labels.pkl')

print(X.shape)

dim_data = get_Isomap(X, dim=20, with_normalize=True)
print(dim_data.shape)

# get color list based on labels
colors = get_color(labels)
# print(colors)

# get two coordinates
x = [i[0] for i in dim_data]
y = [i[1] for i in dim_data]
# draw
draw_scatter(x, y, labels, colors)
