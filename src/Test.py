import joblib
from DimensionReduction import get_pca, t_SNE, feature_agglomeration
from Clustering import k_means, hca, hca_labels
from Metrics import ARI
from Decorator import time_indicator
from Utils import get_color, draw_scatter

dataset_name = 'PBMC'


X = joblib.load('datasets/' + dataset_name + '.pkl')
labels = joblib.load('datasets/' + dataset_name + '_labels.pkl')

print(X.shape)

# t-SNE
dim_data = t_SNE(X, dim=2, with_normalize=True, perp=5)

# FA
# dim_data = feature_agglomeration(X, dim=20)

joblib.dump(dim_data, 'outputs/' + dataset_name + '_tSNE_2_perp5.pkl')

# get color list based on labels
colors = get_color(labels)
# print(colors)

# get two coordinates
x = [i[0] for i in dim_data]
y = [i[1] for i in dim_data]
# draw
draw_scatter(x, y, labels, colors)
