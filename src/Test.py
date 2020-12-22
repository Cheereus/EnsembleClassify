import joblib
from DimensionReduction import get_pca, t_SNE
from Clustering import k_means, hca, hca_labels
from Metrics import ARI
from Decorator import time_indicator
from Utils import get_color, draw_scatter

dataset_name = 'PBMC'


X = joblib.load('ae_output/ae_dim_data_99.pkl')
labels = joblib.load('datasets/' + dataset_name + '_labels.pkl')

print(X.shape)

# t-SNE
tSNE_data = t_SNE(X, dim=2, with_normalize=True, perp=5)

# joblib.dump(tSNE_data, 'outputs/' + dataset_name + '_tSNE_2.pkl')

# get color list based on labels
colors = get_color(labels)
print(colors)

# get two coordinates
x = [i[0] for i in tSNE_data]
y = [i[1] for i in tSNE_data]
# draw
draw_scatter(x, y, labels, colors)
