import joblib
from DimensionReduction import get_pca, t_SNE
from Clustering import k_means, hca, hca_labels
from Metrics import ARI
from Decorator import time_indicator

dataset_name = 'human_islets'

X = joblib.load('datasets/' + dataset_name + '.pkl')
labels = joblib.load('datasets/' + dataset_name + '_labels.pkl')

print(X.shape)

# PCA
pca_data, ratio, result = get_pca(X, c=20, with_normalize=True)
print(sum(ratio))
joblib.dump(pca_data, 'outputs/' + dataset_name + '_pca_20.pkl')

# t-SNE
tSNE_data = t_SNE(X, dim=2, with_normalize=True, perp=5)

joblib.dump(tSNE_data, 'outputs/' + dataset_name + '_tSNE_2.pkl')
