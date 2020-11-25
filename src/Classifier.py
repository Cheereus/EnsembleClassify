import joblib
from DimensionReduction import get_pca, t_SNE
from Clustering import k_means, hca, hca_labels
from Metrics import ARI
from Decorator import time_indicator

X = joblib.load('datasets/PBMC.pkl')
labels = joblib.load('datasets/PBMC_labels.pkl')

print(X.shape)

# PCA
pca_data, ratio, result = get_pca(X, c=20, with_normalize=True)
print(sum(ratio))

# t-SNE
# tSNE_data = t_SNE(pca_data, dim=5, with_normalize=True)

pca_k_means_labels = k_means(pca_data, k=6)
# tSNE_labels = k_means(tSNE_data, k=6)

print(ARI(labels, pca_k_means_labels))
# print(ARI(labels, tSNE_labels))

model = hca(pca_data)
pca_hca_labels = hca_labels(model, 6)
print(ARI(labels, pca_hca_labels))

joblib.dump(pca_k_means_labels, 'temp/PBMC_pca_k_means_labels.pkl')
joblib.dump(pca_hca_labels, 'temp/PBMC_pca_hca_labels.pkl')
