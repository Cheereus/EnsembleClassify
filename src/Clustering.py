import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from Decorator import time_indicator


@time_indicator
def k_means(X, k):
    k_m_model = KMeans(n_clusters=k, max_iter=300, n_init=40, init='k-means++', n_jobs=-1)
    k_m_model.fit(X)
    return k_m_model.labels_.tolist()


@time_indicator
def knn(X, y, k):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X, y)
    return knn_model


@time_indicator
def hca(X, k=None):
    hca_model = linkage(X, 'ward')
    return hca_model


@time_indicator
# dendogram for hca
def hca_dendrogram(model):
    plt.figure(figsize=(50, 10))
    dendrogram(model, leaf_rotation=90., leaf_font_size=8)
    plt.show()


@time_indicator
# labels of hca
def hca_labels(model, n_clusters):
    labels = fcluster(model, n_clusters, criterion='maxclust')
    return labels
