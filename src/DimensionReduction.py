import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.cluster import FeatureAgglomeration
from Decorator import time_indicator


# Normalize
def get_normalize(data):
    preprocess = Normalizer()
    X = preprocess.fit_transform(data)
    return X


# t-SNE
@time_indicator
def t_SNE(data, dim=2, perp=30, with_normalize=False):
    if with_normalize:
        data = get_normalize(data)

    data = np.array(data)
    tsne = TSNE(n_components=dim, init='pca', perplexity=perp, method='exact')
    tsne.fit_transform(data)

    return tsne.embedding_


# PCA
@time_indicator
def get_pca(data, c=3, with_normalize=False):
    if with_normalize:
        data = get_normalize(data)

    pca_result = PCA(n_components=c)
    pca_result.fit(data)
    newX = pca_result.fit_transform(data)

    return newX, pca_result.explained_variance_ratio_, pca_result


# Feature Agglomeration
@time_indicator
def feature_agglomeration(data, dim=2, with_normalize=False):
    if with_normalize:
        data = get_normalize(data)

    agglo = FeatureAgglomeration(n_clusters=dim)
    agglo.fit(data)
    data_reduced = agglo.transform(data)

    return data_reduced
