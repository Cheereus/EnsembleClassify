import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import joblib


# 关联矩阵
def RelevanceMatrix(labels):

    n_samples = len(labels)
    rm = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if labels[i] == labels[j]:
                rm[i, j] = 1
                rm[j, i] = 1
    return rm


# 距离矩阵
def GetDistanceMatrix(data, metric):
    if metric == 'euclidean' or metric == 'cosine':
        return pairwise_distances(data, metric=metric)
