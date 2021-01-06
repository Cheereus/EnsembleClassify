import joblib
import numpy as np
from Config import dimension_reduction, cluster_method

dataset = 'PBMC'

n_samples = 3694

rel_mats = []
ind_vectors = []

# load relevance matrix
for dr in dimension_reduction:
    for cm in cluster_method:
        print(dr, cm)
        rel_mats.append(joblib.load('rel_mat/' + dataset + '/' + dataset + dr + cm + '.pkl'))

# get indicate vector
for i in range(n_samples):
    for j in range(i+1, n_samples):
        vec = []
        for rel_mat in rel_mats:
            vec.append(rel_mat[i, j])
        ind_vectors.append(vec)
        print(i, j)

ind_vectors = np.array(ind_vectors)
print(ind_vectors.shape)