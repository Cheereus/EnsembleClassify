import joblib
import numpy as np
from Config import dimension_reduction, cluster_method

dataset = 'PBMC'

rel_mats = []
ind_vectors = []
y_true = []

# load true labels
rel_true = joblib.load('rel_mat/' + dataset + '/' + dataset + '_True.pkl')
n_samples = rel_true.shape[0]

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
        y_true.append(rel_true[i, j])
        print(i, j)

ind_vectors = np.array(ind_vectors)
print(ind_vectors.shape)

joblib.dump(ind_vectors, 'train_data/' + dataset + '_indicator.pkl')
joblib.dump(y_true, 'train_data/' + dataset + '_labels.pkl')
