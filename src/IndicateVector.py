import joblib
import numpy as np
from Config import dimension_reduction, cluster_method
from datetime import datetime

start_time = datetime.now()

dataset = 'PBMC'
n_samples = 3694

rel_mats = []
ind_vectors = []
y_true = []

# load true labels
rel_true = joblib.load('rel_mat/' + dataset + '/' + dataset + '_True.pkl')

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

end_time = datetime.now()
ind_vectors = np.array(ind_vectors)

print(ind_vectors.shape)
print(start_time, end_time)

joblib.dump(ind_vectors, 'train_data/' + dataset + '_indicator.pkl')
joblib.dump(y_true, 'train_data/' + dataset + '_labels.pkl')
