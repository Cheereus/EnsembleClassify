import joblib
import numpy as np
from tqdm import trange

dataset = 'PBMC'

y_pred = joblib.load('rel_mat/PBMC_Y_Pred_Proba.pkl')
y_true_index = joblib.load('train_data/PBMC_y_index.pkl')
# load true labels
rel_true = joblib.load('rel_mat/' + dataset + '/' + dataset + '_True.pkl')
n_samples = rel_true.shape[0]

rel_pred = np.ones((n_samples, n_samples))

idx = 0
# get predict relevance matrix
for i in trange(n_samples):
    for j in range(i+1, n_samples):
        rel_pred[i, j] = y_pred[y_true_index[idx]]
        rel_pred[j, i] = y_pred[y_true_index[idx]]
        idx += 1

joblib.dump(rel_pred, 'rel_mat/' + dataset + '/' + dataset + '_Rel_Pred_Proba.pkl')