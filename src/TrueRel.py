import joblib
from Distance import RelevanceMatrix

dataset_name = 'PBMC'
labels_true = joblib.load('datasets/' + dataset_name + '_labels.pkl')

rel_mat = RelevanceMatrix(labels_true)
joblib.dump(rel_mat, 'rel_mat/' + dataset_name + '/_True.pkl')