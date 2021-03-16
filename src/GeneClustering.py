import joblib
from Clustering import k_means
from Distance import RelevanceMatrix
from tqdm import trange

dataset = 'PBMC'

data = joblib.load('datasets/PBMC.pkl').T

genes = data.shape[0]

for k in trange(2, 50):
    labels = k_means(data, k)
    rel = RelevanceMatrix(labels)
    joblib.dump(rel, 'gene_rel/' + dataset + '/k_means_' + str(k) + '.pkl')

