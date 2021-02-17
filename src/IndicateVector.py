import joblib
import numpy as np
from Config import dimension_reduction_methods, cluster_methods
from tqdm import trange
from Decorator import time_indicator


# TODO 本部分非常消耗内存
@time_indicator
def get_indicate_vector(dataset):

    rel_mats = []
    ind_vectors = []
    y_true = []
    y_true_index = []

    # load true labels
    rel_true = joblib.load('rel_mat/' + dataset + '/_True.pkl')
    n_samples = rel_true.shape[0]

    # load relevance matrix
    for dr in dimension_reduction_methods:
        for cm in cluster_methods:
            rel_mats.append(joblib.load('rel_mat/' + dataset + '/' + dr + cm + '.pkl'))

    y_idx = 0
    # get indicate vector
    for i in trange(n_samples):
        for j in range(i+1, n_samples):
            vec = []
            for rel_mat in rel_mats:
                vec.append(rel_mat[i, j])
            ind_vectors.append(vec)
            y_true.append(rel_true[i, j])
            y_true_index.append(y_idx)
            y_idx += 1
            # print(i, j)

    ind_vectors = np.array(ind_vectors)
    print(ind_vectors.shape)

    joblib.dump(ind_vectors, 'train_data/' + dataset + '_indicator.pkl')
    joblib.dump(y_true, 'train_data/' + dataset + '_labels.pkl')
    joblib.dump(y_true_index, 'train_data/' + dataset + '_y_index.pkl')


if __name__ == '__main__':
    dataset_name = 'Chu_cell_type'
    get_indicate_vector(dataset_name)
