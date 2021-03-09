import os
import joblib
from collections import Counter


def make_dir(dataset):
    path_list = ['dim_data', 'labels_pred', 'models', 'rel_mat']
    for path in path_list:
        isExists = os.path.exists(path + '/' + dataset)
        if not isExists:
            os.makedirs(path + '/' + dataset)

    print('Directory created')


def get_n_clusters(dataset):
    labels_true = joblib.load('datasets/' + dataset + '_labels.pkl')
    count_result = Counter(labels_true)
    return len(count_result.keys())


if __name__ == '__main__':
    get_n_clusters('GSE84133')
