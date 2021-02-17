import joblib
import numpy as np
from tqdm import trange


def get_rel_from_pred(dataset):

    y_pred = joblib.load('labels_pred/' + dataset + '/_DNN_Pred_Proba.pkl')
    y_true_index = joblib.load('train_data/' + dataset + '_y_index.pkl')
    # load true labels
    rel_true = joblib.load('rel_mat/' + dataset + '/' + '_True.pkl')
    n_samples = rel_true.shape[0]

    rel_pred = np.ones((n_samples, n_samples))

    idx = 0
    # get predict relevance matrix
    for i in trange(n_samples):
        for j in range(i+1, n_samples):
            rel_pred[i, j] = y_pred[y_true_index[idx]]
            rel_pred[j, i] = y_pred[y_true_index[idx]]
            idx += 1

    joblib.dump(rel_pred, 'rel_mat/' + dataset + '/' + '_DNN_Pred_Proba.pkl')
    print('Predicted Relevance Matrix Saved')


if __name__ == '__main__':
    dataset_name = 'Chu_cell_type'
    get_rel_from_pred(dataset_name)
