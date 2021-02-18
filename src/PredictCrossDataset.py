import joblib
import numpy as np
import tensorflow as tf
from tqdm import trange
from Clustering import k_means
from Metrics import ARI, NMI


def predict_cross_dataset(dataset, reference_dataset):
    ref_model = tf.keras.models.load_model('models/' + reference_dataset)
    data = joblib.load('train_data/' + dataset + '_indicator.pkl')
    y_pred = ref_model.predict(data)
    joblib.dump(y_pred, 'pred_across_dataset/' + dataset + '/' + reference_dataset + '_DNN_Pred_Proba.pkl')


def get_rel_from_pred_across_dataset(dataset, reference_dataset):

    y_pred = joblib.load('pred_across_dataset/' + dataset + '/' + reference_dataset + '_DNN_Pred_Proba.pkl')
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

    joblib.dump(rel_pred, 'pred_across_dataset/' + dataset + '/' + reference_dataset + '_rel_mat.pkl')
    print('Predicted Relevance Matrix Saved')


def cluster_from_pred_rel_across_dataset(dataset, reference_dataset, n_clusters):

    rel_pred = joblib.load('pred_across_dataset/' + dataset + '/' + reference_dataset + '_rel_mat.pkl')
    labels_true = joblib.load('datasets/' + dataset + '_labels.pkl')

    labels_pred = k_means(rel_pred, n_clusters)

    joblib.dump(labels_pred, 'pred_across_dataset/' + dataset + '/' + reference_dataset + 'Final_Pred.pkl')

    print('Final Clustering Finished')
    print(ARI(labels_true, labels_pred))
    print(NMI(labels_true, labels_pred))


if __name__ == '__main__':
    dataset_name = 'Yan_human'
    ref_dataset = 'Chu_cell_type'
    predict_cross_dataset(dataset_name, ref_dataset)
    get_rel_from_pred_across_dataset(dataset_name, ref_dataset)
    cluster_from_pred_rel_across_dataset(dataset_name, ref_dataset, 8)
