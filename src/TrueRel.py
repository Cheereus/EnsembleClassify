import joblib
from Distance import RelevanceMatrix


def get_true_rel_mat(dataset):

    labels_true = joblib.load('datasets/' + dataset + '_labels.pkl')

    rel_mat = RelevanceMatrix(labels_true)
    # print(sum(rel_mat))
    joblib.dump(rel_mat, 'rel_mat/' + dataset + '/_True.pkl')


if __name__ == '__main__':
    dataset_name = 'Chu_cell_type'
    get_true_rel_mat(dataset_name)
