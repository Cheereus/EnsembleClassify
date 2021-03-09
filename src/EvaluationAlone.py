import joblib
from ReadData import data_to_csv
import numpy as np
from Metrics import ARI, NMI
from Utils import get_color, draw_scatter
from Config import dimension_reduction_methods, cluster_methods


def calc_all_evaluate(dataset):

    labels_true = joblib.load('datasets/' + dataset + '_labels.pkl')
    n_cm, n_dr = len(cluster_methods), len(dimension_reduction_methods)
    ARI_matrix = np.zeros((n_cm, n_dr))
    NMI_matrix = np.zeros((n_cm, n_dr))

    for i in range(n_cm):
        for j in range(n_dr):
            labels_pred = joblib.load('labels_pred/' + dataset + '/' + dimension_reduction_methods[j] + cluster_methods[i] + '.pkl')
            ari = ARI(labels_true, labels_pred)
            nmi = NMI(labels_true, labels_pred)
            # 计算并输出评价指标
            # print('-------------')
            # print(dataset, dimension_reduction_methods[j], cluster_methods[i])
            # print('ARI:', ari)
            # print('NMI:', nmi)
            ARI_matrix[i, j] = ari
            NMI_matrix[i, j] = nmi

    joblib.dump(ARI_matrix, 'evaluation/' + dataset + '_ARI_matrix.pkl')
    joblib.dump(NMI_matrix, 'evaluation/' + dataset + '_NMI_matrix.pkl')

    data_to_csv(ARI_matrix, 'evaluation/' + dataset + '_ARI_matrix.csv')
    data_to_csv(NMI_matrix, 'evaluation/' + dataset + '_NMI_matrix.csv')


if __name__ == '__main__':
    dataset_name = 'PBMC'
    calc_all_evaluate(dataset_name)
