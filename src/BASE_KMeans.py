import joblib
from Distance import RelevanceMatrix
from Clustering import k_means
from Metrics import ARI, NMI
from Decorator import time_indicator
from Config import dimension_reduction_methods


# TODO 修改为多线程并行
@time_indicator
def rel_mat_k_means(dataset, dr_methods, n_clusters):
    for method in dr_methods:

        data = joblib.load('dim_data/' + dataset + '/' + method + '.pkl')
        labels_true = joblib.load('datasets/' + dataset + '_labels.pkl')

        labels_pred = k_means(data, n_clusters)

        # 计算并输出评价指标
        print('-------------')
        print(dataset, method, 'k_means')
        print('ARI:', ARI(labels_true, labels_pred))
        print('NMI:', NMI(labels_true, labels_pred))

        # 保存聚类结果，用于绘图和其他分析
        joblib.dump(labels_pred, 'labels_pred/' + dataset + '/' + method + 'kmeans.pkl')

        # 生成相关矩阵并保存，用于后续处理
        rel_mat = RelevanceMatrix(labels_pred)
        joblib.dump(rel_mat, 'rel_mat/' + dataset + '/' + method + 'kmeans.pkl')


if __name__ == '__main__':
    dataset_name = 'PBMC'
    n = 6
    rel_mat_k_means(dataset_name, dimension_reduction_methods, 6)
