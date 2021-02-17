from GetDimData import get_dim_data
from BASE_KMeans import rel_mat_k_means
from BASE_AGNES import rel_mat_AGNES
from BASE_Spectral import rel_mat_spectral
from BASE_GMM import rel_mat_GMM
from BASE_HCA import rel_mat_hca
from Config import dimension_reduction_methods, cluster_methods


def base_classify(dataset, cl_methods, n_clusters):

    if 'kmeans' in cl_methods:
        rel_mat_k_means(dataset, dimension_reduction_methods, n_clusters)
    if 'AGNES' in cl_methods:
        rel_mat_AGNES(dataset, dimension_reduction_methods, n_clusters)
    if 'GMM' in cl_methods:
        rel_mat_GMM(dataset, dimension_reduction_methods, n_clusters)
    if 'Spectral' in cl_methods:
        rel_mat_spectral(dataset, dimension_reduction_methods, n_clusters)
    if 'HCA' in cl_methods:
        rel_mat_hca(dataset, dimension_reduction_methods, n_clusters)

    print('Base Classifiers Saved')


if __name__ == '__main__':
    dataset_name = 'Chu_cell_type'
    base_classify(dataset_name, cluster_methods, 7)
