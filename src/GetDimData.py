import joblib
from DimensionReduction import *
from Decorator import time_indicator
from Config import dimension_reduction_methods


@time_indicator
def get_dim_data(dataset, methods):
    X = joblib.load('datasets/' + dataset + '.pkl')
    labels = joblib.load('datasets/' + dataset + '_labels.pkl')

    if '_tSNE_' in methods:
        tSNE_dim_data = t_SNE(X, dim=5, with_normalize=True)
        joblib.dump(tSNE_dim_data, 'dim_data/' + dataset + '/_tSNE_.pkl')

    if '_PCA_' in methods:
        PCA_dim_data = get_pca(X, dim=20, with_normalize=True)
        joblib.dump(PCA_dim_data, 'dim_data/' + dataset + '/_PCA_.pkl')

    if '_FA_' in methods:
        FA_dim_data = feature_agglomeration(X, dim=20, with_normalize=True)
        joblib.dump(FA_dim_data, 'dim_data/' + dataset + '/_FA_.pkl')

    if '_UMAP_' in methods:
        UMAP_dim_data = get_umap(X, dim=20, with_normalize=True)
        joblib.dump(UMAP_dim_data, 'dim_data/' + dataset + '/_UMAP_.pkl')

    if '_LLE_' in methods:
        LLE_dim_data = get_lle(X, dim=20, with_normalize=True)
        joblib.dump(LLE_dim_data, 'dim_data/' + dataset + '/_LLE_.pkl')


if __name__ == '__main__':
    dataset_name = 'PBMC'
    get_dim_data(dataset_name, dimension_reduction_methods)
