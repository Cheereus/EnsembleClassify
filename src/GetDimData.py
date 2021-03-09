import joblib
from DimensionReduction import *
from Decorator import time_indicator
from Config import dimension_reduction_methods


# TODO 修改为多线程并行
@time_indicator
def get_dim_data(dataset, dr_methods):
    X = joblib.load('datasets/' + dataset + '.pkl')
    print('Data shape:', X.shape)

    # 三百六十度花式降维
    if '_tSNE_' in dr_methods:
        tSNE_dim_data = t_SNE(X, dim=2, with_normalize=True)
        joblib.dump(tSNE_dim_data, 'dim_data/' + dataset + '/_tSNE_.pkl')

    if '_PCA_' in dr_methods:
        PCA_dim_data, ratio, _ = get_pca(X, dim=20, with_normalize=True)
        joblib.dump(PCA_dim_data, 'dim_data/' + dataset + '/_PCA_.pkl')

    if '_FA_' in dr_methods:
        FA_dim_data = feature_agglomeration(X, dim=20, with_normalize=True)
        joblib.dump(FA_dim_data, 'dim_data/' + dataset + '/_FA_.pkl')

    if '_UMAP_' in dr_methods:
        UMAP_dim_data = get_umap(X, dim=20, with_normalize=True)
        joblib.dump(UMAP_dim_data, 'dim_data/' + dataset + '/_UMAP_.pkl')

    if '_LLE_' in dr_methods:
        LLE_dim_data = get_lle(X, dim=20, with_normalize=True)
        joblib.dump(LLE_dim_data, 'dim_data/' + dataset + '/_LLE_.pkl')

    if '_MDS_' in dr_methods:
        MDS_dim_data = get_mds(X, dim=20, with_normalize=True)
        joblib.dump(MDS_dim_data, 'dim_data/' + dataset + '/_MDS_.pkl')

    if '_Isomap_' in dr_methods:
        Isomap_dim_data = get_Isomap(X, dim=20, with_normalize=True)
        joblib.dump(Isomap_dim_data, 'dim_data/' + dataset + '/_Isomap_.pkl')

    print('Dim Data Saved')


if __name__ == '__main__':
    dataset_name = 'Chu_cell_type'
    get_dim_data(dataset_name, dimension_reduction_methods)
