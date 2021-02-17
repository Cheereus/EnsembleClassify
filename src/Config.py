# 在此文件中定义全局配置

# 数据集
datasets = [
    'PBMC'
]

# 数据集对应的聚类数目
cluster_numbers = [
    6
]

# 使用的降维方法
dimension_reduction_methods = [
    # '_AE_',
    '_tSNE_',
    '_PCA_',
    '_FA_',
    '_UMAP_',
    '_LLE_',
    '_MDS_',
    '_Isomap_'
]

# 使用的聚类方法
cluster_methods = [
    'kmeans',
    'AGNES',
    'GMM',
    'Spectral'
]
