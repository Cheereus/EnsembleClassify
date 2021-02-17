from TrueRel import get_true_rel_mat
from GetDimData import get_dim_data
from BASE_ import base_classify
from IndicateVector import get_indicate_vector
from Ensemble import ensemble_learning
from RelevanceFromPred import get_rel_from_pred
from SubsequentClust import cluster_from_pred_rel
from EvaluationAlone import calc_all_evaluate
# 降维方法和聚类方法在 Config.py 中配置
from Config import dimension_reduction_methods, cluster_methods

# 数据集
dataset_name = 'Chu_cell_type'

# 聚类数目
n = 7

# 将数据真实标签转化为相关矩阵
get_true_rel_mat(dataset_name)

# 对原始数据进行多种降维并保存
get_dim_data(dataset_name, dimension_reduction_methods)

# 对降维数据进行多方法聚类并保存相关矩阵
base_classify(dataset_name, cluster_methods, n)

# 将聚类结果得到的相关矩阵拼接成指示向量
get_indicate_vector(dataset_name)

# 将指示向量输入神经网络训练集成分类器
ensemble_learning(dataset_name)

# 将神经网络预测结果重新构造为相关矩阵
get_rel_from_pred(dataset_name)

# 对预测得到的相关矩阵进行再次聚类并计算指标
cluster_from_pred_rel(dataset_name, n)

# 保存基础分类器的评价指标
calc_all_evaluate(dataset_name)
