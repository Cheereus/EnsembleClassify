from TrueRel import get_true_rel_mat
from GetDimData import get_dim_data
from BASE_ import base_classify
from IndicateVector import get_indicate_vector
from Ensemble import ensemble_learning
from RelevanceFromPred import get_rel_from_pred
from SubsequentClust import cluster_from_pred_rel
from Config import dimension_reduction_methods, cluster_methods

dataset_name = 'Chu_cell_time'
n = 6

get_true_rel_mat(dataset_name)

get_dim_data(dataset_name, dimension_reduction_methods)

base_classify(dataset_name, cluster_methods, n)

get_indicate_vector(dataset_name)

ensemble_learning(dataset_name)

get_rel_from_pred(dataset_name)

cluster_from_pred_rel(dataset_name)

