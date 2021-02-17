# EnsembleClassify
Single cell RNA-seq clustering based on ensemble learning

## 注意

数据格式均为 `cells*genes` 即行为细胞，列为基因

大数据集的 t-SNE 非常慢，建议在第一次运行时就存为 `.pkl` 文件，方便以后再用

其他需要重复使用的中间数据也强烈建议存为 `.pkl` 文件，例如降维后的数据、距离矩阵、关联矩阵等

耗时较长的函数请使用装饰器：

```python
from Decorator import time_indicator

@time_indicator
def xxx():
    print('这样会自动在控制台输出此函数的运行起止时间')
```

## 核心函数库

请不要擅自修改核心文件中的内容

### Utils.py

包括一些工具函数，主要用于绘图：

* `get_color()` 将类别标签对应转换为可供绘图的颜色序列
* `draw_scatter()` 根据类别及颜色进行 2D 散点图绘制
* `draw_scatter3d()` 根据类别及颜色进行 3D 散点图绘制

### ReadData.py

包括数据读取及存储的函数，均返回 `np.array` 格式：

* `read_from_mat()` 从 `.mat` 文件中读取数据
* `read_from_csv()` 从 `.csv` 文件中读取数据
* `read_from_tsv()` 从 `.tsv` 文件中读取数据
* `read_from_txt()` 从 `.txt` 文件中读取数据
* `data_to_csv()` 将二维数据以表格形式存储到 `.csv` 文件中

### DimensionReduction.py

包括降维及相关函数：

* `get_normalize()` 数据标准化
* `t_SNE()` t-SNE 降维
* `get_pca()` pca 降维

### Distance.py

包括各种距离度量的计算：

* `RelevanceMatrix()` 根据标签输出关联矩阵

### Clustering.py

包括聚类相关算法：

* `k_means()` k-means 聚类
* `knn()` k-NN
* `hca()` 层次聚类，输出模型
* `hca_dendrogram()` 对层次聚类模型进行绘图
* `hca_labels()` 对层次聚类结果进行标记

### Metrics.py

包括各种分类和聚类性能评价指标：

* `accuracy()` 准确率指标
* `ARI()` 调整兰德系数 ARI
* `NMI()` 归一化互信息 NMI
* `F1()` F1-score

### Decorator.py

包括一些装饰器：

* `time_indicator()` 在控制台输出函数的运行起止时间

## 核心流程文件

### Config.py

* 全局配置文件

### GetDimData.py

* 对原始数据进行多种降维并保存

### BASE_*.py

* 对降维数据进行多方法聚类并保存相关矩阵

### IndicateVector.py

* 将聚类结果得到的相关矩阵拼接成指示向量

### Ensemble.py

* 将指示向量输入神经网络训练集成分类器

### RelevanceFromPred.py

* 将神经网络预测结果重新构造为相关矩阵

### SubsequentPred.py

* 对预测得到的相关矩阵进行再次聚类并计算指标

### Main.py

* 运行所有流程