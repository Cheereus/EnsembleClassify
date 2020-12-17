# EnsembleClassify
Single cell RNA-seq clustering based on ensemble learning

## Notes

The data shape should be cells * genes

数据格式均为 细胞*基因 即行为细胞，列为基因

大数据集的 t-SNE 非常慢，建议在第一次运行时就存为 .pkl 文件，方便以后再用

其他需要重复使用的中间数据也强烈建议存为 .pkl 文件，例如降维后的数据、距离矩阵、关联矩阵等

耗时较长的函数请使用装饰器：

```python
from Decorator import time_indicator

@time_indicator
def xxx():
    print('这样会自动在控制台输出此函数的运行起止时间')
```

