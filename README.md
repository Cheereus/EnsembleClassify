# EnsembleClassify
Single cell RNA-seq clustering based on ensemble learning

## Notes

The data shape should be cells * genes

数据格式均为 细胞*基因 即行为细胞，列为基因

大数据集的 t-SNE 非常慢，建议在第一次运行时就存为 .pkl 文件，方便以后再用