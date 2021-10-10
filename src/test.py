from ReadData import read_from_txt
from DimensionReduction import t_SNE
from Utils import draw_scatter
import numpy as np

head, data = read_from_txt('datasets/GSE85241.csv')
data = data.T.astype(np.float)
print(len(head), data.shape)

labels = [i.split('-')[0] for i in head]
dim_data = t_SNE(data, with_normalize=True)
x = [i[0] for i in dim_data]
y = [i[1] for i in dim_data]

colors = ['c', 'b', 'g', 'r']

draw_scatter(x, y, labels, colors)


