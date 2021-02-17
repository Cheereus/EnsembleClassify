import joblib
from Utils import get_color, draw_scatter


def draw_2d_from_tSNE(dataset):
    default_colors = ['b', 'g', 'r', 'm', 'y', 'k', 'c']
    dim_data = joblib.load('dim_data/' + dataset + '/_tSNE_.pkl')
    x = [i[0] for i in dim_data]
    y = [i[1] for i in dim_data]

    labels = joblib.load('datasets/' + dataset + '_labels.pkl')
    colors = get_color(labels, default_colors)
    draw_scatter(x, y, labels, colors, title='True Label with t-SNE, ' + dataset)

    labels_pred = joblib.load('labels_pred/' + dataset + '/' + 'Final_Pred.pkl')
    colors = get_color(labels_pred, default_colors)
    draw_scatter(x, y, labels_pred, colors, title='Predict Label with t-SNE, ' + dataset)


if __name__ == '__main__':
    dataset_name = 'Chu_cell_type'
    draw_2d_from_tSNE(dataset_name)
