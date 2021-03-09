import os


def make_dir(dataset):
    path_list = ['dim_data', 'labels_pred', 'models', 'rel_mat']
    for path in path_list:
        isExists = os.path.exists(path + '/' + dataset)
        if not isExists:
            os.makedirs(path + '/' + dataset)

    print('Directory created')


if __name__ == '__main__':
    make_dir('Kolodziejczyk')
