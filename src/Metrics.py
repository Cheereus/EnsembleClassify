from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, f1_score


def accuracy(true_labels, predict_labels):
    return accuracy_score(true_labels, predict_labels)


def ARI(true_labels, predict_labels):
    return adjusted_rand_score(true_labels, predict_labels)


def NMI(true_labels, predict_labels):
    return normalized_mutual_info_score(true_labels, predict_labels)


def F1(true_labels, predict_labels):
    return f1_score(true_labels, predict_labels, average='weighted')
