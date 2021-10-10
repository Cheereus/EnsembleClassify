import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from sklearn.metrics import accuracy_score

def p_q_loss(y_true, y_pred):
    

def ensemble_learning(dataset):

    data = joblib.load('train_data/' + dataset + '_indicator.pkl')
    labels = joblib.load('train_data/' + dataset + '_labels.pkl')
    n_samples = data.shape[0]
    print('Data loaded', sum(labels) / len(labels))

    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    data_shuffled = data[idx]
    labels = np.array(labels)[idx]
    print('Data shuffled')

    # X_train, X_test, Y_train, Y_test = train_test_split(data_shuffled, labels, test_size=0.2)

    batch_size = 128

    model = Sequential([
        layers.Dense(32, activation='sigmoid'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='sigmoid'),
        layers.Dense(16, activation='sigmoid'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
    model.fit(x=data_shuffled, y=labels, validation_split=0.2, epochs=5, batch_size=batch_size)
    y_pred = model.predict(data)
    model.save('models/' + dataset)
    # acc = 0
    # for i in trange(len(y_pred)):
    #     if labels[i] == y_pred[i]:
    #         acc += 1
    # print('Acc:', acc / len(y_pred))
    joblib.dump(y_pred, 'labels_pred/' + dataset + '/_DNN_Pred_Proba.pkl')
    print('Ensemble Learning Finish')


if __name__ == '__main__':
    dataset_name = 'GSE84133'
    ensemble_learning(dataset_name)
