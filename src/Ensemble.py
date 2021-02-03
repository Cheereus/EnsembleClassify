import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import trange

data = joblib.load('train_data/PBMC_indicator.pkl')
labels = joblib.load('train_data/PBMC_labels.pkl')
n_samples = data.shape[0]
print('Data loaded', sum(labels) / len(labels))

idx = np.arange(n_samples)
np.random.shuffle(idx)
data_shuffled = data[idx]
labels = np.array(labels)[idx]
print('Data shuffled')

# X_train, X_test, Y_train, Y_test = train_test_split(data_shuffled, labels, test_size=0.2)

batch_size = 256

model = Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='sigmoid'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
model.fit(x=data_shuffled, y=labels, validation_split=0.2, epochs=5, batch_size=batch_size)
y_pred = model.predict(data)
# acc = 0
# for i in trange(len(y_pred)):
#     if labels[i] == y_pred[i]:
#         acc += 1
# print(acc / len(y_pred))
joblib.dump(y_pred, 'rel_mat/PBMC_Y_Pred_Proba.pkl')

