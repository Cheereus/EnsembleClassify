import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from sklearn.model_selection import train_test_split

data = joblib.load('train_data/PBMC_indicator.pkl')
labels = joblib.load('train_data/PBMC_labels.pkl')
n_samples = data.shape[0]
print('Data loaded', sum(labels) / len(labels))

idx = np.arange(n_samples)
np.random.shuffle(idx)
data = data[idx]
labels = np.array(labels)[idx]
print('Data shuffled')

X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2)

batch_size = 256

model = Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='sigmoid'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
model.fit(x=X_train, y=Y_train, validation_split=0.2, epochs=10, batch_size=batch_size)
scores = model.evaluate(X_test, Y_test)
print('Evaluate:', scores[1])

