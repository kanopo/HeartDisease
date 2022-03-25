from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras import layers

def ANN(dataset):

    y = dataset.pop("target")
    X = dataset
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20)

    avg = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - avg) / std
    X_val = (X_val - avg) / std

    model = models.Sequential()

    model.add(layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=0.0001)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

    history = model.fit(X_train, y_train, epochs=300, batch_size=8, validation_split=0.2, verbose=1, callbacks=[es])

    test_loss, test_pr = model.evaluate(X_val, y_val)

    return test_pr

data = pd.read_csv("./4 - data cleaning/train.csv")

print(ANN(data))
