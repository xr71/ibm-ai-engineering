from tensorflow import keras
from tensorflow.keras import Sequential
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# tensorflow is from Google, with keras integrated
# pytorch is from torch which is written in Lua
# pytorch is gaining popular because it is more pythonic
# pytorch is supported by Facebook - feels like NumPy


print(keras)
print(Sequential)


df = pd.read_csv(
    "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv")


print(df.head())


# LR

def LinearRegression(df):
    model = Sequential()

    label = df.pop('strength')
    features = df.copy()
    n_cols = features.shape[1]

    model.add(keras.layers.Dense(5, activation='relu', input_shape=(n_cols,)))
    model.add(keras.layers.Dense(1))
    print(model.summary())

    model.compile(optimizer='adam', loss='mse')

    history = model.fit(features, label, epochs=1000, batch_size=64)
    return history


lr = LinearRegression(df)


# Classification

def Classification(df):
    model = Sequential()
    model.add(keras.layers.Dense(8, activation='relu', input_shape=(n_cols,)))
    model.add(keras.layers.Dense(4, activation='relu'))
    model.add(keras.layers.Dense(4, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])


clf = Classification()
