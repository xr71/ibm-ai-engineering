import tensorflow as tf
from tensorflow import keras

model = keras.Sequential()

model.add(tf.keras.layers.Conv2D(16, kernel_size=(2, 2), strides=(1, 1),
                                 activation='relu',
                                 input_shape=(128, 128, 3)
                                 ))


model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

print(model.summary())
