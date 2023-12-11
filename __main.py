import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plt

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))

features = np.genfromtxt('data-pd-rychtarka.csv', delimiter=',', skip_header=1, usecols=[7])
labels = np.genfromtxt('data-pd-rychtarka.csv', delimiter=',', skip_header=1, usecols=[5])

print(features, labels)
history = model.fit(features, labels, epochs=200, verbose=True)
print("Finished training the model")

# plt.xlabel('Epoch Number')
# plt.ylabel("Loss Magnitude")
# plt.plot(history.history['loss'])

print(model.predict(["2023-12-09 17:30:01"]))