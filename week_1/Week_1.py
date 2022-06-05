

import tensorflow as tf
import numpy as np
from tensorflow import keras

print(tf.__version__)

# Build a sequntal model
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

#compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

#define model input and output for the training
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

#start the training
model.fit(xs, ys, epochs=500)

#print the model predictions
print(model.predict([10.0]))