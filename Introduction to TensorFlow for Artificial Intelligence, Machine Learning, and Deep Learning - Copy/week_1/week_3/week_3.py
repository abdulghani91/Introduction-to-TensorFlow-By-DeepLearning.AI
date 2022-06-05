
import tensorflow as tf
import numpy as np
from keras.layers.core import activation
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import models

print(tf.__version__)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    # Check accuracy
     if(logs.get('accuracy') >= 0.92): # Experiment with changing this value
      print("\nReached 92% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
#Loading Fashion Minst Dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

# Loading the train and test splite of mnist dataset
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the pixel values of the train and test images
training_images  = training_images / 255.0
test_images = test_images / 255.0

# Build the classification model with Convolutional Neural Network

# convolutions and max pooling
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (3,3), activation='relu',
                                                           input_shape=(28, 28, 1)),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    # the same layers as before
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Print the model summary
model.summary()

# Try editing the convolutions. Change the 32s to either 16 or 64
# Remove the final Convolution
# How about adding more Convolutions?
# Remove all Convolutions but the first. What impact do you think this will have?
#
# To start the training of the model we need to  compile it and fit it
# 1- compile the model
model.compile(optimizer=tf.optimizers.Adam(),loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 2-fit the model
print(f'\nMODEL TRAINING:')
model.fit(training_images, training_labels, epochs=30, callbacks=[callbacks])

# Evaluate on the test set
print(f'\nMODEL EVALUATION:')
test_loss = model.evaluate(test_images, test_labels)

print(test_labels[:100])

f, axarr = plt.subplots(3, 4)

FIRST_IMAGE = 0
SECOND_IMAGE = 23
THIRD_IMAGE = 28
CONVOLUTION_NUMBER = 1

layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

for x in range(0, 4):
    f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[0, x].grid(False)

    f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[1, x].grid(False)

    f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[2, x].grid(False)