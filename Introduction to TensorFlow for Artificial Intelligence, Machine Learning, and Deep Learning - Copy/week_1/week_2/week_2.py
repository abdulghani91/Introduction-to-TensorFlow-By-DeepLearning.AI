
import tensorflow as tf
import numpy as np
from keras.layers.core import activation
from tensorflow import keras
import matplotlib.pyplot as plt


print(tf.__version__)


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    # Check accuracy
     if(logs.get('accuracy') >= 0.9): # Experiment with changing this value
    # Or we can check the loss
  #if (logs.get('loss') < 0.3):
  #print("\nLoss is lower than 0.4 so cancelling training!")
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True


# we need to add callbacks=[callbacks] in the model fit
callbacks = myCallback()
#Loading Fashion Minst Dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

# Loading the train and test splite of mnist dataset
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

# You can put between 0 to 59999 here
index = 0

# Set number of characters per row when printing
np.set_printoptions(linewidth=320)

# Print the label and image
print(f'LABEL: {training_labels[index]}')
print(f'\nIMAGE PIXEL ARRAY:\n {training_images[index]}')

# Visualize the image
plt.imshow(training_images[index], cmap='Greys')

# Normalize the pixel values of the train and test images
training_images  = training_images / 255.0
test_images = test_images / 255.0

# Build the classification model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Declare sample inputs and convert to a tensor
inputs = np.array([[1.0, 3.0, 4.0, 2.0]])
inputs = tf.convert_to_tensor(inputs)
print(f'input to softmax function: {inputs.numpy()}')

# Feed the inputs to a softmax activation function
outputs = tf.keras.activations.softmax(inputs)
print(f'output of softmax function: {outputs.numpy()}')

# Get the sum of all values after the softmax
sum = tf.reduce_sum(outputs)
print(f'sum of outputs: {sum}')

# Get the index with highest value
prediction = np.argmax(outputs)
print(f'class with highest probability: {prediction}')

# To start the training of the model we need to  compile it and fit it
# 1- compile the model
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# fit the model
model.fit(training_images, training_labels, epochs=30, callbacks=[callbacks])

# Evaluate the model on unseen data
model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])

print('\n', test_labels[0])