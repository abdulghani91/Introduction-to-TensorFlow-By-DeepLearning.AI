
from pickletools import optimize
from sre_parse import Verbose
from matplotlib import image
from sklearn import metrics
import tensorflow as tf
import numpy as np
from tensorflow import keras


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop


print(tf.__version__)

train_dir ='./'
test_dir ='./'

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(300, 300),
                                                    batch_size=128,

                                                    class_mode='binary')
test_generator = validation_datagen.flow_from_directory(test_dir,
                                                    target_size=(300, 300),
                                                    batch_size=128,
                                                    class_mode='binary')

# convolutions and max pooling for the colored images
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu',
                                input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        # the same layers as before
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])

# Print the model summary
model.summary()

# compile the model
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001),
             metrics=['accuracy'])

# fit the model 
history = model.fit(train_generator, steps_per_epoch=8,
                    epochs=15, validation_data= test_generator,
                    validation_steps=8, verbose=2)

# to predict we can use the code below

#import numpy as np
#from google.colab import files
#from keras.preprocessing import image

#uploaded = file.upload()

#for fn in uploaded.keys():
    #predicting images
    #path = './' + fn
    #img = image.load_imag(path, target_size=(300, 300))
    #x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)

    #images = np.vstack([x])
    #classes = model.predict(images, batch_size=10)
    #print(classes[0])
    #if classes[0] > 0.5:
    #    print(fn + "is a humman")
    #else: print(fn + "is a hours")