from logging import FATAL
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import config


#load model
#model = keras.models.load_model(".\my_model\my_model.h5", compile=False) #load HDF5 format
model = keras.models.load_model("my_model") #load SavedModel format

#predict with new data
testimg_path = "./test_redherring2.png"

img = tf.keras.utils.load_img(testimg_path, target_size=(config.img_height, config.img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
decimal_score = 100 * np.max(score)

if decimal_score < 90:
  print("This image most likely belongs to {} with a {:.2f} percent confidence.".format("unknown object", decimal_score))
else:
  print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(config.train_ds.class_names[np.argmax(score)], decimal_score)
  )