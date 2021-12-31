import numpy as np
import tensorflow as tf
from tensorflow import keras

import training

img_height = 180
img_width = 180

#load model
model = keras.models.load_model("my_model")

#predict with new data
testimg_path = "./test_redherring2.png"

img = tf.keras.utils.load_img(testimg_path, target_size=(img_height, img_width))
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
      .format(training.class_names[np.argmax(score)], decimal_score)
  )