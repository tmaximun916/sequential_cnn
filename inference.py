from logging import FATAL
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import config
img_height=config.img_height
img_width=config.img_width

from PIL import Image, ImageOps
import tensorflow as tf

import cv2
import numpy as np
import os
import sys


def import_and_predict(image_data, model):

        img_array = tf.keras.utils.img_to_array(image_data)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        # size = (75,75)    
        # image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        # image = image.convert('RGB')
        # image = np.asarray(image)
        # image = (image.astype(np.float32) / 255.0)

        # img_reshape = image[np.newaxis,...]

        # prediction = model.predict(img_reshape)
        
        return score

#load model
model = keras.models.load_model(".\my_model\my_model.h5") #load model in HDF5 format
#model = keras.models.load_model("my_model") #load model in SavedModel format

model.summary()
    
cap = cv2.VideoCapture(".\strawberry2.mp4")

if (cap.isOpened()):
    print("Camera OK")
else:
    cap.open()

while (True):
    ret, frame = cap.read()
    if ret:
      cv2.imwrite(filename='img.jpg', img=frame)
      image = Image.open('img.jpg')

      if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

    # Display the predictions
    prediction = import_and_predict(image, model)
    decimal_score = 100 * np.max(prediction)
    if decimal_score < 85:
      txt="This image most likely belongs to {} with a {:.2f} percent confidence.".format("unknown", decimal_score)
      print(txt)
    else:
      txt="This image most likely belongs to {} with a {:.2f} percent confidence.".format(config.train_ds.class_names[np.argmax(prediction)], decimal_score)
      print(txt)

    cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    scale_percent = 40 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim)
    cv2.imshow("Classification", frame)

cap.release()
cv2.destroyAllWindows()
sys.exit()