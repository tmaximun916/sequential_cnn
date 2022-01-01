import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import config
batch_size = config.batch_size
img_height = config.img_height
img_width = config.img_width

train_ds = config.train_ds
val_ds = config.val_ds

class_names = train_ds.class_names
print(class_names)

import matplotlib.pyplot as plt
#Visualize the data, first 9 images (optional)
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")

# for image_batch, labels_batch in train_ds:
#   print(image_batch.shape)
#   print(labels_batch.shape)
#   break


#Configure the dataset for performance
#buffered prefetching so you can yield data from disk without having I/O become blocking.
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) #keeps the images in memory after they're loaded off disk during the first epoch
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#Standardize the data
normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# check the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

#code to augment data using keras api(from tensorflow website)
data_augmentation = keras.Sequential(
  [
    layers.Resizing(img_height, img_width),
    layers.RandomFlip("horizontal_and_vertical",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  ]
)

#Create the model
num_classes = len(class_names)

model = Sequential([
  data_augmentation,
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

#Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#use callbacks to save the model at its highest accuracy
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=".\my_model\my_model.h5", monitor='val_accuracy', mode='max', save_best_only=True) #save model
#model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='weights/cp.h5', monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True,) #save weights only

#train the model
epochs=15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[model_checkpoint_callback]
)

model.summary()

#save the model at the end of training
# model.save("my_model") #save the model in SavedModel format
# model.save(".\my_model\my_model.h5") #save the model in HDF5 format
# model.save_weights('weights/cp') #save the weights only in checkpoint format

#visualise
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
