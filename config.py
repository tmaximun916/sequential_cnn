import tensorflow as tf

batch_size = 16
img_height = 180
img_width = 180

import pathlib
data_dir = ".\img"
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg'))) # .glob to look inside subtree
print(image_count) # print out no. of images

#use a validation split when developing model. 80% of the images for training, and 20% for validation.
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2, 
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)