"""
OUTDATED! Since 29.01.2022
This was used to save the loaded dataset, but its easier to load it again from the folders with the same seed and
saves space on the harddrive.

This script creates a train and validation dataset from the image data with a 80% to 20% splitt.
The Datasets are normalized to be between 0 and 1 and then saved with tf.data.experimental.save
Can be loaded with train_ds_loaded = tf.data.experimental.load(ds_save_dir+"/ds_train")
- "ds_val" for the validation dataset

DATA can also just be loaded once again for each model with the same seed that will result in using the same validation
and training set as well.
"""

import tensorflow as tf
import numpy as np
import pandas as pd

data_dir = "C:/Users/Christian/PycharmProjects/_data/InnovationsProjekt"  # data to be loaded
ds_save_dir = "/"  # data is saved to this path

### LOAD IMAGES

batch_size = 32
img_height = 224
img_width = 224
# TODO can you resize the images after the dataset is created?

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  #color_mode="grayscale",
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  #color_mode="grayscale",
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

### VISUALIZE 9 Images
#import matplotlib.pyplot as plt
#
#print("plot now")
#plt.figure(figsize=(10, 10))
#for images, labels in train_ds.take(1):
#  for i in range(9):
#    ax = plt.subplot(3, 3, i + 1)
#    plt.imshow(images[i].numpy().astype("uint8"))
#    plt.title(class_names[labels[i]])
#    plt.axis("off")


# shape is (32, 224, 224, 1)

### STANDARDIZE
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds_train = train_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_ds_val = val_ds.map(lambda x, y: (normalization_layer(x), y))

### save the data (is already saved in greyscale and normalized [0-1])
tf.data.experimental.save(normalized_ds_val, ds_save_dir+"/ds_val")
print("val saved")
tf.data.experimental.save(normalized_ds_train, ds_save_dir+"/ds_train")
print("train saved")

### load data
#train_ds_loaded = tf.data.experimental.load(ds_save_dir+"/ds_train")
#val_ds_loaded = tf.data.experimental.load(ds_save_dir+"/ds_val") # LoadDataset Objects

# Performnance

# needs to be done after loading for better performance
#AUTOTUNE = tf.data.AUTOTUNE

#train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
#val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)