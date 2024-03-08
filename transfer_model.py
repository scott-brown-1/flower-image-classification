import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense, InputLayer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LeakyReLU, ReLU

from keras.applications import EfficientNetB5

USE_GPU = False

if USE_GPU:
    #tf.debugging.set_log_device_placement(True)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

## Define path to data and parameters for loader
TRAIN_PATH = './data/training/'
TEST_PATH = './data/testing/'
LABELS_PATH = './data/training_labels.csv'

TEST_SIZE = 0.25
BATCH_SIZE = 16
#IMG_HEIGHT = 180
#IMG_WIDTH = 180

N_CLASSES = 5

TARGET_SIZE = (456, 456)

# Read in data and assign labels

## Get labels for train and test data
labels_df = pd.read_csv(LABELS_PATH)
labels_df['full_id'] = [os.path.join('/home/scottbrown/byu/stat486/projects/flower-image-classification/data/training/training', l) for l in labels_df.ID]

# In the future, organize directory structure for `flow_from_directory`
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=TEST_SIZE)
train_gen = datagen.flow_from_dataframe(
    labels_df, 
    directory=TRAIN_PATH, 
    target_size=TARGET_SIZE,
    subset='training',
    x_col='full_id', 
    y_col='target', 
    class_mode='categorical', 
    batch_size=BATCH_SIZE)

test_gen = datagen.flow_from_dataframe(
    labels_df, 
    directory=TRAIN_PATH, 
    target_size=TARGET_SIZE,
    subset='validation',
    x_col='full_id', 
    y_col='target', 
    class_mode='categorical', 
    batch_size=BATCH_SIZE)

## Read in test data to predict on
new_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
new_gen = new_datagen.flow_from_directory(
    TEST_PATH, 
    target_size=TARGET_SIZE, 
    class_mode=None, 
    shuffle=False, 
    batch_size=1)

class_names = list(train_gen.class_indices.keys())

base_model = EfficientNetB5(weights='imagenet', include_top=False, drop_connect_rate=0.4, input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
UNFREEZE_N = 2

# Freeze all but N layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Optionally, unfreeze the top N layers
for layer in base_model.layers[-UNFREEZE_N:]:
    layer.trainable = True

cnn = tf.keras.Sequential([
    ## Use base model for transfer learning
    base_model,

    ## 3 Convolutional => MaxPooling layers
    tf.keras.layers.Conv2D(48, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Conv2D(48, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.4),

    # tf.keras.layers.Conv2D(32, (2, 2)),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.ReLU(),
    # #tf.keras.layers.MaxPooling2D(),
    # tf.keras.layers.Dropout(0.4),

    # ## 1 Fully connected layer
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),#LeakyReLU(alpha=0.01)),
    
    # ## Output layer
    tf.keras.layers.Dense(N_CLASSES, activation='softmax')
])

cnn.compile(
  optimizer='adam',
  loss=tf.keras.losses.CategoricalCrossentropy(),
  metrics=['accuracy'])

## Fit model
cnn.fit( # TODO: checkpoints
  train_gen,
  validation_data=test_gen,
  epochs=2#25
)

test_eval = cnn.evaluate(test_gen)#X_test, y_test, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

## Predict new data

y_pred = cnn.predict(new_gen)

pred_files = [f.split('/')[-1] for f in new_gen.filenames]
pred_labels = [class_names[i] for i in y_pred.argmax(axis=1)]
submission_df = pd.DataFrame({'ID': pred_files, 'Prediction': pred_labels})

## Write to CSV
submission_df.to_csv('submission.csv', index=False)


