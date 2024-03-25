import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, InputLayer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LeakyReLU, ReLU

from keras.applications import EfficientNetV2B3

USE_GPU = False
PARALLEL = False

if PARALLEL:
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(10)
elif USE_GPU:
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

## Define path to data and parameters for loader
TRAIN_PATH = './data/training/'
TEST_PATH = './data/testing/'
LABELS_PATH = './data/training_labels.csv'

print(os.listdir(TRAIN_PATH))
print(os.listdir(TEST_PATH))

TEST_SIZE = 0.15
BATCH_SIZE = 32
N_CLASSES = 5
TARGET_SIZE = (256,256)

## Get labels for train and test data
labels_df = pd.read_csv(LABELS_PATH)
labels_df['full_id'] = [os.path.join('/home/scottdb1/flower_classification/data/training/training', l) for l in labels_df.ID]

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
    class_mode='categorical', 
    shuffle=False, 
    batch_size=BATCH_SIZE)

class_names = list(train_gen.class_indices.keys())
initializer = tf.keras.initializers.HeNormal(seed=9) # Joe Burrow

#base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
base_model = EfficientNetV2B3(include_preprocessing=False, weights='imagenet', include_top=False, input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3)) #B3
UNFREEZE_N = 1

# Freeze all but N layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Optionally, unfreeze the top N layers
for layer in base_model.layers[-UNFREEZE_N:]:
    layer.trainable = True

cnn = tf.keras.Sequential([
    ## Augmentation layers
    tf.keras.layers.RandomFlip("horizontal_and_vertical"), #"horizontal"
    tf.keras.layers.RandomRotation(0.2),
    #tf.keras.layers.GaussianNoise(0.1),

    # Base model
    base_model,
    tf.keras.layers.Dropout(0.4),

    ## Convolutional => MaxPooling layers
    tf.keras.layers.Conv2D(72, (3, 3), kernel_initializer=initializer, activation='silu', input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.45),

    ## Fully connected layers
    tf.keras.layers.Flatten(),
	
    tf.keras.layers.Dense(56, activation='silu', kernel_initializer=initializer),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(56, activation='silu', kernel_initializer=initializer),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    
    ## Output layer
    tf.keras.layers.Dense(N_CLASSES, activation='softmax', kernel_initializer=initializer)
])

cnn.compile(
  optimizer=tf.keras.optimizers.AdamW(), #weight_decay=0.006
  loss=tf.keras.losses.CategoricalCrossentropy(),
  metrics=['accuracy'])

## Fit model
checkpoint_filepath = 'ckpt/latest.weights.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=10)

cnn.fit( 
  train_gen,
  validation_data=test_gen,
  callbacks=[earlystop_callback, model_checkpoint_callback],
  epochs=500
)

test_eval = cnn.evaluate(test_gen)

cnn.load_weights(checkpoint_filepath)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

## Predict new data

y_pred = cnn.predict(new_gen)

pred_files = [f.split('/')[-1] for f in new_gen.filenames]
pred_labels = [class_names[i] for i in y_pred.argmax(axis=1)]
submission_df = pd.DataFrame({'ID': pred_files, 'Prediction': pred_labels})

## Write to CSV
submission_df.to_csv('latest.csv', index=False)

print('program complete.')
