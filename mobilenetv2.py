#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 13:38:29 2024

@author: Leila malihi

'''
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
import itertools
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import time
from sklearn.metrics import accuracy_score

'''------------------------------------------------Dataset---------------------'''
# Set paths to your image folders
data_path = '....'
train_data_dir = data_path + "train"
validation_data_dir = data_path + "val"
test_data_dir = data_path + "test"

# Image dimensions
img_width, img_height = 224, 224  # MobileNetV2 input shape
input_shape = (img_width, img_height, 3)

# Batch size
batch_size = 32
epochs = 30

'''------------------------------------------------Augmentation---------------------'''
# Create ImageDataGenerator instances with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# Flow validation images in batches using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# Flow test images in batches using test_datagen generator
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# Number of training, validation, and test samples
nb_train_samples = len(train_generator.filenames)
nb_validation_samples = len(validation_generator.filenames)
nb_test_samples = len(test_generator.filenames)

'''-----------------------------------------------------Build the model---------------------'''
# Load MobileNetV2 pre-trained on ImageNet without the top layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

# Add GlobalAveragePooling2D layer and a Dense layer with sigmoid activation for binary classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

'''----------------------------------------------------Checkpoint---------------------'''
# Define the filepath for saving the best model
checkpoint_filepath = 'Mobilenetv2-best_model.h5'

# Create a ModelCheckpoint callback to save the best model during training
checkpoint = ModelCheckpoint(
    checkpoint_filepath,
    monitor='val_accuracy',  # Monitor validation accuracy to determine the best model
    save_best_only=True,     # Save only the best model
    mode='max',              # Mode to maximize validation accuracy
    verbose=1                # Print messages about the saving process
)

# Combine base model and top layers into a new model
model = Model(inputs=base_model.input, outputs=predictions)

'''-----------------------------------------------Freeze the layers---------------------'''
# Freeze the layers of the base model (just train the 2 last layers)
for layer in base_model.layers:
    layer.trainable = False
 

'''------------------------------------------------Train the model---------------------'''
# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Start timer
start_time = time.time()

history = model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[checkpoint]  # Pass the list of callbacks here
)

# End timer
end_time = time.time()
training_time = end_time - start_time

'''-------------------------------------------------Test the model---------------------'''
# Load the best model based on validation accuracy
best_model = load_model(checkpoint_filepath)

# Start validation timer
start_validation_time = time.time()

# Evaluate the model on test data and save test loss and accuracy
test_loss, test_acc = best_model.evaluate(test_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)


# End validation timer
end_validation_time = time.time()
validation_time = end_validation_time - start_validation_time

# Predict classes for test data
Y_pred = best_model.predict(test_generator)
y_pred_test = np.round(Y_pred)

# Get true classes for test data
y_test = test_generator.classes

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, Y_pred)
roc_auc = auc(fpr, tpr)

accuracy = accuracy_score(y_test, y_pred_test)
# Calculate loss manually
loss_fn = tf.keras.losses.BinaryCrossentropy()
loss = loss_fn(y_test, Y_pred).numpy()

