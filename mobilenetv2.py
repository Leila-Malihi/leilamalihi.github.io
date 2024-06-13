#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 13:38:29 2024

@author: lemalihi
"""
'''#Guide
This code is mobilenetv2 with checkpoint
it works with Tensorflow:2.7 and keras:2.7
the gole is to classify binary classification
you should just determine the data-path for whole train,test,val
also 
nb_train_samples = ..
nb_validation_samples = ..
nb_test_samples = ..

the code save the best model in the name of 'Mobilenetv2-best_model.h5'
also the result like:acc,presioson,recall , and timein a text file
also confusion matrix and loss and acc curve , roc curve all in a same direcotry as your code is

We freeze whole model and added:
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
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
#data_path = '/net/projects/scratch/winter/valid_until_31_July_2024/lemalihi/lemalihi/ZIEL_labelstudio_data/new data for maceration-balance-500/full-splitted/'
data_path = '/net/projects/scratch/winter/valid_until_31_July_2024/lemalihi/lemalihi/ZIEL_labelstudio_data/balance-500-wound types/maceration/maceration-cropped double-checked-splitted/'
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
#for layer in base_model.layers:
  #  layer.trainable = False
    
    # Freeze all layers except the last two
for layer in base_model.layers[:-6]:
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
'''-----------------------------------------------------Save the results---------------------'''
'''-text file'''
# Save test loss and accuracy to a text file
with open('test_results-Inception.txt', 'w') as f:
    f.write(f'Test Loss Inception: {test_loss}\n')
    f.write(f'Test Accuracy Inception: {test_acc}\n')
    f.write(f'Test Loss manually: {loss}\n')
    f.write(f'Test Accuracy manually: {accuracy }\n')
    f.write(f'Precision: {precision}\n')
    f.write(f'Recall: {recall}\n')
    f.write(f'F1-score: {f1}\n')
    f.write(f'Training Time: {training_time} seconds\n')
    f.write(f'Validation Time: {validation_time} seconds\n')

'''-confusion'''
# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)

# Plot confusion matrix
plt.figure(dpi=500)
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Non-Macerated', 'Macerated'], rotation=45)
plt.yticks(tick_marks, ['Non-Macerated', 'Macerated'])

# Print confusion matrix values on plot
thresh = conf_matrix.max() / 2.
for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix-mobilenetv2.png')
# plt.show()

'''-ACC curve'''
plt.figure(dpi=500)
# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.savefig('accuracy_curve-mobilenetv2.png')
# plt.show()

'''-Loss curve'''
plt.figure(dpi=500)
# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('loss_curve-mobilenetv2.png')
# plt.show()

'''-ROC curve'''
plt.figure(dpi=500)
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve-mobilenetv2.png')
# plt.show()
