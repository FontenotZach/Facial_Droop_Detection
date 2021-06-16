import glob
import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from PIL import Image as im
from sklearn.preprocessing import LabelEncoder
from keras.applications.resnet import ResNet50
from keras.models import Model
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from tensorflow.python.keras import optimizers
import tensorflow as tf
from keras_adabound import AdaBound
from keras_radam import RAdam

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

NUM_CLASSES = 3

# Fixed for color images
CHANNELS = 3

IMAGE_RESIZE = 300
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'

# Common accuracy metric for all outputs, but can use different metrics for different output
LOSS_METRICS = ['accuracy']

# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
NUM_EPOCHS = 10
EARLY_STOP_PATIENCE = 3

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# Training images processed in each step would be no.-of-train-images / STEPS_PER_EPOCH_TRAINING
STEPS_PER_EPOCH_TRAINING = 100
STEPS_PER_EPOCH_VALIDATION = 100

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# NOTE that these BATCH* are for Keras ImageDataGenerator batching to fill epoch step input
BATCH_SIZE_TRAINING = 1
BATCH_SIZE_VALIDATION = 1

# Using 1 to easily manage mapping between test_generator & prediction for submission preparation
BATCH_SIZE_TESTING = 1

learning_proportion = .7
testing_proportion = .15
validate_proportion = .15

right_droop_files = glob.glob(".\\training_data\\*right_droop*\\*.jpg", recursive=True)
left_droop_files  = glob.glob(".\\training_data\\*left_droop*\\*.jpg", recursive=True)
negative_files    = glob.glob(".\\training_data\\*negative*\\*.jpg", recursive=True)

right_droop_length = len(right_droop_files)
left_droop_length = len(left_droop_files)
negative_length = len(negative_files)

right_droop_train = np.random.choice(right_droop_files, size=int(right_droop_length * learning_proportion), replace=False)
left_droop_train  = np.random.choice(left_droop_files, size=int(left_droop_length * learning_proportion), replace=False)
negative_train    = np.random.choice(negative_files, size=int(negative_length * learning_proportion), replace=False)
right_droop_files = list(set(right_droop_files) - set(right_droop_train))
left_droop_files  = list(set(left_droop_files) - set(left_droop_train))
negative_files    = list(set(negative_files) - set(negative_train))

right_droop_length = len(right_droop_files)
left_droop_length = len(left_droop_files)
negative_length = len(negative_files)

right_droop_test = np.random.choice(right_droop_files, size=int(right_droop_length / ((1 - learning_proportion) / testing_proportion)), replace=False)
left_droop_test  = np.random.choice(left_droop_files, size=int(left_droop_length / ((1 - learning_proportion) / testing_proportion)), replace=False)
negative_test    = np.random.choice(negative_files, size=int(negative_length / ((1 - learning_proportion) / testing_proportion)), replace=False)
right_droop_files = list(set(right_droop_files) - set(right_droop_test))
left_droop_files  = list(set(left_droop_files) - set(left_droop_test))
negative_files    = list(set(negative_files) - set(negative_test))

right_droop_length = len(right_droop_files)
left_droop_length = len(left_droop_files)
negative_length = len(negative_files)

right_droop_validate = np.random.choice(right_droop_files, size=(right_droop_length), replace=False)
left_droop_validate  = np.random.choice(left_droop_files, size=(left_droop_length), replace=False)
negative_validate    = np.random.choice(negative_files, size=(negative_length), replace=False)

IMG_WIDTH=300
IMG_HEIGHT=300
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)

negative_train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)).astype(np.uint8) for img in negative_train]
negative_train_imgs = np.array(negative_train_imgs)
negative_train_labels = [fn.split("\\")[-1].split("_")[1].strip() for fn in negative_train]

right_droop_train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)).astype(np.uint8) for img in right_droop_train]
right_droop_train_imgs = np.array(right_droop_train_imgs)
right_droop_train_labels = [fn.split("\\")[-1].split("_")[1].strip() for fn in right_droop_train]

left_droop_train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)).astype(np.uint8) for img in left_droop_train]
left_droop_train_imgs = np.array(left_droop_train_imgs)
left_droop_train_labels = [fn.split("\\")[-1].split("_")[1].strip() for fn in left_droop_train]

negative_validate_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)).astype(np.uint8) for img in negative_validate]
negative_validate_imgs = np.array(negative_validate_imgs)
negative_validate_labels = [fn.split("\\")[-1].split("_")[1].strip() for fn in negative_validate]

right_droop_validate_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)).astype(np.uint8) for img in right_droop_validate]
right_droop_validate_imgs = np.array(right_droop_validate_imgs)
right_droop_validate_labels = [fn.split("\\")[-1].split("_")[1].strip() for fn in right_droop_validate]

left_droop_validate_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)).astype(np.uint8) for img in left_droop_validate]
left_droop_validate_imgs = np.array(left_droop_validate_imgs)
left_droop_validate_labels = [fn.split("\\")[-1].split("_")[1].strip() for fn in left_droop_validate]

negative_test_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)).astype(np.uint8) for img in negative_test]
negative_test_imgs = np.array(negative_test_imgs)
negative_test_labels = [fn.split("\\")[-1].split("_")[1].strip() for fn in negative_test]

right_droop_test_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)).astype(np.uint8) for img in right_droop_test]
right_droop_test_imgs = np.array(right_droop_test_imgs)
right_droop_test_labels = [fn.split("\\")[-1].split("_")[1].strip() for fn in right_droop_test]

left_droop_test_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)).astype(np.uint8) for img in left_droop_test]
left_droop_test_imgs = np.array(left_droop_test_imgs)
left_droop_test_labels = [fn.split("\\")[-1].split("_")[1].strip() for fn in left_droop_test]

train_imgs = np.concatenate((negative_train_imgs, right_droop_train_imgs, left_droop_train_imgs))
train_labels = np.concatenate((negative_train_labels, right_droop_train_labels, left_droop_train_labels))

validate_imgs = np.concatenate((negative_validate_imgs, right_droop_validate_imgs, left_droop_validate_imgs))
validate_labels = np.concatenate((negative_validate_labels, right_droop_validate_labels, left_droop_validate_labels))

test_imgs = np.concatenate((negative_test_imgs, right_droop_test_imgs, left_droop_test_imgs))
test_labels = np.concatenate((negative_test_labels, right_droop_test_labels, left_droop_test_labels))

le = LabelEncoder()
le.fit(train_labels)
train_labels_enc = le.transform(train_labels)
validate_labels_enc = le.transform(validate_labels)
test_labels_enc = le.transform(test_labels)

# train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
#  width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
#  horizontal_flip=False, fill_mode="nearest")
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(train_imgs, train_labels_enc,batch_size=BATCH_SIZE_TRAINING)
validate_generator = val_datagen.flow(validate_imgs, validate_labels_enc, batch_size=BATCH_SIZE_VALIDATION)
test_generator = test_datagen.flow(test_imgs, test_labels_enc, batch_size=BATCH_SIZE_TESTING)

# x,y = train_generator.next()
# for i in range(0,1):
#     print(y)
#     image = x[i]
#     plt.imshow(image)
#     plt.show()

restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT,IMG_WIDTH,3), classes=NUM_CLASSES)
input_shape=(IMG_HEIGHT,IMG_WIDTH,3)

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(32, 3, strides=2, activation="relu", input_shape=(300,300,3))
    self.max1  = tf.keras.layers.MaxPooling2D(3)
    self.bn1   = tf.keras.layers.BatchNormalization()

    # Layer of Block 2
    self.conv2 = tf.keras.layers.Conv2D(64, (3,3), activation="relu")
    self.bn2   = tf.keras.layers.BatchNormalization()
    self.drop  = tf.keras.layers.Dropout(0.3)

    # GAP, followed by Classifier
    self.gap   = tf.keras.layers.GlobalAveragePooling2D()
    self.dense = tf.keras.layers.Dense(NUM_CLASSES)

  def call(self, inputs):
      x = self.conv1(inputs)
      x = self.max1(x)
      x = self.bn1(x)

      x = self.conv2(x)
      x = self.bn2(x)

      x = self.drop(x)
      x = self.gap(x)
      return self.dense(x)

model = MyModel()

model.compile(optimizer=RAdam(),loss='binary_crossentropy', metrics=['binary_accuracy'])

tf.debugging.set_log_device_placement(True)
with tf.device('/CPU:0'):
    model.fit(train_generator,steps_per_epoch=STEPS_PER_EPOCH_TRAINING, epochs=NUM_EPOCHS, validation_data=validate_generator, validation_steps=50, verbose=1, workers=1)
try:
    model.save_weights("models\\droop_predictor_resnet50_HDF5.h5")
except:
    print("could not save HDF5")
try:
    model.save_weights("models\\droop_predictor_resnet50_tf", save_format="tf")
except:
    print("could not save checkpoint")
model.summary()

# Loads the weights
model.load_weights("models//droop_predictor_resnet50_tf")

# Re-evaluate the model

loss, acc = model.evaluate(test_generator, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# output = restnet.layers[-1].output
# output = keras.layers.Flatten()(output)
# restnet = Model(restnet.input)
# for layer in restnet.layers:
#     layer.trainable = False


#
#
#
# model = Sequential()
# model.add(restnet)
# model.add(Dense(512, activation='relu', input_dim=input_shape))
# model.add(Dropout(0.3))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1, activation='sigmoid'))
# model.layers[0].trainable = False
# model.compile(loss='binary_crossentropy', metrics=['accuracy'])
#
# model.build(input_shape)
# model.summary()
# model.fit(train_generator,steps_per_epoch=100, epochs=100, validation_data=validate_generator, validation_steps=50, verbose=1)
# plt.imshow(negative_train_imgs[23])
# plt.show()


# train_imgs = np.concatenate(negative_train_imgs, right_droop_train_imgs, left_droop_train_imgs)
# train_labels = np.concatenate(negative_train_labels, right_droop_train_labels, left_droop_train_labels)
#
# validate_imgs = np.concatenate(negative_validate_imgs, right_droop_validate_imgs, left_droop_validate_imgs)
# validate_labels = np.concatenate(negative_validate_labels, right_droop_validate_labels, left_droop_validate_labels)
