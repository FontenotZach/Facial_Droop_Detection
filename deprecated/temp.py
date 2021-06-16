import glob
import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

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

negative_train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in negative_train]
negative_train_imgs = np.array(negative_train_imgs)
negative_train_labels = [fn.split("\\")[-1].split("_")[1].strip() for fn in negative_train]

print(negative_train_labels)
