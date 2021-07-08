import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import glob
import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from pathlib import Path
import shutil


image_size = (90, 90)
batch_size = 16

LEFT_DROOP_MSG = "\tFeatures of facial asymmetry indicating left facial paralysis."
RIGHT_DROOP_MSG = "\tFeatures of facial asymmetry indicating right facial paralysis."
NEGATIVE_MSG = "\tNo features of facial asymmetry detected."

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
    ]
)

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same", trainable=True)(x)
    # x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same", trainable=True)(x)
    # x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=3)


retr_dir_path = ".\\data\\capture_data\\"


shutil.rmtree(retr_dir_path)
Path(retr_dir_path).mkdir(parents=True, exist_ok=True)
exec(open("TestDataCollection.py").read())

capture_files = glob.glob(retr_dir_path + "*", recursive=True)

num_files = len(capture_files)

models = glob.glob(".\\models\\saved_models\\best_models\\*.h5", recursive=True)
# models = glob.glob(".\\models\\saved_models\\second_iteration\\cleaning_3\\*.h5", recursive=True)
model_prediction = []

output = []

negative_pred = 0
left_pred = 0
right_pred = 0

negative_pred_avg = 0
left_pred_avg = 0
right_pred_avg = 0

for loaded_model in models:

    model.load_weights(loaded_model)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        retr_dir_path,
        label_mode="categorical",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(1e-2),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    result = model.predict(test_ds)

    len_result = len(result)

    total_negative = 0
    total_right_droop = 0
    total_left_droop = 0

    for row in result:
        if row[1] > row[2] and row[1] > row[0]:
            total_negative += 1
        elif row[2] > row[1] and row[2] > row[0]:
            total_right_droop += 1
        elif True:
            total_left_droop += 1



    avg_negative = total_negative / len_result
    avg_right_droop = total_right_droop / len_result
    avg_left_droop = total_left_droop / len_result

    if total_left_droop > total_negative and total_left_droop > total_right_droop:
        left_pred += 1
    elif total_right_droop > total_negative and total_right_droop > total_left_droop:
        right_pred += 1
    elif True:
        negative_pred +=1

    if avg_left_droop > avg_negative and avg_left_droop > avg_right_droop:
        left_pred_avg += 1
    elif avg_right_droop > avg_negative and avg_right_droop > avg_left_droop:
        right_pred_avg += 1
    elif True:
        negative_pred_avg +=1

print("\n---Median picker result---\n")
print("\tNegative: " + str(negative_pred))
print("\tRight: " + str(right_pred))
print("\tLeft: " + str(left_pred))
print("\n\tDetermination:")
if right_pred > left_pred and right_pred > negative_pred:
    print(RIGHT_DROOP_MSG)
elif left_pred > negative_pred and left_pred > right_pred:
    print(LEFT_DROOP_MSG)
elif True:
    print(NEGATIVE_MSG)

print("\n\n---Mean picker result---\n")
print("\tNegative: " + str(negative_pred_avg))
print("\tRight: " + str(right_pred_avg))
print("\tLeft: " + str(left_pred_avg))
print("\n\tDetermination:")
if right_pred_avg > left_pred_avg and right_pred_avg > negative_pred_avg:
    print(RIGHT_DROOP_MSG)
elif left_pred_avg > negative_pred_avg and left_pred_avg > right_pred_avg:
    print(LEFT_DROOP_MSG)
elif True:
    print(NEGATIVE_MSG)
