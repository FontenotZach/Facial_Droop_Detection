import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import glob
import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

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


retr_dir_path = ".\\data\\conglom_data\\test_data\\"

right_droop_dirs = glob.glob(retr_dir_path + "right_droop\\*", recursive=True)
left_droop_dirs  = glob.glob(retr_dir_path + "left_droop\\*", recursive=True)
negative_dirs    = glob.glob(retr_dir_path + "*negative\\*", recursive=True)

right_total = len(right_droop_dirs)
left_total = len(left_droop_dirs)
negative_total = len(negative_dirs)

models = glob.glob(".\\models\\multiclass_conv\\*.h5", recursive=True)
# models = glob.glob(".\\models\\saved_models\\second_iteration\\cleaning_3\\*.h5", recursive=True)
model_prediction = []

output = []

for loaded_model in models:

    right_correct = 0
    right_left = 0
    right_negative = 0
    left_correct = 0
    left_right = 0
    left_negative = 0
    negative_correct = 0
    negative_right = 0
    negative_left = 0

    for dir in right_droop_dirs:
        model.load_weights(loaded_model)

        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            dir,
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

        avg_negative = 0
        avg_right_droop = 0
        avg_left_droop = 0

        for row in result:
            if row[1] > row[2] and row[1] > row[0]:
                avg_negative += 1
            elif row[2] > row[1] and row[2] > row[0]:
                avg_right_droop += 1
            elif True:
                avg_left_droop += 1



        avg_negative /= len_result
        avg_right_droop /= len_result
        avg_left_droop /= len_result

        # print("\n\n--- Images processed ---")
        # print("\nProbabilities:")
        # print("\tnegative probability: " + str(avg_negative))
        # print("\tleft droop probability: " + str(avg_left_droop))
        # print("\tright droop probability: " + str(avg_right_droop))
        #
        # print("\nAI suggestion:")
        if avg_negative > avg_left_droop and avg_negative > avg_right_droop:
            # print(NEGATIVE_MSG)
            right_negative += 1
        elif avg_right_droop > avg_negative and avg_right_droop > avg_left_droop:
            # print(RIGHT_DROOP_MSG)
            right_correct += 1
        elif True:
            right_left +=1
            # print(LEFT_DROOP_MSG)

    for dir in left_droop_dirs:
        model.load_weights(loaded_model)

        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            dir,
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

        avg_negative = 0
        avg_right_droop = 0
        avg_left_droop = 0

        for row in result:
            if row[1] > row[2] and row[1] > row[0]:
                avg_negative += 1
            elif row[2] > row[1] and row[2] > row[0]:
                avg_right_droop += 1
            elif True:
                avg_left_droop += 1
            # avg_negative += row[1]
            # avg_right_droop += row[2]
            # avg_left_droop += row[0]



        # avg_negative /= len_result
        # avg_right_droop /= len_result
        # avg_left_droop /= len_result

        stroke_risk = (1 - avg_negative) * 100
        # print("\n\n--- Images processed ---")
        # print("\nProbabilities:")
        # print("\tnegative probability: " + str(avg_negative))
        # print("\tleft droop probability: " + str(avg_left_droop))
        # print("\tright droop probability: " + str(avg_right_droop))
        #
        # print("\nAI suggestion:")
        if avg_negative > avg_left_droop and avg_negative > avg_right_droop:
            # print(NEGATIVE_MSG)
            left_negative += 1
        elif avg_right_droop > avg_negative and avg_right_droop > avg_left_droop:
            # print(RIGHT_DROOP_MSG)
            left_right += 1
        elif True:
            # print(LEFT_DROOP_MSG)
            left_correct += 1

    for dir in negative_dirs:
        model.load_weights(loaded_model)

        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            dir,
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

        avg_negative = 0
        avg_right_droop = 0
        avg_left_droop = 0

        for row in result:
            if row[1] > row[2] and row[1] > row[0]:
                avg_negative += 1
            elif row[2] > row[1] and row[2] > row[0]:
                avg_right_droop += 1
            elif True:
                avg_left_droop += 1



        avg_negative /= len_result
        avg_right_droop /= len_result
        avg_left_droop /= len_result

        stroke_risk = (1 - avg_negative) * 100
        # print("\n\n--- Images processed ---")
        # print("\nProbabilities:")
        # print("\tnegative probability: " + str(avg_negative))
        # print("\tleft droop probability: " + str(avg_left_droop))
        # print("\tright droop probability: " + str(avg_right_droop))
        #
        # print("\nAI suggestion:")
        if avg_negative > avg_left_droop and avg_negative > avg_right_droop:
            # print(NEGATIVE_MSG)
            negative_correct += 1
        elif avg_right_droop > avg_negative and avg_right_droop > avg_left_droop:
            # print(RIGHT_DROOP_MSG)
            negative_right += 1
        elif True:
            negative_left += 1
            # print(LEFT_DROOP_MSG)


    right_percentage = "{:.00%}".format(right_correct/right_total)
    left_percentage = "{:.00%}".format(left_correct/left_total)
    negative_percentage = "{:.0%}".format(negative_correct/negative_total)

    right_left_confusion = "{:.00%}".format(right_left/right_total)
    right_negative_confusion = "{:.00%}".format(right_negative/right_total)
    negative_right_confusion = "{:.00%}".format(negative_right/right_total)
    negative_left_confusion = "{:.00%}".format(negative_left/right_total)
    left_right_confusion = "{:.00%}".format(left_right/right_total)
    left_negative_confusion = "{:.00%}".format(left_negative/right_total)

    total_accuracy = "{:.00%}".format( ( (right_correct/right_total) + (left_correct/left_total) + (negative_correct/negative_total) ) / 3)

    output.append("Model: " + loaded_model + "\tright: " + right_percentage +"\trl_con: " + right_left_confusion + "\trn_con: "  + right_negative_confusion
    + "\tleft " + left_percentage + "\tlr_con: " + left_right_confusion + "\tln_con: " + left_negative_confusion
    + "\tnegative: " + negative_percentage + "\tnr_con: " + negative_right_confusion + "\tnl_con: " + negative_left_confusion
    + "\ttotal accuracy: " + total_accuracy)

    # right_percentage = "{:.0%}".format(right_correct/right_total)
    # left_percentage = "{:.0%}".format(left_correct/left_total)
    # negative_percentage = "{:.0%}".format(negative_correct/negative_total)
    #
    # print("\n\n\n\t--- Test Accuracy Report ---\n")
    # print("\t Right Accuracy: " + right_percentage)
    # print("\t Left Accuracy: " + left_percentage)
    # print("\t Negative Accuracy: " + negative_percentage)

for thing in output:
    print(thing)
