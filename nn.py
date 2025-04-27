import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib

print(tf.__version__)

# Data Parameters
batch_size = 512

# Image Parameters
img_height, img_width = 128, 128

data_dir = pathlib.Path('/Users/martin/Desktop/LTAI/CircuitNet/dataset/circuit_dataset')


# Loading training dataset from Drive
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=567,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=567,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print('Number of Classes:', len(class_names))
print(class_names)

# Updated Rescaling Layer (without experimental namespace)
normalization_layer = tf.keras.layers.Rescaling(1. / 255)

AUTOTUNE = tf.data.AUTOTUNE  # Updated tf.data.experimental.AUTOTUNE to tf.data.AUTOTUNE (new API)

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

regularization = 0.0005

# Updated model definition
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255),

    tf.keras.layers.Conv2D(32, 4, activation='relu', kernel_regularizer="l2"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 5, activation='relu', kernel_regularizer="l2"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 5, activation='relu', kernel_regularizer="l2"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 6, activation='relu', kernel_regularizer="l2"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer="l2"),

    tf.keras.layers.Dense(len(class_names), kernel_regularizer="l2")
])

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),  # Kept from_logits=True
    metrics=['accuracy'])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=250
)


plt.subplots(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy and Val. Accuracy")
plt.grid()  # Fixed grid call (removed b=True)
plt.legend(["Traning", "Validation"])
plt.ylabel("Accuracy (%)")
plt.xlabel("EPOCH")

plt.subplots(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
# plt.yscale('log')
plt.legend(["Traning", "Validation"])
plt.title("Loss and Val. Loss")
plt.ylabel("SCCE Loss")
plt.xlabel("EPOCH")
plt.grid(which='both')  # Fixed grid call (removed b=True)

plt.show()

model.save('/Users/martin/Desktop/LTAI/CircuitNet/models/2000_250_4_l_v2.keras')
