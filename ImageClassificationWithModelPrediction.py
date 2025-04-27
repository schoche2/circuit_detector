import tensorflow as tf
import numpy as np
import os

# Parameters for prediction
img_height, img_width = 128, 128  # Should match the training settings
target_folder = '/Users/martin/Desktop/LTAI/CircuitNet/dataset/circuit_dataset/'  # Path to the folder with images to predict
model_path = '/Users/martin/Desktop/LTAI/CircuitNet/models/2000_250_4_l.keras'

# Load the model
model = tf.keras.models.load_model(model_path)


# Recreate the training dataset to simulate predictions
train_eval_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/Users/martin/Desktop/LTAI/CircuitNet/dataset/testdata',
    color_mode='grayscale',
    image_size=(128, 128),
    batch_size=512
)

print(train_eval_ds.class_names)

loss, accuracy = model.evaluate(train_eval_ds, verbose=1)
print(f"Model accuracy on training data: {accuracy}")
