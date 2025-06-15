import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model("models/2000_250_4_l_v2.keras")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("circuit_model_v2.tflite", "wb") as f:
    f.write(tflite_model)