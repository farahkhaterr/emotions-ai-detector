import json
import tensorflow as tf
import keras

# Load model configuration
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# Reconstruct model architecture
model = keras.saving.deserialize_keras_object(config)

# Load trained weights
model.load_weights("model.weights.h5")

# Save complete Keras model
model.save("emotion_detection_final.keras")

print("Model converted successfully!")