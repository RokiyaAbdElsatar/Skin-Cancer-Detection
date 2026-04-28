import tensorflow as tf
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'skin_cancer_model.h5')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'skin_cancer_model.tflite')

MODEL_PATH = os.path.abspath(MODEL_PATH)
OUTPUT_PATH = os.path.abspath(OUTPUT_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

os.makedirs('model', exist_ok=True)
with open(OUTPUT_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model saved to {OUTPUT_PATH}")

file_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"Model size: {file_size:.2f} MB")