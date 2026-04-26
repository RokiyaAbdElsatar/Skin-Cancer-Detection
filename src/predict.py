import numpy as np
import argparse
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf
import os

IMG_SIZE = 128


def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def predict_tflite(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return prediction


def predict_h5(model, img_array):
    prediction = model.predict(img_array)
    return prediction


def main(image_path, model_type='h5'):
    if model_type == 'tflite':
        model_path = 'model/skin_cancer_model.tflite'
        if not os.path.exists(model_path):
            print(f"Error: TFLite model not found at {model_path}")
            print("Run convert_tflite.py first to create the model")
            return
        
        interpreter = load_tflite_model(model_path)
        print(f"Loaded TFLite model from {model_path}")
    else:
        model = load_model("model/skin_cancer_model.h5")
        print("Loaded H5 model from model/skin_cancer_model.h5")
    
    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if model_type == 'tflite':
        prediction = predict_tflite(interpreter, img_array)
    else:
        prediction = predict_h5(model, img_array)
    
    confidence = float(prediction[0][0]) if model_type == 'tflite' else float(prediction[0][0])
    
    is_malignant = confidence > 0.5
    
    print("")
    print("=" * 50)
    if is_malignant:
        print("RESULT: MALIGNANT (Cancerous)")
        print("=" * 50)
        print(f"Confidence: {confidence:.1%}")
        print("")
        print("WARNING: This lesion shows signs of potential malignancy!")
        print("Please consult a dermatologist for professional diagnosis.")
    else:
        print("RESULT: BENIGN (Non-Cancerous)")
        print("=" * 50)
        print(f"Confidence: {(1-confidence):.1%}")
        print("")
        print("This lesion appears to be benign (NOT cancer).")
        print("However, consult a dermatologist for confirmation.")
    
    return is_malignant


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Skin Cancer Detection')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--model', choices=['h5', 'tflite'], default='h5', help='Model type to use')
    
    args = parser.parse_args()
    main(args.image_path, args.model)