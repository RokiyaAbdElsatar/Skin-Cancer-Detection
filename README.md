# Mobile Skin Analyzer - Lightweight CNN for Skin Cancer Detection

## Project Overview

This project aims to build a **lightweight CNN model** for on-device/mobile skin cancer detection. The goal is to achieve real-time skin lesion classification while maintaining a small model size suitable for mobile deployment.

The key focus is on:
* Developing a lightweight CNN architecture optimized for mobile devices
* Converting the model to TFLite format for edge deployment
* Building an intuitive GUI for instant predictions
* Demonstrating model optimization and edge deployment skills

---

## Dataset Description

We are using the **Skin Cancer Dataset** (or similar skin lesion dataset).

### Target Variable

* `class` → (benign, malignant)
  * This makes it a **binary classification problem**

### Feature Categories

1. **Image Features**

   * Input: 128x128 RGB skin lesion images
   * Preprocessing: Rescaling, rotation, flip augmentation

2. **Model Architecture**

   * Convolutional layers (32 → 64 → 128 filters)
   * MaxPooling layers for downsampling
   * Dense layers with dropout for classification

---

## What Was Implemented

### Core Components

* **Lightweight CNN Model**
  * 3 convolutional blocks with increasing filters
  * MaxPooling for spatial reduction
  * Fully connected layers with dropout
  * Total parameters: ~1.7M (lightweight for mobile)

* **Model Optimization**
  * TFLite conversion for mobile deployment
  * Float16 quantization for 50% smaller model
  * Model size: ~3.25 MB (TFLite)

* **Data Augmentation**
  * Rotation (20 degrees)
  * Width/height shift (20%)
  * Shear (20%)
  * Zoom (20%)
  * Horizontal flip

---

## Advanced Part (Bonus)

* **Model Conversion**
  * TensorFlow to TFLite conversion
  * Mobile-optimized inference
  * ONNX conversion (optional)

* **Edge Deployment**
  * TFLite model ready for Android/iOS deployment
  * Low memory footprint (~3.25 MB)
  * Fast inference time

---

## Project Structure

```
Final Project/
│
├── data/                   # Dataset files
│   ├── train/
│   │   ├── benign/       # Training benign images
│   │   └── malignant/    # Training malignant images
│   └── test/
│       ├── benign/       # Test benign images
│       └── malignant/    # Test malignant images
│
├── src/                   # Source code
│   ├── train.py          # CNN training script
│   ├── predict.py      # Prediction script
│   ├── convert_tflite.py # TFLite conversion
│
├── gui/                   # GUI application
│   └── app.py          # Mobile Skin Analyzer GUI
│
├── model/                 # Trained models
│   ├── skin_cancer_model.h5
│   └── skin_cancer_model.tflite
│
├── README.md             # This file
```

---

## Workflow

### 1. Data Preparation

* Organize images in train/test folders
* Apply data augmentation
* Normalize pixel values (0-1)

---

### 2. Model Training

* Run `python src/train.py`
* Training with validation split
* Early stopping for best model

---

### 3. Model Conversion

* Convert to TFLite: `python src/convert_tflite.py`
* Convert to ONNX (optional): `python src/convert_onnx.py`

---

### 4. Prediction

* Single image: `python src/predict.py image.jpg --model tflite`
* GUI app: `python gui/app.py`

---

### 5. Mobile Deployment

* Use the TFLite model with TensorFlow Lite Android/iOS APIs
* Add camera permission to app manifest

---

## Expected Results

* **Training Accuracy**: ~80%
* **Validation Accuracy**: ~70-75%
* **Model Size**: ~3.25 MB (TFLite)
* **Inference Time**: ~50ms on mobile device

---

## Model Architecture

```
┌─────────────────────────────────────┐
│ Conv2D (32 filters, 3x3)            │
│ MaxPooling (2x2)                    │
├─────────────────────────────────────┤
│ Conv2D (64 filters, 3x3)            │
│ MaxPooling (2x2)                    │
├─────────────────────────────────────┤
│ Conv2D (128 filters, 3x3)           │
│ MaxPooling (2x2)                    │
├─────────────────────────────────────┤
│ Flatten                             │
│ Dense (64, ReLU)                    │
│ Dropout (0.5)                       │
│ Dense (1, Sigmoid)                  │
└─────────────────────────────────────┘
Total Parameters: ~1.7M
```

---

## Usage

### Train the Model
```bash
python src/train.py
```

### Convert to TFLite
```bash
python src/convert_tflite.py
```

### Run Prediction
```bash
python src/predict.py path/to/image.jpg --model tflite
```

### Run GUI App
```bash
python gui/app.py
```

---

## Output Examples

### Benign (Non-Cancerous)
```
RESULT: BENIGN (Non-Cancerous)
Confidence: 75.3%
This lesion appears to be benign (NOT cancer).
However, consult a dermatologist for confirmation.
```

### Malignant (Cancerous)
```
RESULT: MALIGNANT (Cancerous)
Confidence: 82.1%
WARNING: This lesion shows signs of potential malignancy.
Please consult a dermatologist for professional diagnosis.
```

---

## Technical Details

| Component | Details |
|-----------|---------|
| Image Size | 128x128 RGB |
| Model Type | Lightweight CNN |
| Total Parameters | ~1.7M |
| H5 Model Size | ~6.5 MB |
| TFLite Model Size | ~3.25 MB |
| Training Epochs | 10 (default) |
| Batch Size | 64 |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |

---

## Technologies Used

* **TensorFlow/Keras** - Deep learning framework
* **NumPy** - Numerical computing
* **Pillow** - Image processing
* **Tkinter** - GUI framework
* **TensorFlow Lite** - Mobile deployment

---

## Limitations & Notes

* Model is for educational/demonstration purposes only
* Not a substitute for professional medical advice
* Results may vary with different skin types
* More training data and GPU training would improve accuracy

---

## Goal

Build a **professional, well-structured CNN project** that demonstrates:

* Understanding of CNN architecture design
* Model optimization for mobile deployment
* Clean implementation
* Edge deployment skills (TFLite conversion)

---

## Disclaimer

This tool is for educational and demonstration purposes only. Always consult a dermatologist for professional medical diagnosis. The model predictions should not be used as the sole basis for medical decisions.