# Edge AI App for Recyclable Classification

## Description
This Edge AI application demonstrates a convolutional neural network (CNN) model trained on the Fashion MNIST dataset to classify items as recyclable or non-recyclable. The model is converted to TensorFlow Lite (TFLite) format for efficient deployment on edge devices. It performs binary classification where classes 0-4 are considered recyclable and 5-9 are non-recyclable.

## Requirements
- Python 3.x
- TensorFlow 2.x
- NumPy

## Installation
1. Ensure Python 3.x is installed on your system.
2. Install required packages:
   ```
   pip install tensorflow numpy
   ```

## Usage
Run the script directly:
```
python edge_ai_app.py
```

The script will:
1. Load and preprocess the Fashion MNIST dataset.
2. Train a CNN model for binary classification.
3. Evaluate the model on test data.
4. Convert the model to TFLite format and save it as 'recyclable_classifier.tflite'.
5. Perform sample inference on a test image.

## Output
- Console output showing training progress, test accuracy, and sample prediction.
- TFLite model file: `recyclable_classifier.tflite`

## Notes
- Training may take several minutes depending on your hardware.
- The model uses Fashion MNIST as a proxy dataset for demonstration purposes.
- For real-world use, train on actual recyclable/non-recyclable image data.
