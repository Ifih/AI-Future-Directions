import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Load Fashion MNIST as proxy dataset (classify as recyclable/non-recyclable)
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Preprocess
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Binary classification: classes 0-4 recyclable, 5-9 non
train_labels_binary = (train_labels < 5).astype(int)
test_labels_binary = (test_labels < 5).astype(int)

# Build model
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
print("Training model...")
model.fit(train_images, train_labels_binary, epochs=5, validation_split=0.1, verbose=1)

# Evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels_binary, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
with open('recyclable_classifier.tflite', 'wb') as f:
    f.write(tflite_model)
print("TFLite model saved as 'recyclable_classifier.tflite'")

# Load and run inference
interpreter = tf.lite.Interpreter(model_path='recyclable_classifier.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Sample inference
sample_image = test_images[0:1].astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], sample_image)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
prediction = (output > 0.5).astype(int)
print(f"Sample Prediction: {'Recyclable' if prediction[0][0] else 'Non-Recyclable'}")

print("Edge AI app deployed and tested.")
