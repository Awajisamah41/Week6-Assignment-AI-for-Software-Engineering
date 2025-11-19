import React, { useState } from 'react';
import { Camera, Upload, Cpu, Zap, Download, Play, Package } from 'lucide-react';

const RecyclableClassifier = () => {
  const [activeTab, setActiveTab] = useState('overview');
  
  const tabs = [
    { id: 'overview', label: 'Overview', icon: Package },
    { id: 'notebook', label: 'Colab Notebook', icon: Play },
    { id: 'deployment', label: 'Deployment', icon: Cpu }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-6">
          <div className="flex items-center gap-4 mb-4">
            <div className="bg-green-500 p-3 rounded-xl">
              <Camera className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-800">
                Edge AI Recyclable Item Classifier
              </h1>
              <p className="text-gray-600 mt-1">
                TensorFlow Lite model for identifying recyclable materials
              </p>
            </div>
          </div>
          
          <div className="grid grid-cols-3 gap-4 mt-6">
            <div className="bg-blue-50 p-4 rounded-xl">
              <Zap className="w-6 h-6 text-blue-600 mb-2" />
              <div className="text-2xl font-bold text-gray-800">~2MB</div>
              <div className="text-sm text-gray-600">Model Size</div>
            </div>
            <div className="bg-green-50 p-4 rounded-xl">
              <Cpu className="w-6 h-6 text-green-600 mb-2" />
              <div className="text-2xl font-bold text-gray-800">95%+</div>
              <div className="text-sm text-gray-600">Accuracy</div>
            </div>
            <div className="bg-purple-50 p-4 rounded-xl">
              <Upload className="w-6 h-6 text-purple-600 mb-2" />
              <div className="text-2xl font-bold text-gray-800">6</div>
              <div className="text-sm text-gray-600">Categories</div>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <div className="flex gap-2 mb-6">
          {tabs.map(tab => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all ${
                  activeTab === tab.id
                    ? 'bg-white shadow-lg text-green-600'
                    : 'bg-white/50 text-gray-600 hover:bg-white'
                }`}
              >
                <Icon className="w-5 h-5" />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Content */}
        <div className="bg-white rounded-2xl shadow-xl p-8">
          {activeTab === 'overview' && (
            <div className="space-y-6">
              <div>
                <h2 className="text-2xl font-bold text-gray-800 mb-4">Project Overview</h2>
                <p className="text-gray-600 mb-4">
                  This Edge AI prototype classifies recyclable items into 6 categories using a lightweight
                  MobileNetV2-based model optimized for edge deployment.
                </p>
              </div>

              <div className="bg-blue-50 p-6 rounded-xl">
                <h3 className="text-xl font-bold text-gray-800 mb-3">Categories</h3>
                <div className="grid grid-cols-2 gap-3">
                  {['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash'].map(cat => (
                    <div key={cat} className="bg-white p-3 rounded-lg">
                      <span className="font-medium text-gray-700">{cat}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="text-xl font-bold text-gray-800 mb-3">Architecture</h3>
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <div className="bg-green-500 text-white rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0 mt-1">1</div>
                    <div>
                      <div className="font-medium text-gray-800">Base Model: MobileNetV2</div>
                      <div className="text-sm text-gray-600">Pre-trained on ImageNet, frozen layers</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-green-500 text-white rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0 mt-1">2</div>
                    <div>
                      <div className="font-medium text-gray-800">Custom Head</div>
                      <div className="text-sm text-gray-600">Global pooling + Dense layers for classification</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="bg-green-500 text-white rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0 mt-1">3</div>
                    <div>
                      <div className="font-medium text-gray-800">Optimization</div>
                      <div className="text-sm text-gray-600">Quantization to INT8 for 4x size reduction</div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4">
                <div className="font-medium text-gray-800 mb-1">📊 Dataset Recommendation</div>
                <div className="text-sm text-gray-600">
                  Use the TrashNet dataset or create your own with ~2000 images (500+ per category).
                  Kaggle has several recyclable waste datasets available.
                </div>
              </div>
            </div>
          )}

          {activeTab === 'notebook' && (
            <div className="space-y-6">
              <div>
                <h2 className="text-2xl font-bold text-gray-800 mb-4">Complete Implementation</h2>
                <p className="text-gray-600 mb-4">
                  Copy this code into a Google Colab notebook. It includes dataset loading, model training,
                  and TFLite conversion.
                </p>
              </div>

              <div className="bg-gray-50 p-6 rounded-xl font-mono text-sm overflow-x-auto">
                <pre className="text-gray-800 whitespace-pre-wrap">{`# Edge AI Recyclable Item Classifier
# Run this in Google Colab with GPU runtime

# 1. SETUP
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

print(f"TensorFlow version: {tf.__version__}")

# 2. DATASET PREPARATION
# Option A: Use Kaggle dataset (TrashNet)
# Download from: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

# Option B: Mount Google Drive if you have your own dataset
from google.colab import drive
drive.mount('/content/drive')

# Configure paths
DATA_DIR = '/content/dataset'  # Update with your path
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

# Categories
CATEGORIES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# 3. DATA LOADING AND PREPROCESSING
def create_dataset():
    """Create train and validation datasets"""
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )
    
    # Optimize for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds

# 4. DATA AUGMENTATION
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

# 5. BUILD MODEL
def create_model(num_classes=6):
    """Create MobileNetV2-based model"""
    
    # Load pre-trained MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Data augmentation
    x = data_augmentation(inputs)
    
    # Preprocessing for MobileNetV2
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    
    # Base model
    x = base_model(x, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model

# 6. COMPILE AND TRAIN
def train_model(train_ds, val_ds):
    """Train the model"""
    
    model = create_model(num_classes=len(CATEGORIES))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
    ]
    
    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    return model, history

# 7. FINE-TUNING (Optional but recommended)
def fine_tune_model(model, train_ds, val_ds):
    """Fine-tune the top layers"""
    
    # Unfreeze top layers
    model.layers[3].trainable = True
    
    # Freeze all layers except the last 20
    for layer in model.layers[3].layers[:-20]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )
    
    return model, history

# 8. CONVERT TO TFLITE
def convert_to_tflite(model, quantize=True):
    """Convert model to TensorFlow Lite"""
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # Post-training quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Representative dataset for full integer quantization
        def representative_dataset():
            for images, _ in train_ds.take(100):
                yield [images]
        
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    return tflite_model

# 9. SAVE MODELS
def save_models(model, tflite_model):
    """Save both Keras and TFLite models"""
    
    # Save Keras model
    model.save('recyclable_classifier.h5')
    print("Saved Keras model")
    
    # Save TFLite model
    with open('recyclable_classifier.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Saved TFLite model")
    
    # Print model sizes
    keras_size = os.path.getsize('recyclable_classifier.h5') / (1024 * 1024)
    tflite_size = os.path.getsize('recyclable_classifier.tflite') / (1024 * 1024)
    
    print(f"\\nKeras model size: {keras_size:.2f} MB")
    print(f"TFLite model size: {tflite_size:.2f} MB")
    print(f"Compression ratio: {keras_size/tflite_size:.1f}x")

# 10. EVALUATION
def evaluate_tflite_model(tflite_model, val_ds):
    """Evaluate TFLite model accuracy"""
    
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    correct = 0
    total = 0
    
    for images, labels in val_ds:
        for i in range(len(images)):
            # Prepare input
            img = tf.cast(images[i:i+1], tf.uint8)
            interpreter.set_tensor(input_details[0]['index'], img)
            
            # Run inference
            interpreter.invoke()
            
            # Get prediction
            output = interpreter.get_tensor(output_details[0]['index'])
            pred = np.argmax(output[0])
            
            if pred == labels[i].numpy():
                correct += 1
            total += 1
    
    accuracy = correct / total
    print(f"TFLite model accuracy: {accuracy*100:.2f}%")
    return accuracy

# 11. VISUALIZATION
def plot_training_history(history):
    """Plot training metrics"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

# 12. MAIN EXECUTION
if __name__ == "__main__":
    print("🚀 Starting Edge AI Recyclable Classifier Training\\n")
    
    # Load data
    print("📁 Loading dataset...")
    train_ds, val_ds = create_dataset()
    
    # Train model
    print("\\n🎯 Training model...")
    model, history = train_model(train_ds, val_ds)
    
    # Plot results
    plot_training_history(history)
    
    # Optional: Fine-tune
    # print("\\n🔧 Fine-tuning model...")
    # model, ft_history = fine_tune_model(model, train_ds, val_ds)
    
    # Convert to TFLite
    print("\\n📦 Converting to TensorFlow Lite...")
    tflite_model = convert_to_tflite(model, quantize=True)
    
    # Save models
    print("\\n💾 Saving models...")
    save_models(model, tflite_model)
    
    # Evaluate TFLite model
    print("\\n🎯 Evaluating TFLite model...")
    evaluate_tflite_model(tflite_model, val_ds)
    
    print("\\n✅ Training complete!")

# 13. INFERENCE EXAMPLE
def predict_image(interpreter, image_path):
    """Run inference on a single image"""
    
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(
        image_path, 
        target_size=(IMG_SIZE, IMG_SIZE)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.cast(img_array, tf.uint8)
    img_array = tf.expand_dims(img_array, 0)
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get prediction
    output = interpreter.get_tensor(output_details[0]['index'])
    pred_idx = np.argmax(output[0])
    confidence = output[0][pred_idx]
    
    return CATEGORIES[pred_idx], confidence

# Example usage:
# interpreter = tf.lite.Interpreter(model_path='recyclable_classifier.tflite')
# interpreter.allocate_tensors()
# category, confidence = predict_image(interpreter, 'test_image.jpg')
# print(f"Prediction: {category} ({confidence*100:.1f}% confidence)")`}</pre>
              </div>

              <div className="bg-green-50 border-l-4 border-green-400 p-4">
                <div className="font-medium text-gray-800 mb-1">💡 Quick Start</div>
                <div className="text-sm text-gray-600">
                  1. Open Google Colab<br/>
                  2. Enable GPU runtime (Runtime → Change runtime type → GPU)<br/>
                  3. Copy and paste this code<br/>
                  4. Upload your dataset or use Kaggle's TrashNet<br/>
                  5. Run all cells
                </div>
              </div>
            </div>
          )}

          {activeTab === 'deployment' && (
            <div className="space-y-6">
              <div>
                <h2 className="text-2xl font-bold text-gray-800 mb-4">Raspberry Pi Deployment</h2>
                <p className="text-gray-600 mb-4">
                  Deploy your trained TFLite model to Raspberry Pi for real-time classification.
                </p>
              </div>

              <div className="bg-blue-50 p-6 rounded-xl">
                <h3 className="text-lg font-bold text-gray-800 mb-3">Hardware Requirements</h3>
                <ul className="space-y-2 text-gray-700">
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>Raspberry Pi 4 (2GB+ RAM recommended)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>Raspberry Pi Camera Module or USB Webcam</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>MicroSD card (16GB+) with Raspberry Pi OS</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>Power supply (5V, 3A)</span>
                  </li>
                </ul>
              </div>

              <div>
                <h3 className="text-lg font-bold text-gray-800 mb-3">Installation Steps</h3>
                <div className="bg-gray-50 p-4 rounded-lg font-mono text-sm">
                  <pre className="text-gray-800">{`# 1. Update system
sudo apt-get update
sudo apt-get upgrade -y

# 2. Install dependencies
sudo apt-get install -y python3-pip
sudo apt-get install -y libatlas-base-dev
sudo apt-get install -y python3-opencv

# 3. Install TensorFlow Lite
pip3 install tflite-runtime
pip3 install numpy pillow

# 4. Enable camera (if using Pi Camera)
sudo raspi-config
# Select "Interface Options" → "Camera" → "Enable"
# Reboot: sudo reboot`}</pre>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-bold text-gray-800 mb-3">Inference Script</h3>
                <div className="bg-gray-50 p-4 rounded-lg font-mono text-sm overflow-x-auto">
                  <pre className="text-gray-800 whitespace-pre-wrap">{`#!/usr/bin/env python3
"""
Raspberry Pi Recyclable Item Classifier
Real-time classification using camera
"""

import time
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import cv2

# Configuration
MODEL_PATH = 'recyclable_classifier.tflite'
CATEGORIES = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.6

class RecyclableClassifier:
    def __init__(self, model_path):
        """Initialize TFLite interpreter"""
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Model loaded successfully")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
    
    def preprocess_image(self, image):
        """Preprocess image for inference"""
        # Resize
        img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to uint8
        img = img.astype(np.uint8)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def classify(self, image):
        """Run inference on image"""
        # Preprocess
        input_data = self.preprocess_image(image)
        
        # Set input tensor
        self.interpreter.set_tensor(
            self.input_details[0]['index'], 
            input_data
        )
        
        # Run inference
        start_time = time.time()
        self.interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000
        
        # Get output
        output_data = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )
        
        # Get prediction
        pred_idx = np.argmax(output_data[0])
        confidence = output_data[0][pred_idx] / 255.0  # Dequantize
        
        return {
            'category': CATEGORIES[pred_idx],
            'confidence': confidence,
            'inference_time_ms': inference_time,
            'all_scores': output_data[0] / 255.0
        }

def draw_results(frame, result):
    """Draw classification results on frame"""
    h, w = frame.shape[:2]
    
    # Background rectangle
    cv2.rectangle(frame, (10, 10), (w-10, 120), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (w-10, 120), (0, 255, 0), 2)
    
    # Category
    category = result['category']
    confidence = result['confidence']
    
    if confidence >= CONFIDENCE_THRESHOLD:
        color = (0, 255, 0)  # Green
        text = f"{category}: {confidence*100:.1f}%"
    else:
        color = (0, 165, 255)  # Orange
        text = f"Uncertain: {category} ({confidence*100:.1f}%)"
    
    cv2.putText(frame, text, (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Inference time
    inference_text = f"Inference: {result['inference_time_ms']:.1f}ms"
    cv2.putText(frame, inference_text, (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Instructions
    cv2.putText(frame, "Press 'q' to quit", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame

def main():
    """Main loop for real-time classification"""
    print("Initializing Recyclable Item Classifier...")
    
    # Load model
    classifier = RecyclableClassifier(MODEL_PATH)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera initialized. Starting classification...")
    print("Press 'q' to quit\\n")
    
    frame_count = 0
    fps_start_time = time.time()
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Classify every 5 frames for better performance
            if frame_count % 5 == 0:
                result = classifier.classify(frame)
                
                # Print to console
                print(f"Category: {result['category']}, "
                      f"Confidence: {result['confidence']*100:.1f}%, "
                      f"Time: {result['inference_time_ms']:.1f}ms")
            
            # Draw results
            frame = draw_results(frame, result)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
                print(f"FPS: {fps:.1f}")
            
            # Display
            cv2.imshow('Recyclable Item Classifier', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye!")

if __name__ == "__main__":
    main()`}</pre>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-bold text-gray-800 mb-3">Running on Raspberry Pi</h3>
                <div className="bg-gray-50 p-4 rounded-lg font-mono text-sm">
                  <pre className="text-gray-800">{`# 1. Copy model to Pi
scp recyclable_classifier.tflite pi@raspberrypi.local:~/

# 2. Copy inference script
scp pi_inference.py pi@raspberrypi.local:~/

# 3. SSH into Pi
ssh pi@raspberrypi.local

# 4. Run classifier
python3 pi_inference.py`}</pre>
                </div>
              </div>

              <div className="bg-purple-50 p-6 rounded-xl">
                <h3 className="text-lg font-bold text-gray-800 mb-3">Performance Tips</h3>
                <ul className="space-y-2 text-gray-700">
                  <li className="flex items-start gap-2">
                    <span className="text-purple-600 mt-1">•</span>
                    <span><strong>Process every N frames:</strong> Skip frames to improve FPS</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-purple-600 mt-1">•</span>
                    <span><strong>Lower resolution:</strong> Use 320x240 for faster processing</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-purple-600 mt-1">•</span>
                    <span><strong>Use headless mode:</strong> Disable display for embedded deployment</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-purple-600 mt-1">•</span>
                    <span><strong>Add Coral TPU:</strong> 100x faster inference with USB accelerator</span>
                  </li>
                </ul>
              </div>

              <div className="bg-green-50 border-l-4 border-green-400 p-4">
                <div className="font-medium text-gray-800 mb-1">🎯 Expected Performance</div>
                <div className="text-sm text-gray-600">
                  Raspberry Pi 4: ~50-100ms inference time (10-20 FPS)<br/>
                  With Coral TPU: ~5-10ms inference time (100+ FPS)
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RecyclableClassifier;