#Python
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import os

# --- STEP 1: Prepare the Model (Simulation of Training) ---
print("Loading base model (MobileNetV2)...")
# We use MobileNetV2 because it is specifically designed for mobile/edge devices
model = MobileNetV2(weights='imagenet', include_top=True)

# --- STEP 2: Convert to TensorFlow Lite (Edge Optimization) ---
print("Converting model to TensorFlow Lite format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Optimization: Quantization (reduces model size and latency for edge hardware)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the model locally
tflite_model_path = 'mobilenet_v2_quant.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
print(f"Model saved to {tflite_model_path}. Size: {len(tflite_model)/1024:.2f} KB")

# --- STEP 3: The Inference Engine (Simulating the Edge Device) ---
def run_edge_inference(tflite_path, image_path):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess the image (Resize to 224x224 as required by MobileNet)
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array) # MobileNet specific preprocessing

        # Run Inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Decode results (We use Keras utils here for mapping IDs to names)
        results = decode_predictions(output_data, top=3)[0]
        
        print("\n--- Edge AI Inference Results ---")
        print(f"Processing File: {image_path}")
        for i, (id, label, prob) in enumerate(results):
            print(f"{i+1}. {label}: {prob*100:.2f}% confidence")
            
    except Exception as e:
        print(f"Error: Could not process file. Details: {e}")

# --- STEP 4: User Interaction ---
print("\n" + "="*30)
print(" EDGE AI SIMULATOR READY")
print("="*30)

# Explicitly asking the user for their file choice as requested
user_file = input("Please enter the full path to your image file (e.g., /content/dog.jpg): ")

if os.path.exists(user_file):
    run_edge_inference(tflite_model_path, user_file)
else:
    print("File not found. Please check the path and try again.")
