import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os

# Step 1: Create a sample recyclable waste classification model
# (In practice, you'd load your pre-trained model)
def create_sample_model(input_shape=(224, 224, 3), num_classes=6):
    """
    Creates a simple CNN for recyclable waste classification
    Classes: cardboard, glass, metal, paper, plastic, trash
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Step 2: Convert model to TensorFlow Lite
def convert_to_tflite(model, model_path='recyclable_model.tflite', quantize=False):
    """
    Converts a Keras model to TensorFlow Lite format
    
    Args:
        model: Keras model to convert
        model_path: Path to save the .tflite file
        quantize: Whether to apply quantization for smaller model size
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # Apply dynamic range quantization for smaller model size
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("Applying quantization for model compression...")
    
    tflite_model = converter.convert()
    
    # Save the model
    with open(model_path, 'wb') as f:
        f.write(tflite_model)
    
    model_size = os.path.getsize(model_path) / 1024  # Size in KB
    print(f"✓ TFLite model saved to: {model_path}")
    print(f"✓ Model size: {model_size:.2f} KB")
    
    return tflite_model

# Step 3: Load and run TFLite model
class TFLiteModel:
    def __init__(self, model_path):
        """Initialize TFLite interpreter"""
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Input type: {self.input_details[0]['dtype']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
    
    def predict(self, input_data):
        """Run inference on input data"""
        # Ensure input data is the correct type
        input_data = input_data.astype(self.input_details[0]['dtype'])
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensor
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return output_data

# Step 4: Test on sample dataset
def test_tflite_model(original_model, tflite_model_path, num_samples=10):
    """
    Test TFLite model and compare with original model
    """
    # Class names for recyclable waste
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    # Create sample test data
    input_shape = original_model.input_shape[1:]  # (224, 224, 3)
    test_data = np.random.rand(num_samples, *input_shape).astype(np.float32)
    
    print(f"\n{'='*60}")
    print("Testing TFLite Model on Sample Dataset")
    print(f"{'='*60}\n")
    
    # Load TFLite model
    tflite_model = TFLiteModel(tflite_model_path)
    
    # Compare predictions
    print(f"Running inference on {num_samples} samples...\n")
    
    differences = []
    
    for i in range(num_samples):
        # Prepare single sample
        sample = np.expand_dims(test_data[i], axis=0)
        
        # Original model prediction
        original_pred = original_model.predict(sample, verbose=0)
        original_class = np.argmax(original_pred)
        original_confidence = np.max(original_pred) * 100
        
        # TFLite model prediction
        tflite_pred = tflite_model.predict(sample)
        tflite_class = np.argmax(tflite_pred)
        tflite_confidence = np.max(tflite_pred) * 100
        
        # Calculate difference
        pred_diff = np.abs(original_pred - tflite_pred).mean()
        differences.append(pred_diff)
        
        print(f"Sample {i+1}:")
        print(f"  Original: {class_names[original_class]} ({original_confidence:.2f}%)")
        print(f"  TFLite:   {class_names[tflite_class]} ({tflite_confidence:.2f}%)")
        print(f"  Difference: {pred_diff:.6f}")
        print()
    
    # Summary statistics
    avg_diff = np.mean(differences)
    max_diff = np.max(differences)
    
    print(f"{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    print(f"Average prediction difference: {avg_diff:.6f}")
    print(f"Maximum prediction difference: {max_diff:.6f}")
    print(f"\n✓ Conversion successful! TFLite model is ready for deployment.")

# Main execution
if __name__ == "__main__":
    print("Recyclable Waste Model - TensorFlow Lite Conversion\n")
    
    # Step 1: Create or load model
    print("Step 1: Creating sample model...")
    model = create_sample_model()
    print(f"✓ Model created with {model.count_params():,} parameters\n")
    
    # Step 2: Convert to TFLite
    print("Step 2: Converting to TensorFlow Lite...")
    tflite_model_path = 'recyclable_model.tflite'
    convert_to_tflite(model, tflite_model_path, quantize=True)
    print()
    
    # Step 3: Test the model
    print("Step 3: Testing TFLite model...")
    test_tflite_model(model, tflite_model_path, num_samples=5)
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Deploy the .tflite model to your mobile/edge device")
    print("2. Use TensorFlow Lite runtime for inference")
    print("3. For Android: Use TensorFlow Lite Android library")
    print("4. For iOS: Use TensorFlow Lite iOS library")
    print("5. For embedded: Use TensorFlow Lite for Microcontrollers")