from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load the trained CNN model
MODEL_PATH = os.path.join(os.getcwd(), "model.h5")  # Absolute path
IMAGE_SIZE = (128, 128)  # Match model input size

# Load model safely
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  # Handle error if model loading fails

@app.route('/')
def home():
    return render_template('index.html')  # Serves the front-end

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Load and preprocess the image
        image = Image.open(file).convert("RGB")  # Ensure 3 channels
        image = image.resize(IMAGE_SIZE)  # Resize to model input shape
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make a prediction
        prediction = model.predict(image)[0][0]  
        label = "Real" if prediction < 0.5 else "AI-Generated"

        return jsonify({
            "result": label,
            "confidence": float(prediction)
        })

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({"error": "Error processing image"}), 500

if __name__ == '__main__':
    app.run(debug=True)
