# 🧠 Real vs AI-Generated Image Detection

This project is a deep learning-based web application that detects whether an image is **real** or **AI-generated** (e.g., deepfakes or synthetic content). It uses a **Convolutional Neural Network (CNN)** for binary classification and is deployed using a simple **Flask** web interface.

---


## 🎯 Objective

The rise of AI-generated images has made it difficult to distinguish real from fake visuals. This tool helps:

- Detect fake media content
- Support journalists, researchers, and the public
- Educate users about AI-generated media

---

## 🗃️ Dataset

- Source: [Kaggle](https://www.kaggle.com/)
- **Categories**: Real Images & AI-Generated Images
- Preprocessing:
  - Resized to 128x128 pixels
  - Normalized pixel values (0–1)
  - Data augmentation: rotation, flipping, zooming
- Split: 80% training, 20% validation

---

## 🏗️ Model Architecture

Built using TensorFlow & Keras:

- 3 × Conv2D + MaxPooling2D layers
- Flatten layer
- Dense(128) + ReLU activation
- Dropout(0.5) to reduce overfitting
- Final Dense(1) with **Sigmoid** activation

### Loss & Optimizer:
- Binary Crossentropy
- Adam Optimizer

### Performance:
- Achieved 84% accuracy on the test 

---

## 🌐 Web App (Flask)

- Upload image via browser
- App resizes, normalizes, and feeds it to the model
- Model returns:
