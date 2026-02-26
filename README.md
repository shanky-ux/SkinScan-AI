<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=200&section=header&text=SkinScan-AI&fontSize=40&fontColor=ffffff&animation=fadeIn&fontAlignY=35"/>
</p>

<p align="center">
  <b>AI-Powered Skin Disease Classification using Deep Learning</b>
</p>

<p align="center">
  
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/PyTorch-DeepLearning-red?style=for-the-badge&logo=pytorch"/>
  <img src="https://img.shields.io/badge/Streamlit-WebApp-ff4b4b?style=for-the-badge&logo=streamlit"/>
  <img src="https://img.shields.io/badge/Computer%20Vision-CNN-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge"/>

</p>

---

## 📌 Overview

SkinScan-AI is a deep learning-based web application that predicts possible skin diseases from uploaded images.

It demonstrates the practical application of **Convolutional Neural Networks (CNNs)** in medical image classification using PyTorch and Streamlit.

---

## 🧠 Problem Statemen

Skin diseases often present visually similar symptoms, making early-stage identification difficult without professional expertise.  
This project aims to assist in preliminary classification of skin conditions using deep learning-based image recognition.

The goal is not to replace medical professionals but to demonstrate how computer vision can assist in medical diagnostics.

---

## 🏗️ System Architecture

The system follows a modular pipeline architecture:

1. **User Interface Layer (Streamlit)**
   - Handles image upload
   - Displays predictions
   - Provides interactive UI

2. **Preprocessing Layer**
   - Image resizing
   - Normalization
   - Tensor conversion
   - Ensures compatibility with trained CNN model

3. **Model Inference Layer**
   - Loads trained PyTorch CNN model
   - Performs forward pass
   - Generates prediction scores

4. **Post-processing Layer**
   - Converts output tensor to readable class label
   - Displays final prediction result

---

## 🧩 Module Breakdown

### `app.py`
- Entry point of the application
- Manages Streamlit UI
- Connects frontend with backend logic

### `model_utils.py`
- Loads trained PyTorch model
- Handles prediction logic
- Maps output indices to disease labels

### `image_processor.py`
- Preprocesses input images
- Applies resizing and normalization
- Converts images to tensors

### `accuracy_optimizer.py`
- Contains model evaluation utilities
- Designed for training performance monitoring

---

## 🧠 Model Details

- Model Type: Convolutional Neural Network (CNN)
- Framework: PyTorch
- Input: RGB Skin Image
- Output: Predicted Disease Class
- Inference Type: Single Image Classification

The CNN extracts spatial features from images using convolutional layers and applies learned weights to classify disease categories.

---

## 📊 Model Performance

*(Update these values if you have exact numbers)*

- Training Accuracy: XX%
- Validation Accuracy: XX%
- Loss Function: CrossEntropyLoss
- Optimizer: Adam

---

## 🚀 Deployment Readiness

This project can be deployed using:

- Streamlit Cloud
- Render
- Railway
- Docker containerization

---

## 🔐 Limitations

- Requires high-quality image input
- Limited to trained disease categories
- Not medically certified
- Model accuracy depends on dataset quality

---

## 📈 Future Enhancements

- Add confidence score visualization
- Implement Grad-CAM for model explainability
- Expand dataset for better generalization
- Add user authentication
- Store prediction history in database
- Convert into mobile-friendly application

---

## 🎓 Learning Outcomes

Through this project, the following concepts were implemented:

- Deep Learning model integration
- Image preprocessing pipelines
- Model inference in production environment
- Web deployment using Streamlit
- Modular Python architecture

---
