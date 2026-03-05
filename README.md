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
  <img src="https://img.shields.io/badge/Status-Actively%20Maintained-brightgreen?style=for-the-badge"/>
</p>

---

## 📌 Overview

SkinScan-AI is a deep learning-powered web application that predicts potential skin diseases from uploaded images.

Built using PyTorch and deployed with Streamlit, this project demonstrates the practical implementation of Convolutional Neural Networks (CNNs) in medical image classification.

The system showcases end-to-end integration of:..

- Image preprocessing  
- Deep learning model inference  
- Web-based deployment  
- Modular Python architecture  

---

## 🚀 Development Status

This project is actively maintained and continuously enhanced.

Ongoing improvements include:

- Model optimization  
- UI refinement  
- Performance tuning  
- Codebase refactoring  
- Explainability enhancements  
- Deployment readiness updates  

Regular commits are pushed to ensure scalability and long-term maintainability.

---

## 🧠 Problem Statement

Skin diseases often present visually similar symptoms, making early-stage identification challenging without professional expertise.

This project aims to assist in preliminary classification of skin conditions using computer vision and deep learning techniques.

⚠️ Note: This system is intended for educational and demonstration purposes only. It does not replace medical diagnosis by certified professionals.

---

## 🏗️ System Architecture

The application follows a modular pipeline architecture:

### 1️⃣ User Interface Layer (Streamlit)
- Handles image uploads  
- Displays predictions  
- Provides interactive UI components  

### 2️⃣ Preprocessing Layer
- Image resizing  
- Normalization  
- Tensor conversion  
- Ensures compatibility with trained CNN model  

### 3️⃣ Model Inference Layer
- Loads trained PyTorch CNN model  
- Performs forward propagation  
- Generates prediction probabilities  

### 4️⃣ Post-processing Layer
- Converts output tensor to readable class labels  
- Displays final predicted result  
- Optionally shows confidence scores  

---

## 🧩 Module Breakdown

### `app.py`
- Entry point of the application  
- Manages Streamlit interface  
- Connects frontend with backend inference logic  

### `model_utils.py`
- Loads trained PyTorch model  
- Handles inference pipeline  
- Maps output indices to disease categories  

### `image_processor.py`
- Handles image preprocessing  
- Applies resizing and normalization  
- Converts images to tensors  

### `accuracy_optimizer.py`
- Contains model evaluation utilities  
- Designed for monitoring training performance  
- Assists in hyperparameter tuning  

---

## 🧠 Model Details

- Model Type: Convolutional Neural Network (CNN)  
- Framework: PyTorch  
- Input: RGB Skin Image  
- Output: Predicted Disease Class  
- Inference Type: Single Image Classification  
- Loss Function: CrossEntropyLoss  
- Optimizer: Adam  

The CNN extracts hierarchical spatial features using convolutional layers and learns complex visual patterns to classify skin disease categories.

---

## 📊 Model Performance

(Update with actual metrics if available)

- Training Accuracy: XX%  
- Validation Accuracy: XX%  
- Training Loss: XX  
- Validation Loss: XX  

Performance depends heavily on dataset quality, diversity, and preprocessing techniques.

---

## ⚙️ Installation & Setup

Clone repository:

```bash
git clone https://github.com/shanky-ux/SkinScan-AI.git
cd SkinScan-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit application:

```bash
streamlit run app.py
```

Application runs locally at:

```
http://localhost:8501
```

---

## 🌍 Deployment Options

SkinScan-AI can be deployed using:

- Streamlit Cloud  
- Render  
- Railway  
- Docker containerization  
- AWS / Azure VM  

---

## 🔐 Limitations

- Requires clear, high-quality image input  
- Limited to trained disease categories  
- Not medically certified  
- Accuracy depends on dataset size and diversity  
- May struggle with unseen skin conditions  

---

## 📈 Future Enhancements

- Confidence score visualization  
- Grad-CAM model explainability  
- Expanded dataset for better generalization  
- REST API integration  
- User authentication system  
- Prediction history storage in database  
- Mobile-friendly interface  
- Model quantization for faster inference  

---

## 🎓 Learning Outcomes

Through this project, the following concepts were implemented:

- Deep learning model training and inference  
- Image preprocessing pipelines  
- CNN architecture understanding  
- Model deployment using Streamlit  
- Modular Python project structure  
- AI + Web integration workflow  

---

## 📅 Development Log

This section is updated regularly:

- Improved preprocessing pipeline  
- Optimized CNN architecture  
- Enhanced inference speed  
- Refactored model loading logic  
- Improved UI responsiveness  
- Added prediction formatting improvements  

(Continuously evolving)

---

## 🎯 Why This Project Stands Out

- Demonstrates real-world AI application  
- Shows end-to-end ML deployment capability  
- Clean modular Python architecture  
- Integrates Deep Learning with Web UI  
- Portfolio-ready AI healthcare demo  

---

## 👨‍💻 Author

Ravi Shankar  
B.Tech CSE (AIML)  
AI & Full Stack Enthusiast  

GitHub: https://github.com/shanky-ux  

---

## 📜 License

This project is licensed under the MIT License.
