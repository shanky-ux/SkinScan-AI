<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:11998e,50:38ef7d,100:0575E6&height=200&section=header&text=SkinScan%20AI&fontSize=45&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=AI-Powered%20Skin%20Disease%20Detection%20System&descAlignY=55&descSize=18"/>
</p>

<p align="center">
  <b>🩺 Early Skin Disease Detection Using Deep Learning</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch"/>
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit"/>
  <img src="https://img.shields.io/badge/EfficientNet-B3-success?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge"/>
</p>

---

# 📌 Overview

**SkinScan AI** is an AI-powered skin disease detection system that classifies skin conditions from uploaded images using a deep learning model based on **EfficientNet-B3**.

The project helps users perform an initial screening of common skin diseases within seconds through an intuitive web application. It demonstrates the practical application of Artificial Intelligence in healthcare by combining computer vision, deep learning, and an interactive web interface.

---

# 🏗️ System Architecture

```mermaid
flowchart TD

A[User Uploads Skin Image]
--> B[Streamlit Web Interface]

B --> C[Image Preprocessing]

C --> D[EfficientNet-B3 CNN]

D --> E[Softmax Classification]

E --> F[Predicted Disease]

F --> G[Confidence Score]

G --> H[Display Results]
```

---

# 🔄 Prediction Workflow

```mermaid
flowchart TD

A[Upload Image]
--> B[Resize Image]

B --> C[Normalize Pixels]

C --> D[EfficientNet-B3 Model]

D --> E[Generate Prediction]

E --> F[Confidence Score]

F --> G[Display Disease Name]
```

---

# ☁️ Application Flow

```mermaid
flowchart LR

User --> Streamlit

Streamlit --> ImageProcessing

ImageProcessing --> AIModel

AIModel --> Prediction

Prediction --> ResultPage
```

---

# 🔁 Inference Lifecycle

```mermaid
sequenceDiagram

participant U as User

participant W as Streamlit App

participant M as AI Model

U->>W: Upload Skin Image

W->>M: Process Image

M-->>W: Disease Prediction

W-->>U: Show Disease + Confidence
```

---

# 🚀 Current Development

SkinScan AI continues to evolve with improvements including:

- Better prediction accuracy
- Faster inference
- Additional skin disease classes
- Improved UI/UX
- Mobile optimization
- Explainable AI visualizations (Grad-CAM)
- Cloud deployment

---

# ✨ Key Features

- 🤖 Deep Learning based Skin Disease Classification
- 📷 Upload skin lesion images
- ⚡ Real-time prediction
- 📊 Confidence score visualization
- 🧠 EfficientNet-B3 CNN architecture
- 🌐 Interactive Streamlit interface
- 📱 Responsive web application
- 🩺 Supports multiple skin diseases

---

# 🤖 AI Model

SkinScan AI uses **EfficientNet-B3**, one of Google's state-of-the-art CNN architectures optimized for medical image classification.

## Workflow

1. Upload skin image

2. Image preprocessing

3. EfficientNet-B3 feature extraction

4. Classification layer predicts disease

5. Confidence score generated

6. Results displayed

---

## Model Information

| Parameter | Value |
|------------|--------|
| Architecture | EfficientNet-B3 |
| Framework | PyTorch |
| Image Size | 300×300 |
| Classes | 11 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |

---

# 📂 Project Structure

```text
SkinScan-AI/
│
├── app.py
├── model_utils.py
├── image_processor.py
├── accuracy_optimizer.py
├── best_effnet_skin.h5
├── final_effnet_skin.h5
├── requirements.txt
├── assets/
├── models/
├── dataset/
├── screenshots/
└── README.md
```

---

# ⚙️ Installation

```bash
git clone https://github.com/shanky-ux/SkinScan-AI.git

cd SkinScan-AI

pip install -r requirements.txt

streamlit run app.py
```

---

# 📈 Model Performance

| Metric | Value |
|----------|--------|
| Accuracy | 95%+ |
| Deep Learning Model | EfficientNet-B3 |
| Skin Diseases | 11 |
| Framework | PyTorch |
| Prediction Time | <2 Seconds |

---

# 🧬 Supported Skin Diseases

- Acne
- Actinic Keratosis
- Basal Cell Carcinoma
- Dermatofibroma
- Eczema
- Melanoma
- Nevus
- Psoriasis
- Seborrheic Keratosis
- Tinea Ringworm
- Normal Skin

---

# 🖥️ Technologies Used

| Category | Technologies |
|-----------|--------------|
| Language | Python |
| AI Framework | PyTorch |
| CNN Model | EfficientNet-B3 |
| Web Framework | Streamlit |
| Image Processing | OpenCV, Pillow |
| Data Science | NumPy, Pandas |
| Visualization | Matplotlib |

---

# 🎯 Why This Project Stands Out

- Healthcare-focused AI application
- Deep Learning with EfficientNet-B3
- End-to-end image classification pipeline
- Modern Streamlit interface
- Real-world medical imaging application
- Recruiter-friendly project demonstrating Computer Vision skills

---

# 👨‍💻 Author

**Ravi Shankar**

B.Tech Computer Science (AIML)

Machine Learning Engineer | AI Enthusiast

GitHub:
https://github.com/shanky-ux

---

# ⚠️ Medical Disclaimer

SkinScan AI is intended for educational and research purposes only.

The predictions generated by this application should **not** be considered a medical diagnosis. Always consult a qualified dermatologist for professional medical advice.

---

# 📜 License

Licensed under the MIT License.

---

<p align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0575E6,50:38ef7d,100:11998e&height=120&section=footer"/>
</p>

<p align="center">
⭐ If you found this project useful, consider giving it a star!
</p>

<p align="center">
<i>"Artificial Intelligence has the power to make healthcare more accessible, one prediction at a time."</i>
</p>
