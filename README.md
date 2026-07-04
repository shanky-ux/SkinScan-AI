<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=200&section=header&text=SkinScan-AI&fontSize=45&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=AI-Powered%20Skin%20Disease%20Classification%20Platform&descAlignY=55&descSize=18"/>
</p>

<p align="center">
  <b>🩺 Instant Skin Condition Screening — Upload a Photo, Get AI-Backed Insights</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/Next.js-14-black?style=for-the-badge&logo=next.js"/>
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi"/>
  <img src="https://img.shields.io/badge/PyTorch-DeepLearning-EE4C2C?style=for-the-badge&logo=pytorch"/>
  <img src="https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  <a href="https://ravish4nkar.vercel.app" target="_blank">
    <img src="https://img.shields.io/badge/🌐%20Portfolio-ravish4nkar.vercel.app-blueviolet?style=for-the-badge"/>
  </a>
</p>

---

## 📌 Overview

**SkinScan-AI** is a full-stack, AI-assisted skin disease classification platform rebuilt from an original Streamlit prototype into a production-style medical-tech demo.

Skin conditions are among the most commonly self-diagnosed (and misdiagnosed) health issues — many people delay seeing a dermatologist simply because they don't know if a mark on their skin is benign or serious. SkinScan-AI lets a user upload a photo of a skin lesion and get an instant, AI-backed classification across 9 dermatological categories, complete with severity guidance, confidence scoring, and test-time-augmented inference for more reliable predictions.

The repo contains:

- `backend/` — FastAPI service (`skinscan_api`) for image validation, preprocessing, TTA inference, and disease metadata
- `frontend/` — Next.js 14 App Router UI with Tailwind CSS, Framer Motion, and Recharts
- `attached_assets/` — sample images and the default model checkpoint used by the backend
- `train_skin_disease_model.py` / `fetch_and_train.py` — training pipeline for the underlying CNN

---

## 🏗️ System Architecture

```mermaid
flowchart TD

    A[User Browser] --> B[Next.js Frontend - App Router]
    B --> C[Upload / Webcam Capture Component]
    C --> D[Results & Insights Layer - Recharts]
    C --> E[API Client - lib/api.ts]

    E --> F[FastAPI Backend - skinscan_api]
    F --> G[Image Service - Validation & Preprocessing]
    G --> H[Model Service]
    H --> I[CNN Backbone - ResNet18 / EfficientNet-B0]
    I --> J[TTA Inference Engine]
    J --> K[Confidence + Class Probabilities]

    K --> F
    F --> E
    E --> D
    D --> L[Disease Reference Library]
```

---

## 🔄 End-to-End Processing Flow

```mermaid
flowchart TD

    A[User Uploads Skin Image] --> B[Frontend Validates File Type/Size]
    B --> C[POST to /api/predict]
    C --> D[Backend Validates Image Integrity]
    D --> E[Preprocess - Resize 224x224 + ImageNet Normalize]
    E --> F[Run Test-Time Augmented Inference]
    F --> G[Compute Class Probabilities]
    G --> H[Map to Disease Info + Severity]
    H --> I[Return JSON Response]
    I --> J[Frontend Renders Confidence Chart]
    J --> K[Show Severity + Recommendation]
```

---

## ☁️ Cloud Execution Flow

```mermaid
flowchart LR

    User --> Frontend
    Frontend --> API
    API --> Backend
    Backend --> Model
    Model --> Backend
    Backend --> API
    API --> Frontend
    Frontend --> User

    subgraph Frontend_Layer
        Frontend[Next.js 14 - Vercel]
    end

    subgraph Backend_Layer
        Backend[FastAPI - Render Web Service]
        API[uvicorn ASGI Server]
        Model[PyTorch CNN - CPU Inference]
    end
```

---

## 🔁 Prediction Request Lifecycle

```mermaid
sequenceDiagram
    participant U as User
    participant F as Next.js Frontend
    participant A as FastAPI Backend
    participant M as Model Service

    U->>F: Upload Image / Capture Webcam Frame
    F->>A: POST /api/predict (multipart file)
    A->>A: Validate content-type + image integrity
    A->>M: Preprocess + Run Inference
    M-->>A: Predicted Class + Probabilities
    A-->>F: JSON (class, confidence, disease_info)
    F-->>U: Render Result Card + Confidence Chart
```

---

## 🚀 Development Status

SkinScan-AI is actively being deployed and hardened for production hosting.

Ongoing work includes:

- Deploying backend to Render and frontend to Vercel
- Resolving Docker cold-start latency on the free tier
- Improving checkpoint/architecture auto-detection robustness
- Expanding the disease reference library
- Mobile camera-capture refinements

---

## ✨ Key Features

- 🧠 AI-powered skin disease classification across 9 dermatological classes
- 📸 Image upload **and** live webcam capture support
- 🔁 Test-time augmentation (TTA) for more stable confidence scores
- 📊 Per-class probability breakdown with animated Recharts visuals
- 🩹 Severity-tiered guidance (low / moderate / high / unknown) per condition
- 🧾 PDF report export via `jsPDF`
- 🩺 Graceful **demo mode** — API stays usable even without a checkpoint file
- ⚡ Health endpoint reporting model status, architecture, and mode

---

## 🤖 Machine Learning Integration

SkinScan-AI uses a **CNN backbone (ResNet18 in training, EfficientNet-B0 in demo mode)** for 9-class skin lesion classification, with the production architecture auto-detected from the loaded checkpoint's state dict.

### ML Workflow

1. User uploads or captures a skin image
2. Backend validates format, dimensions, and integrity
3. Image is resized to 224×224 and normalized using ImageNet statistics
4. Model runs test-time-augmented inference
5. Softmax probabilities are computed per class
6. Confidence level is bucketed (`healthy` / `low` / `medium` / `high`)
7. Matching disease metadata (description, severity, recommendation) is attached

### Disease Classes Covered

| Class | Severity Level |
|---|---|
| Actinic keratosis | 🟠 Moderate |
| Atopic Dermatitis | 🟡 Low |
| Benign keratosis | 🟡 Low |
| Dermatofibroma | 🟡 Low |
| Melanocytic nevus | 🟡 Low |
| Melanoma | 🔴 High |
| Squamous cell carcinoma | 🔴 High |
| Tinea Ringworm Candidiasis | 🟡 Low |
| Vascular lesion | 🟡 Low |

---

## 📂 Project Structure
