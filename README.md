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
SkinScan-AI/
│
├── backend/
│   ├── skinscan_api/
│   │   ├── main.py             # FastAPI app entrypoint
│   │   ├── routes.py           # /api/health, /api/classes, /api/predict
│   │   ├── services.py         # Preprocessing, TTA inference, disease map
│   │   ├── schemas.py          # Pydantic response models
│   │   └── config.py           # Settings (model path, CORS origin, demo flag)
│   ├── attached_assets/        # Model checkpoint (.pth) + metadata (.json)
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx             # Upload + results landing page
│   │   ├── about/page.tsx
│   │   └── conditions/page.tsx  # Disease reference library
│   ├── components/
│   │   ├── upload.tsx
│   │   ├── results.tsx
│   │   ├── library.tsx
│   │   ├── model.tsx
│   │   └── navbar.tsx / footer.tsx
│   ├── lib/
│   │   ├── api.ts               # Axios/fetch client
│   │   ├── data.ts
│   │   └── types.ts
│   ├── package.json
│   └── Dockerfile
│
├── attached_assets/             # Root-level sample checkpoint (Streamlit legacy)
├── train_skin_disease_model.py  # CNN training pipeline
├── fetch_and_train.py
├── image_processor.py           # Legacy Streamlit preprocessing
├── model_utils.py               # Legacy Streamlit model loading
├── app.py                       # Legacy Streamlit entrypoint
├── docker-compose.yml
├── requirements.txt
└── README.md

---

## 🔐 Environment Variables

Frontend (`frontend/.env.local`):
NEXT_PUBLIC_API_URL=http://localhost:8000

Backend (set in Render dashboard or shell):
SKINSCAN_MODEL_PATH=attached_assets/skin_disease_model_1755756972916.pth
SKINSCAN_FRONTEND_ORIGIN=http://localhost:3000
SKINSCAN_ALLOW_DEMO_MODEL=true

---

## ⚙️ Installation & Local Setup

```bash
# Clone the repository
git clone https://github.com/shanky-ux/SkinScan-AI.git
cd SkinScan-AI

# Setup Backend
cd backend
pip install -r requirements.txt
uvicorn skinscan_api.main:app --reload --port 8000
# API runs at http://localhost:8000

# Setup Frontend (new terminal)
cd frontend
npm install
npm run dev
# Frontend runs at http://localhost:3000
```

### Or run both with Docker Compose

```bash
docker compose up --build
```

That exposes the frontend at `http://localhost:3000` and the backend at `http://localhost:8000`.

---

## 🚀 Deployment

| Service | Platform | Notes |
|---|---|---|
| 🔌 Backend (FastAPI) | Render Web Service | `Dockerfile` in `backend/`, exposes port 8000 |
| 🖥️ Frontend (Next.js) | Vercel | Set `NEXT_PUBLIC_API_URL` to the Render backend URL |
| 🗄️ Model Checkpoint | Bundled in image (`attached_assets/`) | Falls back to demo mode if missing |

**Render Settings:**
- Root Directory: `backend`
- Dockerfile Path: `backend/Dockerfile`
- Start Command: handled by `Dockerfile` (`uvicorn skinscan_api.main:app --host 0.0.0.0 --port 8000`)

**Vercel Settings:**
- Root Directory: `frontend`
- Build Command: `next build`
- Environment Variable: `NEXT_PUBLIC_API_URL=<render-backend-url>`

---

## 🔌 Key API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Model status, mode (checkpoint/demo), architecture, class count |
| `/api/classes` | GET | List all 9 disease classes with reference metadata |
| `/api/predict` | POST | Upload a skin image, receive prediction + confidence + guidance |

---

## 🎯 Why This Project Stands Out

- Real-world healthcare-adjacent problem with a working end-to-end AI pipeline
- Migrated from a Streamlit prototype to a decoupled Next.js + FastAPI production architecture
- Graceful demo-mode fallback keeps the app usable even without a trained checkpoint
- Clean separation of concerns: services, schemas, routes on the backend; components, lib, app router on the frontend
- Fully containerized with Docker Compose for one-command local spin-up

---

## 👨‍💻 Author

**Ravi Shankar (Shanky)**
B.Tech Computer Science (AIML)
Full Stack Developer | AI/ML Enthusiast

GitHub: https://github.com/shanky-ux
Portfolio: https://ravish4nkar.vercel.app

---

## 📜 License

This project is licensed under the MIT License — built for educational and demonstration purposes.

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:2c5364,50:203a43,100:0f2027&height=120&section=footer&animation=fadeIn"/>
</p>

<p align="center">
  <a href="https://ravish4nkar.vercel.app">🌐 Portfolio</a> &nbsp;|&nbsp;
  <a href="https://github.com/shanky-ux/SkinScan-AI">⭐ Star this Repo</a>
</p>

<p align="center"><i>"AI can't replace a dermatologist — but it can help someone decide whether to book that appointment sooner."</i></p>
