# SkinScan-AI

SkinScan-AI is an AI-assisted skin disease classifier rebuilt as a full-stack medical-tech demo.

The repo now contains:

- `backend/` - FastAPI service for preprocessing, inference, and disease metadata
- `frontend/` - Next.js 14 App Router UI with Tailwind, Framer Motion, and Recharts
- `attached_assets/` - sample images and the default checkpoint path used by the backend

## Architecture

- The frontend uploads an image or webcam capture.
- The backend validates the file, applies the existing preprocessing pipeline, and runs TTA inference.
- The API returns the predicted class, confidence, per-class probabilities, and condition guidance.
- The UI presents the result with animated confidence visuals and a disease reference library.

## API Routes

- `POST /api/predict` - upload a skin image and receive prediction output
- `GET /api/classes` - fetch the 9 disease classes and their reference info
- `GET /api/health` - check service and model status

## Local Development

Backend:

```bash
cd backend
pip install -r requirements.txt
uvicorn skinscan_api.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

The frontend reads `NEXT_PUBLIC_API_URL` from `frontend/.env.local` and points to `http://localhost:8000` by default.

## Docker Compose

Run both services together:

```bash
docker compose up --build
```

That exposes the frontend at `http://localhost:3000` and the backend at `http://localhost:8000`.

## Model Notes

- The backend defaults to `attached_assets/skin_disease_model_1755756972916.pth`.
- If that checkpoint is missing, the API can remain up in demo mode so the UI stays usable.
- Set `SKINSCAN_ALLOW_DEMO_MODEL=false` if you want missing-model failures to be explicit.

## Author

Ravi Shankar

B.Tech CSE AIML

GitHub: https://github.com/shanky-ux

Portfolio: https://ravish4nkar.vercel.app

## Disclaimer

This project is for educational use only and does not provide a medical diagnosis.
