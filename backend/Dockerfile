FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/backend/requirements.txt

COPY backend /app/backend
COPY attached_assets /app/attached_assets

ENV SKINSCAN_MODEL_PATH=attached_assets/skin_disease_model_1755756972916.pth
ENV SKINSCAN_FRONTEND_ORIGIN=http://localhost:3000

EXPOSE 8000

CMD ["uvicorn", "skinscan_api.main:app", "--host", "0.0.0.0", "--port", "8000"]