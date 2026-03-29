# Single Dockerfile for all environments.
# HF Spaces: PORT=7860 (default). Local dev: docker-compose overrides PORT=8000.
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ .
ENV PYTHONPATH=/app
ENV PORT=7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]