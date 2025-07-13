FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/fast_api.py .
COPY backend/etl.py .
COPY backend/ingest.py .

EXPOSE 8000

CMD ["uvicorn", "fast_api:app", "--host", "0.0.0.0", "--port", "8000"]