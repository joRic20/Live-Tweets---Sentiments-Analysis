FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add this to install curl
RUN apt-get update && apt-get install -y curl

COPY fast_api.py .
COPY etl.py .
COPY ingest.py .

EXPOSE 8000

CMD ["uvicorn", "fast_api:app", "--host", "0.0.0.0", "--port", "8000"]
