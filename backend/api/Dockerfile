FROM python:3.11-slim
WORKDIR /app

# Update SSL certificates and install required packages
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Upgrade pymongo to latest version
RUN pip install --upgrade pymongo

COPY fast_api.py .
COPY ingest.py .
EXPOSE 8000
CMD ["uvicorn", "fast_api:app", "--host", "0.0.0.0", "--port", "8000"]