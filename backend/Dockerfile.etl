FROM python:3.11-slim
WORKDIR /app

# Install cron and required packages
RUN apt-get update && apt-get install -y cron && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements-etl.txt .
RUN pip install --no-cache-dir -r requirements-etl.txt

# Copy ETL script
COPY etl.py .
COPY run_etl.sh .
RUN chmod +x run_etl.sh

# Setup cron job (correct syntax: minute hour day month weekday command)
RUN echo "0 * * * * /app/run_etl.sh >> /var/log/etl.log 2>&1" | crontab -

# Create log file
RUN touch /var/log/etl.log

# Start cron in foreground
CMD ["cron", "-f"]