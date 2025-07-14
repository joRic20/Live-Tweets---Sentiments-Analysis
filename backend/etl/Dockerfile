FROM python:3.11-slim
WORKDIR /app

# Install cron and required packages
RUN apt-get update && apt-get install -y cron && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements-etl.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ETL script and shell script
COPY etl.py .
COPY run_etl.sh .
RUN chmod +x run_etl.sh

# Create log file (before cron runs)
RUN touch /var/log/etl.log

# Add cron job
RUN echo "0 * * * * /app/run_etl.sh >> /var/log/etl.log 2>&1" > /etc/cron.d/etl-cron

# Give execution rights on the cron job and apply it
RUN chmod 0644 /etc/cron.d/etl-cron && crontab /etc/cron.d/etl-cron

# Start cron in foreground
CMD ["cron", "-f"]
