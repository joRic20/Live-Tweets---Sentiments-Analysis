#!/bin/bash
# ETL Runner for CapRover

echo "=== ETL Started at $(date) ==="

# Environment variables are already set by CapRover
cd /app

# Run ETL
python etl.py

echo "=== ETL Completed at $(date) ==="