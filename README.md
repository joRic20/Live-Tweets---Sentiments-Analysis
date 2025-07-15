# 🐦 Twitter Sentiment Analysis Dashboard

A real-time Twitter sentiment analysis platform that tracks brand mentions, analyzes sentiment using state-of-the-art NLP, and provides interactive analytics dashboards. Built with FastAPI, Streamlit, MongoDB, PostgreSQL, and Docker.

**⚡ Powered by Twitter Developer API v2**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![Twitter API](https://img.shields.io/badge/Twitter-API%20v2-1DA1F2)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Development](#-development)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## Features

- **Real-time Tweet Collection**: Stream tweets using Twitter Developer API v2
- **Advanced Sentiment Analysis**: Utilizes RoBERTa transformer model fine-tuned for tweets
- **Interactive Dashboard**: Real-time analytics with Streamlit
- **Text Cleaning Pipeline**: Removes URLs, mentions, and noise for accurate analysis
- **Multi-Campaign Tracking**: Monitor multiple brands simultaneously
- **Comprehensive Analytics**:
  - Sentiment distribution (positive/neutral/negative)
  - Volume trends over time
  - Top hashtags and users
  - Word clouds
  - Campaign comparison metrics
- **RESTful API**: Access all data programmatically
- **Automated ETL**: Continuous processing with configurable intervals
- **Data Export**: Download analytics and tweet data as CSV

## Architecture

```
┌─────────────────┐
│  Twitter API v2 │
└────────┬────────┘
         │ Stream
┌────────▼────────┐
│  Ingest Service │
└────────┬────────┘
         │ Raw Tweets
┌────────▼────────┐
│     MongoDB     │
└────────┬────────┘
         │ Process
┌────────▼────────┐
│   ETL Service   │
└────────┬────────┘
         │ Clean & Analyze
┌────────▼────────┐
│   PostgreSQL    │
└────────┬────────┘
         │ Serve
┌────────▼────────┐
│ FastAPI Backend │
└────────┬────────┘
         │ Display
┌────────▼────────┐
│    Dashboard    │
└─────────────────┘
```

### Components

1. **Ingest Service** (`ingest.py`): Streams tweets from Twitter Developer API into MongoDB
2. **MongoDB**: Stores raw tweet data with automatic TTL (30 days)
3. **ETL Service** (`etl.py`): Processes tweets with sentiment analysis and cleaning
4. **PostgreSQL**: Stores processed tweets and aggregated analytics
5. **FastAPI Backend** (`fast_api.py`): RESTful API for data access
6. **Streamlit Dashboard** (`dashboard.py`): Interactive web interface

## Prerequisites

- **Twitter Developer Account** with Elevated Access
  - Sign up at https://developer.twitter.com
  - Create a Project and App
  - Generate Bearer Token for API v2
- Docker & Docker Compose
- 4GB+ RAM recommended
- 10GB+ disk space for databases

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <repo-address>
   cd twitter-sentiment-dashboard
   ```

2. **Create .env file**
   ```bash
   cp .env.example .env
   ```

3. **Configure environment variables** (see Configuration)
   - Most importantly: Add your Twitter Developer API Bearer Token

4. **Start the platform**
   ```bash
   docker-compose up -d
   ```

5. **Access the dashboard**
   - Dashboard: http://localhost:8501
   - API Docs: http://localhost:8000/docs

6. **Start tracking a campaign**
   - Open the dashboard
   - Enter a brand/campaign name in the sidebar
   - Click "Start Tracking"

## Configuration

### Twitter Developer API Setup

1. **Create Twitter Developer Account:**
   - Visit https://developer.twitter.com
   - Apply for Elevated Access (required for streaming)
   - Create a new Project and App

2. **Generate Bearer Token:**
   - In your app settings, generate a Bearer Token
   - Copy this token to your .env file

3. **API Limits (with Elevated Access):**
   - 2 million tweets per month
   - 50 requests per 15 minutes (for app-level auth)

### Environment Variables (.env)

```env
# Twitter Developer API Configuration (REQUIRED)
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here

# API Security
API_KEY=your_secure_api_key_here

# PostgreSQL Configuration
POSTGRES_DB=database_name
POSTGRES_USER=username
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_URI=postgres_connection_url

# MongoDB Configuration
MONGO_URI=mongo_connection_url
MONGO_DB=database_name
MONGO_COLL=mongo_collection

# API URLs (for Docker networking)
API_BASE=http://backend:8000

# Ingestion Settings
MAX_TWEETS=100  # Maximum tweets per ingestion session
```

### Docker Services Configuration

The `docker-compose.yml` includes:
- **mongo**: MongoDB for raw tweet storage
- **postgres**: PostgreSQL for analytics
- **backend**: FastAPI REST API
- **dashboard**: Streamlit web interface
- **etl**: Automated processing service (runs every 5 minutes)

## Usage

### Starting New Campaign Tracking

1. Open dashboard at http://localhost:8501
2. In the sidebar, enter a brand/campaign name
3. Click "Start Tracking"
4. The system will use Twitter Developer API to stream matching tweets
5. View real-time analytics as tweets are collected

### Viewing Analytics
- **Campaign Management**: Access campaigns via `/campaigns/` endpoint
- **Date Range Selection**: Retrieve available date ranges using `/date_range/` and `/date_range_hourly/` endpoints
- **Analytics Data**: 
  - General analytics via `/analytics/` endpoint
  - Hourly analytics breakdowns via `/analytics_hourly/`
  - Summary statistics through `/analytics_summary/`
- **Tweet Analysis**:
  - View top performing tweets with `/top_tweets/`
  - Monitor latest tweets using `/latest_tweets/`
- **Data Ingestion**: 
  - Start new data ingestion with `/start_ingest/`
  - Check ingestion status via `/ingestion_status/{term}`
- **System Health**: Monitor API status through `/health` endpoint

### Display Options

- show raw tables
- Auto-refresh every 60 seconds

## API Documentation

### Authentication

All API endpoints (except `/health`) require an API key in the header:
```
Authorisation: your_api_key_here
```

### 🔌 API Reference

These endpoints are exposed by the FastAPI backend for retrieving sentiment insights, campaign data, and ingestion control:

| Method | Endpoint                   | Description                        | Auth |
|--------|----------------------------|------------------------------------|------|
| GET    | `/ingestion_status/{term}` | Get status of a tracked term       | Yes  |
| GET    | `/health`                  | Health check                       | No   |
| POST   | `/start_ingest/`           | Start ingesting tweets by term     | Yes  |
| GET    | `/campaigns/`              | Get list of tracked campaigns      | Yes  |
| GET    | `/date_range/`             | Get min/max dates for analytics    | Yes  |
| GET    | `/analytics/`              | Get aggregated campaign analytics  | Yes  |
| GET    | `/analytics_hourly/`       | Get hourly analytics data          | Yes  |
| GET    | `/date_range_hourly/`      | Get hourly date range              | Yes  |
| GET    | `/analytics_summary/`      | Get campaign sentiment summary     | Yes  |
| GET    | `/top_tweets/`             | Get top-ranked tweets              | Yes  |
| GET    | `/latest_tweets/`          | Get the latest ingested tweets     | Yes  |


### Example API Usage

```python
import requests

headers = {"Authorisation": "your_api_key"}
base_url = "(https://twitter-fastapi-test.quandev.xyz/docs)"

# Start tracking a brand
response = requests.post(
    f"{base_url}/start_ingest/",
    json={"term": "Nike"},
    headers=headers
)

# Get analytics
params = {
    "campaigns": "Grok,Tesla",
    "start": "2024-01-01",
    "end": "2024-01-31"
}
response = requests.get(
    f"{base_url}/analytics/",
    params=params,
    headers=headers
)
```

## Project Structure

```
LIVE-TWEETS---SENTIMENTS-ANALYSIS/
├── backend/
│   ├── api/
│   │   ├── Dockerfile
│   │   ├── fast_api.py           # REST API
│   │   ├── ingest.py             # Twitter ingestion logic
│   │   └── requirements.txt
│   └── etl/
│       ├── Dockerfile
│       ├── etl.py                # Data cleaning & sentiment tagging
│       ├── run_etl.sh            # ETL runner script
│       └── requirements.txt
├── dashboard/
│   ├── Dockerfile
│   ├── dashboard.py              # Streamlit dashboard UI
│   └── requirements.txt
├── docker-compose.yml            # Orchestrates all services
├── .env                          # Environment variables
├── .gitignore
├── README.md
├── requirements.in               # Consolidated requirements
└── requirements.txt

```

### Key Files

- **ingest.py**: Connects to Twitter Developer API v2 streaming endpoint
- **etl.py**: Processes tweets using Hugging Face transformers
- **fast_api.py**: Provides RESTful endpoints for data access
- **dashboard.py**: Creates interactive Streamlit visualizations

## Development

### Local Development

```bash
# Backend development
cd backend
pip install -r requirements.txt
uvicorn fast_api:app --reload

# Dashboard development
cd dashboard
pip install -r requirements.txt
streamlit run dashboard.py
```

### Database Access

```bash
# MongoDB shell
docker exec -it twitter-mongo mongosh

# PostgreSQL shell
docker exec -it twitter-postgres psql -U postgres -d twitter_analytics
```

### Monitoring Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f etl
```

### Twitter API Stream Monitoring

```bash
# Watch ingestion logs
docker-compose logs -f backend | grep "ingest"

# Check tweet collection rate
docker exec -it twitter-mongo mongosh --eval "db.tweets.count()"
```

## Troubleshooting

### Common Issues

#### "Twitter API Authentication Failed"
- Verify Bearer Token in .env file
- Ensure you have Twitter Developer Elevated Access
- Check token hasn't expired or been revoked

#### "No tweets appearing"
- Verify Twitter Bearer Token is valid
- Check if search term is too specific
- Monitor rate limits in Twitter Developer Portal
- Check ingestion logs: `docker-compose logs backend`

#### "Sentiment analysis not working"
- ETL runs every 5 minutes - wait for processing
- Check ETL logs: `docker-compose logs etl`
- Verify transformer model downloaded successfully

#### "Rate limit exceeded"
- Twitter API has rate limits (see your Developer Dashboard)
- Reduce MAX_TWEETS in .env
- Implement longer delays between requests

#### "Database connection failed"
- Ensure databases are healthy: `docker-compose ps`
- Check connection strings in .env
- Verify no port conflicts

### Reset Everything

```bash
# Stop and remove all data
docker-compose down -v

# Rebuild and start fresh
docker-compose up --build -d
```

### Backup Data

```bash
# Backup MongoDB tweets
docker exec twitter-mongo mongodump --db twitter_stream --out /backup

# Backup PostgreSQL analytics
docker exec twitter-postgres pg_dump -U postgres twitter_analytics > analytics_backup.sql
```

## Security Considerations

- Store Twitter Bearer Token securely
- Use strong passwords for databases
- Don't commit .env file to version control
- Regularly rotate API keys
- Monitor API usage in Twitter Developer Portal

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- **Python**: Follow PEP 8
- Use type hints where applicable
- Add docstrings to functions
- Keep functions focused and small

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Twitter Developer Platform for API access
- Hugging Face for the RoBERTa sentiment model
- Streamlit for the amazing dashboard framework
- FastAPI for the high-performance API

## Disclaimer

This project uses the Twitter Developer API and is subject to Twitter's Developer Agreement and Policy. Please ensure compliance with:

- Twitter's rate limits
- Twitter's data usage policies
- Twitter's display requirements
- Local data protection regulations

---

For support, please open an issue in the GitHub repository.
