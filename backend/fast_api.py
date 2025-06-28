# fast_api.py
# FastAPI app for analytics API and on-demand tweet ingestion

# Import FastAPI and dependencies for request handling, security, and background tasks
from fastapi import FastAPI, Query, Depends, Security, HTTPException, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader, APIKey      # For API key auth
from starlette.status import HTTP_403_FORBIDDEN                # For error codes
from fastapi.openapi.utils import get_openapi                  # For customizing docs
from dotenv import load_dotenv                                 # For .env support
import pandas as pd                                            # For SQL/analytics
from sqlalchemy import create_engine, text                     # For Postgres connection
from pymongo import MongoClient                                # For MongoDB connection
import subprocess                                              # For running scripts
import os                                                      # For environment variables

# --- Load secrets and config from .env ---
load_dotenv()  # Load variables from .env into the environment

# Get API key for authentication, or default to "changeme" (should be set in .env)
API_KEY = os.environ.get("API_KEY", "changeme")
API_KEY_NAME = "access_token"                                  # Name of the header field for API key
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)  # FastAPI security scheme

# --- FastAPI application instance, with metadata for docs ---
app = FastAPI(
    title="Tweet Analytics API",
    description="API for analytics and live tweets dashboard. Requires API key.",
    version="1.0.0"
)

# --- Custom OpenAPI schema: add API key security for docs ---
def custom_openapi():
    # Use cached schema if already built
    if app.openapi_schema:
        return app.openapi_schema
    # Generate the default OpenAPI schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    # Add the APIKeyHeader scheme for Swagger "Authorize" button
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": API_KEY_NAME
        }
    }
    # Attach security scheme to all endpoints
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method.setdefault("security", []).append({"APIKeyHeader": []})
    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Assign our custom OpenAPI function to the app
app.openapi = custom_openapi

# --- Security dependency for all endpoints ---
def get_api_key(api_key_header: str = Security(api_key_header)):
    """
    Checks if the API key in the request header matches the server key.
    """
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
        )

# --- Database connections (loaded from .env) ---
engine = create_engine(os.environ.get("POSTGRES_URI"))    # For analytics (PostgreSQL)
MONGO_URI = os.environ.get("MONGO_URI")                   # MongoDB URI
MONGO_DB = os.environ.get("MONGO_DB")                     # MongoDB database name
MONGO_COLL = os.environ.get("MONGO_COLL")                 # MongoDB collection name
mongo_client = MongoClient(MONGO_URI)                     # Connect to MongoDB
mongo_db = mongo_client[MONGO_DB]                         # Select database
tweets_collection = mongo_db[MONGO_COLL]                  # Select collection

# --- Endpoint: Trigger tweet ingestion for a term (user-provided, no default) ---
@app.post("/start_ingest/")
def start_ingest(
    term: str,                                            # The search term, from request body
    background_tasks: BackgroundTasks,                    # Background task handler (so API doesn't block)
    api_key: APIKey = Depends(get_api_key)                # Require valid API key
):
    """
    Starts tweet ingestion for a new search term (runs ingest.py as background task).
    """
    def run_ingest():
        # Call ingest.py as a subprocess, passing the term as an argument
        # Assumes ingest.py is in the same directory as this file
        subprocess.run(["python", "ingest.py", term])
    # Register the ingestion function to run in the background
    background_tasks.add_task(run_ingest)
    # Return immediately; ingestion will run in the background
    return {"status": "ingestion started", "term": term}

# --- Endpoint: List all campaigns tracked (from Postgres) ---
@app.get("/campaigns/")
def get_campaigns(api_key: APIKey = Depends(get_api_key)):
    """
    Returns a list of all unique campaigns (for dashboard filter).
    """
    q = "SELECT DISTINCT campaign FROM tweet_analytics ORDER BY campaign;"
    campaigns = pd.read_sql(q, engine)['campaign'].tolist()
    return campaigns

# --- Endpoint: Get min and max date in analytics (from Postgres) ---
@app.get("/date_range/")
def get_min_max_dates(api_key: APIKey = Depends(get_api_key)):
    """
    Returns the earliest and latest dates in the analytics table.
    """
    q = "SELECT MIN(date) as min_date, MAX(date) as max_date FROM tweet_analytics;"
    result = pd.read_sql(q, engine)
    d = result.iloc[0].to_dict()
    d = {k: str(v) for k, v in d.items()}  # Convert to string for JSON serialization
    return d

# --- Endpoint: Aggregated analytics for selected campaigns and date range (from Postgres) ---
@app.get("/analytics/")
def analytics(
    campaigns: str = Query(..., description="Comma-separated campaign names"),
    start: str = Query(..., description="Start date YYYY-MM-DD"),
    end: str = Query(..., description="End date YYYY-MM-DD"),
    api_key: APIKey = Depends(get_api_key)
):
    """
    Returns aggregated analytics for selected campaigns and date range.
    """
    # Parse comma-separated campaigns into a list
    campaign_list = [c.strip() for c in campaigns.split(",")]
    # Query analytics table for stats per campaign and day
    query = text("""
        SELECT campaign, date, avg_sentiment, tweet_count, pos_count, neu_count, neg_count
        FROM tweet_analytics
        WHERE campaign IN :campaigns
          AND date BETWEEN :start AND :end
        ORDER BY campaign, date;
    """)
    df = pd.read_sql(query, engine, params={"campaigns": tuple(campaign_list), "start": start, "end": end})
    return df.to_dict(orient="records")

# --- Endpoint: Latest tweets with sentiment for selected campaigns/date range (from MongoDB) ---
@app.get("/latest_tweets/")
def latest_tweets(
    campaigns: str = Query(..., description="Comma-separated campaign names"),
    start: str = Query(..., description="Start date YYYY-MM-DD"),
    end: str = Query(..., description="End date YYYY-MM-DD"),
    limit: int = Query(100, description="Max tweets"),
    api_key: APIKey = Depends(get_api_key)
):
    """
    Returns latest tweets (with sentiment) for colored table/word cloud.
    """
    # Parse campaigns for MongoDB filter
    campaign_list = [c.strip() for c in campaigns.split(",")]
    query_mongo = {
        "matched_term": {"$in": campaign_list},
        "created_at": {"$gte": start, "$lte": end}
    }
    fields = {"text": 1, "sentiment_label": 1, "created_at": 1}
    # Find latest tweets matching the campaigns and time window
    tweets = list(tweets_collection.find(query_mongo, fields).sort("created_at", -1).limit(limit))
    # Convert date for JSON and remove MongoDB's _id field
    for t in tweets:
        if "created_at" in t:
            t["created_at"] = str(t["created_at"])
        if "_id" in t:
            t.pop("_id")
    return tweets
