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
import sys                                                     # For system exit
import logging                                                 # For logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load secrets and config from .env ---
load_dotenv()  # Load variables from .env into the environment

# Helper function to clean environment variables (removes inline comments)
def clean_env_var(value):
    """Remove inline comments and extra quotes from environment variables"""
    if value:
        # Remove comments if present
        if '#' in value:
            value = value.split('#')[0]
        # Strip whitespace and quotes
        value = value.strip().strip('"\'')
    return value

# Get and clean environment variables
API_KEY = clean_env_var(os.environ.get("API_KEY", "changeme"))
POSTGRES_URI = clean_env_var(os.environ.get("POSTGRES_URI"))
MONGO_URI = clean_env_var(os.environ.get("MONGO_URI"))
MONGO_DB = clean_env_var(os.environ.get("MONGO_DB"))
MONGO_COLL = clean_env_var(os.environ.get("MONGO_COLL"))

# Log configuration (without exposing sensitive data)
logger.info(f"API Key: {'Set' if API_KEY and API_KEY != 'changeme' else 'Not Set'}")
logger.info(f"PostgreSQL URI: {'Set' if POSTGRES_URI else 'Not Set'}")
logger.info(f"MongoDB URI: {'Set' if MONGO_URI else 'Not Set'}")
logger.info(f"MongoDB Database: {MONGO_DB}")
logger.info(f"MongoDB Collection: {MONGO_COLL}")

# Validate required environment variables
if not all([POSTGRES_URI, MONGO_URI, MONGO_DB, MONGO_COLL]):
    logger.error("Missing required environment variables!")
    logger.error("Please ensure POSTGRES_URI, MONGO_URI, MONGO_DB, and MONGO_COLL are set in .env")
    sys.exit(1)

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

# --- Database connections ---
try:
    engine = create_engine(POSTGRES_URI)
    # Test PostgreSQL connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("PostgreSQL connection successful")
except Exception as e:
    logger.error(f"Failed to connect to PostgreSQL: {e}")
    logger.error(f"PostgreSQL URI format should be: postgresql://username:password@host:port/database")
    sys.exit(1)

try:
    mongo_client = MongoClient(MONGO_URI)
    mongo_client.server_info()
    mongo_db = mongo_client[MONGO_DB]
    tweets_collection = mongo_db[MONGO_COLL]
    logger.info(f"MongoDB connection successful - Database: {MONGO_DB}, Collection: {MONGO_COLL}")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    logger.error(f"MongoDB URI format should be: mongodb://username:password@host:port/database")
    sys.exit(1)

# --- Health check endpoint ---
@app.get("/health")
def health_check():
    """Check if the API and databases are accessible"""
    try:
        # Check PostgreSQL
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        # Check MongoDB
        mongo_client.server_info()
        
        return {
            "status": "healthy",
            "postgres": "connected",
            "mongodb": "connected"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# --- Endpoint: Trigger tweet ingestion for a term ---
@app.post("/start_ingest/")
def start_ingest(
    term: str,                                            # The search term, from request body
    background_tasks: BackgroundTasks,                    # Background task handler
    api_key: APIKey = Depends(get_api_key)                # Require valid API key
):
    """
    Starts tweet ingestion for a new search term (runs ingest.py as background task).
    """
    logger.info(f"Starting ingestion for term: {term}")
    
    def run_ingest():
        try:
            # Call ingest.py as a subprocess
            result = subprocess.run(["python", "ingest.py", term], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Ingestion failed for {term}: {result.stderr}")
            else:
                logger.info(f"Ingestion completed for {term}")
        except Exception as e:
            logger.error(f"Failed to run ingestion: {e}")
    
    # Register the ingestion function to run in the background
    background_tasks.add_task(run_ingest)
    return {"status": "ingestion started", "term": term}

# --- Endpoint: List all campaigns tracked ---
@app.get("/campaigns/")
def get_campaigns(api_key: APIKey = Depends(get_api_key)):
    """
    Returns a list of all unique campaigns (for dashboard filter).
    """
    try:
        q = "SELECT DISTINCT campaign FROM tweet_analytics ORDER BY campaign;"
        campaigns = pd.read_sql(q, engine)['campaign'].tolist()
        return campaigns
    except Exception as e:
        logger.error(f"Error fetching campaigns: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch campaigns")

# --- Endpoint: Get min and max date in analytics ---
@app.get("/date_range/")
def get_min_max_dates(api_key: APIKey = Depends(get_api_key)):
    """
    Returns the earliest and latest dates in the analytics table.
    """
    try:
        q = "SELECT MIN(date) as min_date, MAX(date) as max_date FROM tweet_analytics;"
        result = pd.read_sql(q, engine)
        
        # Handle empty table
        if result.iloc[0]['min_date'] is None:
            import datetime
            today = datetime.date.today()
            return {
                "min_date": str(today - datetime.timedelta(days=7)),
                "max_date": str(today)
            }
        
        d = result.iloc[0].to_dict()
        d = {k: str(v) for k, v in d.items()}  # Convert to string for JSON serialization
        return d
    except Exception as e:
        logger.error(f"Error fetching date range: {e}")
        # Return default date range on error
        import datetime
        today = datetime.date.today()
        return {
            "min_date": str(today - datetime.timedelta(days=7)),
            "max_date": str(today)
        }

# --- Endpoint: Aggregated analytics for selected campaigns and date range ---
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
    try:
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
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch analytics")

# --- Endpoint: Latest tweets with sentiment for selected campaigns/date range ---
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
    try:
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
    except Exception as e:
        logger.error(f"Error fetching tweets: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch tweets")

# --- Endpoint: Top hashtags ---
@app.get("/top_hashtags/")
def top_hashtags(
    campaigns: str = Query(..., description="Comma-separated campaign names"),
    start: str = Query(..., description="Start date YYYY-MM-DD"),
    end: str = Query(..., description="End date YYYY-MM-DD"),
    limit: int = Query(10, description="Number of top hashtags"),
    api_key: APIKey = Depends(get_api_key)
):
    """
    Returns top hashtags for selected campaigns and date range.
    """
    try:
        campaign_list = [c.strip() for c in campaigns.split(",")]
        # MongoDB aggregation pipeline for hashtag counts
        pipeline = [
            {
                "$match": {
                    "matched_term": {"$in": campaign_list},
                    "created_at": {"$gte": start, "$lte": end},
                    "hashtags": {"$exists": True, "$ne": []}
                }
            },
            {"$unwind": "$hashtags"},
            {"$group": {"_id": "$hashtags", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit}
        ]
        results = list(tweets_collection.aggregate(pipeline))
        return {item["_id"]: item["count"] for item in results}
    except Exception as e:
        logger.error(f"Error fetching hashtags: {e}")
        return {}

# --- Endpoint: Top users ---
@app.get("/top_users/")
def top_users(
    campaigns: str = Query(..., description="Comma-separated campaign names"),
    start: str = Query(..., description="Start date YYYY-MM-DD"),
    end: str = Query(..., description="End date YYYY-MM-DD"),
    limit: int = Query(10, description="Number of top users"),
    api_key: APIKey = Depends(get_api_key)
):
    """
    Returns top users for selected campaigns and date range.
    """
    try:
        campaign_list = [c.strip() for c in campaigns.split(",")]
        # MongoDB aggregation pipeline for user tweet counts
        pipeline = [
            {
                "$match": {
                    "matched_term": {"$in": campaign_list},
                    "created_at": {"$gte": start, "$lte": end}
                }
            },
            {"$group": {"_id": "$user_id", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit}
        ]
        results = list(tweets_collection.aggregate(pipeline))
        return {str(item["_id"]): item["count"] for item in results}
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        return {}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)