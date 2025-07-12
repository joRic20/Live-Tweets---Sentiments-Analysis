# fast_api.py
# FastAPI app for analytics API focused on sentiment measurements
# Updated with hourly analytics support (product sentiment removed)

from fastapi import FastAPI, Query, Depends, Security, HTTPException, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader, APIKey
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.openapi.utils import get_openapi
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text
from pymongo import MongoClient
import subprocess
import os
import sys
import logging
from pydantic import BaseModel
from typing import Optional
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def clean_env_var(value):
    """Remove inline comments and extra quotes from environment variables"""
    if value:
        if '#' in value:
            value = value.split('#')[0]
        value = value.strip().strip('"\'')
    return value

# Environment setup
API_KEY = clean_env_var(os.environ.get("API_KEY", "changeme"))
POSTGRES_URI = clean_env_var(os.environ.get("POSTGRES_URI"))
MONGO_URI = clean_env_var(os.environ.get("MONGO_URI"))
MONGO_DB = clean_env_var(os.environ.get("MONGO_DB"))
MONGO_COLL = clean_env_var(os.environ.get("MONGO_COLL"))
TWITTER_BEARER_TOKEN = clean_env_var(os.environ.get("TWITTER_BEARER_TOKEN"))

# Validate required environment variables
if not all([POSTGRES_URI, MONGO_URI, MONGO_DB, MONGO_COLL]):
    logger.error("Missing required environment variables!")
    sys.exit(1)

API_KEY_NAME = "access_token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Request model for ingestion
class IngestionRequest(BaseModel):
    term: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

# FastAPI application
app = FastAPI(
    title="Campaign Sentiment Analytics API",
    description="API for sentiment measurements with hourly analytics",
    version="2.2.0"
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": API_KEY_NAME
        }
    }
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method.setdefault("security", []).append({"APIKeyHeader": []})
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

def get_api_key(api_key_header: str = Security(api_key_header)):
    """Validate API key"""
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, 
            detail="Could not validate credentials"
        )

def ensure_api_database_schema():
    """Ensure database schema is compatible with API expectations including hourly analytics"""
    try:
        # Check if cleaned_tweets table has required columns
        with engine.connect() as conn:
            # Get current columns
            columns_query = text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'cleaned_tweets'
                ORDER BY ordinal_position
            """)
            
            result = conn.execute(columns_query)
            existing_columns = {row[0]: row[1] for row in result.fetchall()}
            
            logger.info(f"Existing cleaned_tweets columns: {list(existing_columns.keys())}")
            
            # List of columns that should exist for full API functionality
            required_columns = [
                ("confidence_score", "double precision"),
                ("country", "character varying"),
                ("country_code", "character varying"),
                ("region", "character varying"),
                ("city", "character varying"),
                ("username", "character varying"),
                ("created_hour", "timestamp")  # NEW: for hourly aggregation
            ]
            
            # Add missing columns
            for column_name, column_type in required_columns:
                if column_name not in existing_columns:
                    try:
                        if column_type == "double precision":
                            postgres_type = "FLOAT"
                        elif column_type == "character varying":
                            postgres_type = "VARCHAR(255)" if column_name == "username" else "VARCHAR(100)"
                        elif column_type == "timestamp":
                            postgres_type = "TIMESTAMP"
                        else:
                            postgres_type = column_type
                            
                        add_column_sql = text(f"ALTER TABLE cleaned_tweets ADD COLUMN {column_name} {postgres_type}")
                        conn.execute(add_column_sql)
                        conn.commit()
                        logger.info(f"✅ API: Added missing column {column_name}")
                    except Exception as e:
                        logger.warning(f"Could not add column {column_name}: {e}")
            
            # Create user_metrics table if it doesn't exist
            create_user_metrics_sql = """
            CREATE TABLE IF NOT EXISTS user_metrics (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) UNIQUE,
                username VARCHAR(255),
                follower_count INT DEFAULT 0,
                following_count INT DEFAULT 0,
                tweet_count INT DEFAULT 0,
                verified BOOLEAN DEFAULT FALSE,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            conn.execute(text(create_user_metrics_sql))
            
            # NEW: Create hourly analytics table if it doesn't exist
            create_hourly_analytics_sql = """
            CREATE TABLE IF NOT EXISTS tweet_analytics_hourly (
                id SERIAL PRIMARY KEY,
                campaign VARCHAR(255),
                datetime TIMESTAMP,
                hour INT,
                avg_sentiment FLOAT,
                tweet_count INT,
                pos_count INT,
                neu_count INT,
                neg_count INT,
                UNIQUE(campaign, datetime)
            );
            """
            
            conn.execute(text(create_hourly_analytics_sql))
            
            # Create index for hourly analytics
            create_hourly_index_sql = """
            CREATE INDEX IF NOT EXISTS idx_hourly_campaign_datetime 
            ON tweet_analytics_hourly (campaign, datetime);
            """
            
            conn.execute(text(create_hourly_index_sql))
            
            conn.commit()
            
            logger.info("✅ API database schema validated (including hourly analytics)")
            
    except Exception as e:
        logger.warning(f"Could not validate database schema: {e}")

# Database connections
try:
    engine = create_engine(POSTGRES_URI)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("PostgreSQL connection successful")
    
    # Ensure schema compatibility
    ensure_api_database_schema()
    
except Exception as e:
    logger.error(f"Failed to connect to PostgreSQL: {e}")
    sys.exit(1)

try:
    mongo_client = MongoClient(MONGO_URI)
    mongo_client.server_info()
    mongo_db = mongo_client[MONGO_DB]
    tweets_collection = mongo_db[MONGO_COLL]
    logger.info(f"MongoDB connection successful")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    sys.exit(1)

def get_available_columns():
    """Get list of available columns in cleaned_tweets table"""
    try:
        with engine.connect() as conn:
            columns_query = text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'cleaned_tweets'
                ORDER BY ordinal_position
            """)
            result = conn.execute(columns_query)
            return [row[0] for row in result.fetchall()]
    except Exception as e:
        logger.error(f"Error getting available columns: {e}")
        return ['tweet_id', 'campaign', 'original_text', 'sentiment_label', 'sentiment_score', 'created_at']

def build_safe_query(base_query: str, optional_columns: list = None):
    """Build a query that only uses available columns"""
    available_columns = get_available_columns()
    
    if optional_columns:
        # Filter out columns that don't exist
        safe_columns = [col for col in optional_columns if col in available_columns]
        missing_columns = [col for col in optional_columns if col not in available_columns]
        
        if missing_columns:
            logger.info(f"Columns not available: {missing_columns}")
            
        # Replace placeholder in query
        if "{optional_columns}" in base_query:
            if safe_columns:
                columns_str = ", " + ", ".join(safe_columns)
            else:
                columns_str = ""
            base_query = base_query.replace("{optional_columns}", columns_str)
    
    return base_query

# ============================================================================
# REAL-TIME STATUS ENDPOINTS
# ============================================================================

@app.get("/ingestion_status/{term}")
def get_ingestion_status(
    term: str,
    api_key: APIKey = Depends(get_api_key)
):
    """Get real-time status of campaign ingestion and processing"""
    try:
        # Check MongoDB for raw tweets
        raw_tweet_count = tweets_collection.count_documents({"matched_term": term})
        
        # Check PostgreSQL for processed tweets
        query = text("SELECT COUNT(*) as processed_count FROM cleaned_tweets WHERE campaign = :term")
        result = pd.read_sql(query, engine, params={"term": term})
        processed_count = result.iloc[0]['processed_count'] if not result.empty else 0
        
        # Check if campaign exists in analytics
        analytics_query = text("SELECT COUNT(*) as analytics_count FROM tweet_analytics WHERE campaign = :term")
        analytics_result = pd.read_sql(analytics_query, engine, params={"term": term})
        analytics_count = analytics_result.iloc[0]['analytics_count'] if not analytics_result.empty else 0
        
        # Determine status
        if analytics_count > 0:
            status = "✅ Complete - Available in dashboard"
        elif processed_count > 0:
            status = "⚡ Processing - Aggregating analytics..."
        elif raw_tweet_count > 0:
            status = "🔄 Processing - Analyzing sentiment..."
        else:
            status = "⏳ Starting - Collecting tweets..."
        
        return {
            "term": term,
            "status": status,
            "raw_tweets": raw_tweet_count,
            "processed_tweets": processed_count,
            "analytics_ready": analytics_count > 0,
            "estimated_completion": "30-60 seconds" if analytics_count == 0 else "Ready now"
        }
        
    except Exception as e:
        logger.error(f"Error checking status for {term}: {e}")
        return {
            "term": term,
            "status": "❌ Error checking status",
            "error": str(e)
        }

# ============================================================================
# CORE ENDPOINTS
# ============================================================================

@app.get("/health")
def health_check():
    """Check if the API and databases are accessible"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        mongo_client.server_info()
        
        # Check available columns for diagnostics
        available_columns = get_available_columns()
        
        # Check if hourly analytics table exists
        hourly_table_exists = False
        hourly_record_count = 0
        try:
            check_query = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'tweet_analytics_hourly'
                );
            """)
            
            with engine.connect() as conn:
                hourly_table_exists = conn.execute(check_query).scalar()
                
                if hourly_table_exists:
                    count_query = text("SELECT COUNT(*) FROM tweet_analytics_hourly")
                    hourly_record_count = conn.execute(count_query).scalar()
        except:
            pass
        
        return {
            "status": "healthy",
            "postgres": "connected",
            "mongodb": "connected",
            "available_columns": len(available_columns),
            "schema_version": "2.2",
            "hourly_analytics": {
                "table_exists": hourly_table_exists,
                "record_count": hourly_record_count,
                "status": "available" if hourly_table_exists and hourly_record_count > 0 else "not available"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/start_ingest/")
def start_ingest(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
    api_key: APIKey = Depends(get_api_key)
):
    """Start tweet ingestion for a new search term with optional date range"""
    term = request.term
    start_date = request.start_date
    end_date = request.end_date
    
    logger.info(f"Starting ingestion for term: {term}")
    if start_date:
        logger.info(f"Date range: {start_date} to {end_date}")
    
    def run_ingest():
        try:
            env = os.environ.copy()
            env.update({
                'TWITTER_BEARER_TOKEN': TWITTER_BEARER_TOKEN or '',
                'MONGO_URI': MONGO_URI,
                'MONGO_DB': MONGO_DB,
                'MONGO_COLL': MONGO_COLL,
                'MAX_TWEETS': os.environ.get('MAX_TWEETS', '100')
            })
            
            cmd = ["python", "ingest.py", term]
            if start_date:
                cmd.extend(["--start", start_date])
            if end_date:
                cmd.extend(["--end", end_date])
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                env=env,
                cwd="/app"
            )
            
            if result.returncode != 0:
                logger.error(f"Ingestion failed for {term}: {result.stderr}")
            else:
                logger.info(f"Ingestion completed for {term}")
        except Exception as e:
            logger.error(f"Failed to run ingestion: {e}")
    
    background_tasks.add_task(run_ingest)
    
    response = {"status": "ingestion started", "term": term}
    if start_date:
        response["date_range"] = f"{start_date} to {end_date}"
    
    return response

@app.get("/campaigns/")
def get_campaigns(api_key: APIKey = Depends(get_api_key)):
    """Returns a list of all unique campaigns"""
    try:
        q = "SELECT DISTINCT campaign FROM tweet_analytics ORDER BY campaign;"
        campaigns = pd.read_sql(q, engine)['campaign'].tolist()
        return campaigns
    except Exception as e:
        logger.error(f"Error fetching campaigns: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch campaigns")

@app.get("/date_range/")
def get_min_max_dates(api_key: APIKey = Depends(get_api_key)):
    """Returns the earliest and latest dates in the analytics table"""
    try:
        q = "SELECT MIN(date) as min_date, MAX(date) as max_date FROM tweet_analytics;"
        result = pd.read_sql(q, engine)
        
        if result.iloc[0]['min_date'] is None:
            import datetime
            today = datetime.date.today()
            return {
                "min_date": str(today - datetime.timedelta(days=7)),
                "max_date": str(today)
            }
        
        d = result.iloc[0].to_dict()
        d = {k: str(v) for k, v in d.items()}
        return d
    except Exception as e:
        logger.error(f"Error fetching date range: {e}")
        import datetime
        today = datetime.date.today()
        return {
            "min_date": str(today - datetime.timedelta(days=7)),
            "max_date": str(today)
        }

@app.get("/analytics/")
def analytics(
    campaigns: str = Query(..., description="Comma-separated campaign names"),
    start: str = Query(..., description="Start date YYYY-MM-DD"),
    end: str = Query(..., description="End date YYYY-MM-DD"),
    api_key: APIKey = Depends(get_api_key)
):
    """Returns aggregated analytics for selected campaigns and date range"""
    try:
        campaign_list = [c.strip() for c in campaigns.split(",")]
        query = text("""
            SELECT campaign, date, avg_sentiment, tweet_count, pos_count, neu_count, neg_count
            FROM tweet_analytics
            WHERE campaign IN :campaigns
              AND date BETWEEN :start AND :end
            ORDER BY campaign, date;
        """)
        df = pd.read_sql(query, engine, params={
            "campaigns": tuple(campaign_list), 
            "start": start, 
            "end": end
        })
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch analytics")

@app.get("/analytics_hourly/")
def analytics_hourly(
    campaigns: str = Query(..., description="Comma-separated campaign names"),
    start: str = Query(..., description="Start date YYYY-MM-DD"),
    end: str = Query(..., description="End date YYYY-MM-DD"),
    api_key: APIKey = Depends(get_api_key)
):
    """Returns hourly aggregated analytics for selected campaigns and date range"""
    try:
        campaign_list = [c.strip() for c in campaigns.split(",")]
        
        # Convert dates to datetime for hourly query
        start_datetime = f"{start} 00:00:00"
        end_datetime = f"{end} 23:59:59"
        
        # First check if hourly table exists
        check_table_query = text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'tweet_analytics_hourly'
            );
        """)
        
        with engine.connect() as conn:
            table_exists = conn.execute(check_table_query).scalar()
        
        if not table_exists:
            logger.warning("Hourly analytics table doesn't exist, falling back to daily data")
            # Fall back to daily analytics
            return analytics(campaigns, start, end, api_key)
        
        query = text("""
            SELECT 
                campaign,
                datetime,
                hour,
                avg_sentiment,
                tweet_count,
                pos_count,
                neu_count,
                neg_count
            FROM tweet_analytics_hourly
            WHERE campaign IN :campaigns
              AND datetime BETWEEN :start AND :end
            ORDER BY campaign, datetime;
        """)
        
        df = pd.read_sql(query, engine, params={
            "campaigns": tuple(campaign_list), 
            "start": start_datetime, 
            "end": end_datetime
        })
        
        # Format datetime for JSON serialization
        if not df.empty:
            df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"Returning {len(df)} hourly data points for campaigns: {campaign_list}")
        return df.to_dict(orient="records")
        
    except Exception as e:
        logger.error(f"Error fetching hourly analytics: {e}")
        # Fallback to daily data if hourly fails
        logger.info("Falling back to daily analytics due to error")
        return analytics(campaigns, start, end, api_key)

@app.get("/date_range_hourly/")
def get_hourly_date_range(api_key: APIKey = Depends(get_api_key)):
    """Returns the earliest and latest datetime in the hourly analytics table"""
    try:
        # Check if hourly table exists
        check_table_query = text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'tweet_analytics_hourly'
            );
        """)
        
        with engine.connect() as conn:
            table_exists = conn.execute(check_table_query).scalar()
        
        if not table_exists:
            # Return default range if table doesn't exist
            import datetime
            now = datetime.datetime.now()
            return {
                "min_datetime": str(now - datetime.timedelta(days=1)),
                "max_datetime": str(now),
                "available": False
            }
        
        q = """
            SELECT 
                MIN(datetime) as min_datetime, 
                MAX(datetime) as max_datetime,
                COUNT(DISTINCT campaign) as campaign_count,
                COUNT(*) as total_records
            FROM tweet_analytics_hourly;
        """
        result = pd.read_sql(q, engine)
        
        if result.iloc[0]['min_datetime'] is None:
            import datetime
            now = datetime.datetime.now()
            return {
                "min_datetime": str(now - datetime.timedelta(days=1)),
                "max_datetime": str(now),
                "available": False
            }
        
        return {
            "min_datetime": str(result.iloc[0]['min_datetime']),
            "max_datetime": str(result.iloc[0]['max_datetime']),
            "campaign_count": int(result.iloc[0]['campaign_count']),
            "total_records": int(result.iloc[0]['total_records']),
            "available": True
        }
    except Exception as e:
        logger.error(f"Error fetching hourly date range: {e}")
        import datetime
        now = datetime.datetime.now()
        return {
            "min_datetime": str(now - datetime.timedelta(days=1)),
            "max_datetime": str(now),
            "available": False,
            "error": str(e)
        }

@app.get("/analytics_summary/")
def analytics_summary(
    campaigns: str = Query(..., description="Comma-separated campaign names"),
    start: str = Query(..., description="Start date YYYY-MM-DD"),
    end: str = Query(..., description="End date YYYY-MM-DD"),
    api_key: APIKey = Depends(get_api_key)
):
    """Returns summary statistics combining daily and hourly data"""
    try:
        campaign_list = [c.strip() for c in campaigns.split(",")]
        
        # Get daily summary
        daily_query = text("""
            SELECT 
                COUNT(DISTINCT date) as days_tracked,
                SUM(tweet_count) as total_tweets,
                AVG(avg_sentiment) as overall_sentiment,
                SUM(pos_count) as total_positive,
                SUM(neg_count) as total_negative,
                SUM(neu_count) as total_neutral
            FROM tweet_analytics
            WHERE campaign IN :campaigns
              AND date BETWEEN :start AND :end;
        """)
        
        daily_result = pd.read_sql(daily_query, engine, params={
            "campaigns": tuple(campaign_list),
            "start": start,
            "end": end
        })
        
        # Try to get hourly summary for last 24 hours
        hourly_summary = {
            "hourly_data_available": False,
            "last_24h_tweets": 0,
            "last_24h_sentiment": 0
        }
        
        try:
            # Check if hourly table exists
            check_table_query = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'tweet_analytics_hourly'
                );
            """)
            
            with engine.connect() as conn:
                table_exists = conn.execute(check_table_query).scalar()
            
            if table_exists:
                import datetime
                last_24h = datetime.datetime.now() - datetime.timedelta(hours=24)
                
                hourly_query = text("""
                    SELECT 
                        COUNT(*) as hourly_records,
                        SUM(tweet_count) as tweets_24h,
                        AVG(avg_sentiment) as sentiment_24h
                    FROM tweet_analytics_hourly
                    WHERE campaign IN :campaigns
                      AND datetime >= :last_24h;
                """)
                
                hourly_result = pd.read_sql(hourly_query, engine, params={
                    "campaigns": tuple(campaign_list),
                    "last_24h": last_24h
                })
                
                if not hourly_result.empty and hourly_result.iloc[0]['hourly_records'] > 0:
                    hourly_summary = {
                        "hourly_data_available": True,
                        "last_24h_tweets": int(hourly_result.iloc[0]['tweets_24h'] or 0),
                        "last_24h_sentiment": float(hourly_result.iloc[0]['sentiment_24h'] or 0)
                    }
        except Exception as e:
            logger.warning(f"Could not fetch hourly summary: {e}")
        
        # Combine results
        if not daily_result.empty:
            summary = {
                "campaigns": campaign_list,
                "date_range": f"{start} to {end}",
                "days_tracked": int(daily_result.iloc[0]['days_tracked'] or 0),
                "total_tweets": int(daily_result.iloc[0]['total_tweets'] or 0),
                "overall_sentiment": float(daily_result.iloc[0]['overall_sentiment'] or 0),
                "sentiment_breakdown": {
                    "positive": int(daily_result.iloc[0]['total_positive'] or 0),
                    "neutral": int(daily_result.iloc[0]['total_neutral'] or 0),
                    "negative": int(daily_result.iloc[0]['total_negative'] or 0)
                },
                **hourly_summary
            }
            
            # Calculate percentages
            total = summary["sentiment_breakdown"]["positive"] + summary["sentiment_breakdown"]["neutral"] + summary["sentiment_breakdown"]["negative"]
            if total > 0:
                summary["sentiment_percentages"] = {
                    "positive": round(summary["sentiment_breakdown"]["positive"] / total * 100, 1),
                    "neutral": round(summary["sentiment_breakdown"]["neutral"] / total * 100, 1),
                    "negative": round(summary["sentiment_breakdown"]["negative"] / total * 100, 1)
                }
            else:
                summary["sentiment_percentages"] = {
                    "positive": 0,
                    "neutral": 0,
                    "negative": 0
                }
            
            return summary
        else:
            return {
                "campaigns": campaign_list,
                "date_range": f"{start} to {end}",
                "error": "No data found for the specified parameters"
            }
            
    except Exception as e:
        logger.error(f"Error fetching analytics summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch analytics summary")

# ============================================================================
# KEY MEASUREMENT ENDPOINTS
# ============================================================================

@app.get("/influencers/")
def get_influencers(
    campaigns: str = Query(..., description="Comma-separated campaign names"),
    start: str = Query(..., description="Start date YYYY-MM-DD"),
    end: str = Query(..., description="End date YYYY-MM-DD"),
    limit: int = Query(20, description="Number of top influencers"),
    api_key: APIKey = Depends(get_api_key)
):
    """Returns influencers driving sentiment (users with high follower counts and engagement)"""
    try:
        campaign_list = [c.strip() for c in campaigns.split(",")]
        
        # Check if user_metrics table exists and has data
        try:
            with engine.connect() as conn:
                test_query = text("SELECT COUNT(*) FROM user_metrics")
                user_metrics_count = conn.execute(test_query).fetchone()[0]
        except:
            user_metrics_count = 0
        
        if user_metrics_count > 0:
            # Query with user_metrics join for real follower data
            query = text("""
                SELECT 
                    ct.user_id,
                    COALESCE(ct.username, um.username, ct.user_id) as username,
                    COALESCE(um.follower_count, 0) as follower_count,
                    COUNT(*) as tweet_count,
                    AVG(ct.sentiment_score) as avg_sentiment,
                    SUM(CASE WHEN ct.sentiment_label = 'POSITIVE' THEN 1 ELSE 0 END) as positive_count,
                    SUM(CASE WHEN ct.sentiment_label = 'NEGATIVE' THEN 1 ELSE 0 END) as negative_count,
                    (COALESCE(um.follower_count, 0) * COUNT(*)) as reach_score
                FROM cleaned_tweets ct
                LEFT JOIN user_metrics um ON ct.user_id = um.user_id
                WHERE ct.campaign = ANY(:campaigns)
                  AND DATE(ct.created_at) BETWEEN :start AND :end
                  AND COALESCE(um.follower_count, 0) > 0
                GROUP BY ct.user_id, ct.username, um.username, um.follower_count
                HAVING COUNT(*) >= 1
                ORDER BY reach_score DESC
                LIMIT :limit
            """)
        else:
            # Fallback query without user_metrics
            query = text("""
                SELECT 
                    user_id,
                    COALESCE(username, user_id) as username,
                    0 as follower_count,
                    COUNT(*) as tweet_count,
                    AVG(sentiment_score) as avg_sentiment,
                    SUM(CASE WHEN sentiment_label = 'POSITIVE' THEN 1 ELSE 0 END) as positive_count,
                    SUM(CASE WHEN sentiment_label = 'NEGATIVE' THEN 1 ELSE 0 END) as negative_count,
                    COUNT(*) as reach_score
                FROM cleaned_tweets
                WHERE campaign = ANY(:campaigns)
                  AND DATE(created_at) BETWEEN :start AND :end
                GROUP BY user_id, username
                HAVING COUNT(*) >= 1
                ORDER BY tweet_count DESC
                LIMIT :limit
            """)
        
        df = pd.read_sql(
            query,
            engine,
            params={
                "campaigns": campaign_list,
                "start": start,
                "end": end,
                "limit": limit
            }
        )
        
        return df.to_dict(orient="records") if not df.empty else []
            
    except Exception as e:
        logger.error(f"Error fetching influencers: {e}")
        return []

@app.get("/geographic_sentiment/")
def get_geographic_sentiment(
    campaigns: str = Query(..., description="Comma-separated campaign names"),
    start: str = Query(..., description="Start date YYYY-MM-DD"),
    end: str = Query(..., description="End date YYYY-MM-DD"),
    api_key: APIKey = Depends(get_api_key)
):
    """Returns geographic sentiment distribution from real PostgreSQL data"""
    try:
        campaign_list = [c.strip() for c in campaigns.split(",")]
        
        # Check if geographic columns exist
        available_columns = get_available_columns()
        has_geo_columns = 'country' in available_columns and 'country_code' in available_columns
        
        logger.info(f"Querying geographic data for campaigns: {campaign_list}")
        logger.info(f"Geographic columns available: {has_geo_columns}")
        logger.info(f"Available columns: {available_columns}")
        
        if has_geo_columns:
            # First, let's see how much data we have
            count_query = text("""
                SELECT 
                    COUNT(*) as total_tweets,
                    COUNT(country) as tweets_with_country,
                    COUNT(DISTINCT country) as unique_countries
                FROM cleaned_tweets
                WHERE campaign = ANY(:campaigns)
                  AND DATE(created_at) BETWEEN :start AND :end
            """)
            
            count_result = pd.read_sql(
                count_query,
                engine,
                params={
                    "campaigns": campaign_list,
                    "start": start,
                    "end": end
                }
            )
            
            logger.info(f"Data summary: {count_result.iloc[0].to_dict()}")
            
            # Main geographic query
            query = text("""
                SELECT 
                    country,
                    country_code,
                    COUNT(*) as mention_count,
                    AVG(sentiment_score) as avg_sentiment,
                    SUM(CASE WHEN sentiment_label = 'POSITIVE' THEN 1 ELSE 0 END) as positive_count,
                    SUM(CASE WHEN sentiment_label = 'NEGATIVE' THEN 1 ELSE 0 END) as negative_count,
                    SUM(CASE WHEN sentiment_label = 'NEUTRAL' THEN 1 ELSE 0 END) as neutral_count
                FROM cleaned_tweets
                WHERE campaign = ANY(:campaigns)
                  AND DATE(created_at) BETWEEN :start AND :end
                  AND country IS NOT NULL
                GROUP BY country, country_code
                ORDER BY mention_count DESC
            """)
            
            df = pd.read_sql(
                query,
                engine,
                params={
                    "campaigns": campaign_list,
                    "start": start,
                    "end": end
                }
            )
            
            logger.info(f"Geographic query returned {len(df)} countries")
            
            if not df.empty:
                result = df.to_dict(orient="records")
                logger.info(f"Returning real geographic data: {[r['country'] for r in result]}")
                return result
            else:
                logger.info("No tweets with country data found")
                return []
        else:
            logger.warning("Geographic columns not available in database")
            return []
            
    except Exception as e:
        logger.error(f"Error fetching geographic sentiment: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []

# ============================================================================
# LEGACY ENDPOINTS (for backward compatibility)
# ============================================================================

@app.get("/latest_tweets/")
def latest_tweets(
    campaigns: str = Query(..., description="Comma-separated campaign names"),
    start: str = Query(..., description="Start date YYYY-MM-DD"),
    end: str = Query(..., description="End date YYYY-MM-DD"),
    limit: int = Query(100, description="Max tweets"),
    api_key: APIKey = Depends(get_api_key)
):
    """Returns latest tweets with sentiment (for backward compatibility)"""
    try:
        campaign_list = [c.strip() for c in campaigns.split(",")]
        
        # Build query with available columns
        available_columns = get_available_columns()
        
        base_columns = [
            'tweet_id', 'campaign', 'sentiment_label', 'sentiment_score', 'created_at', 'user_id'
        ]
        
        # Handle text columns
        if 'original_text' in available_columns:
            text_select = 'original_text as text'
        elif 'text' in available_columns:
            text_select = 'text'
        else:
            text_select = "'No text available' as text"
        
        # Handle cleaned_text
        if 'cleaned_text' in available_columns:
            cleaned_text_select = ', cleaned_text'
        else:
            cleaned_text_select = ''
        
        query = text(f"""
            SELECT 
                tweet_id,
                campaign,
                {text_select},
                sentiment_label,
                sentiment_score,
                created_at,
                user_id
                {cleaned_text_select}
            FROM cleaned_tweets
            WHERE campaign = ANY(:campaigns)
              AND DATE(created_at) BETWEEN :start AND :end
            ORDER BY created_at DESC
            LIMIT :limit
        """)
        
        df = pd.read_sql(
            query, 
            engine,
            params={
                "campaigns": campaign_list, 
                "start": start, 
                "end": end,
                "limit": limit
            }
        )
        
        if not df.empty:
            tweets = df.to_dict(orient="records")
            for tweet in tweets:
                tweet["created_at"] = str(tweet["created_at"])
            return tweets
        else:
            return []
            
    except Exception as e:
        logger.error(f"Error fetching tweets: {e}")
        return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)