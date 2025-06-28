# backend_api.py
# Provides FastAPI endpoints for analytics and latest tweets for the dashboard.

from fastapi import FastAPI, Query
from sqlalchemy import create_engine, text
from pymongo import MongoClient
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

app = FastAPI()

# PostgreSQL connection for analytics
engine = create_engine(os.environ.get("POSTGRES_URI"))

# MongoDB connection for recent/raw tweets
MONGO_URI = os.environ.get("MONGO_URI")
MONGO_DB = os.environ.get("MONGO_DB")
MONGO_COLL = os.environ.get("MONGO_COLL")
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB]
tweets_collection = mongo_db[MONGO_COLL]

@app.get("/campaigns/")
def get_campaigns():
    """Return list of distinct campaigns."""
    q = "SELECT DISTINCT campaign FROM tweet_analytics ORDER BY campaign;"
    campaigns = pd.read_sql(q, engine)['campaign'].tolist()
    return campaigns

@app.get("/date_range/")
def get_min_max_dates():
    """Return the min and max dates in analytics."""
    q = "SELECT MIN(date) as min_date, MAX(date) as max_date FROM tweet_analytics;"
    result = pd.read_sql(q, engine)
    d = result.iloc[0].to_dict()
    # Convert to str for JSON serialization
    d = {k: str(v) for k, v in d.items()}
    return d

@app.get("/analytics/")
def analytics(
    campaigns: str = Query(..., description="Comma-separated campaign names"),
    start: str = Query(..., description="Start date YYYY-MM-DD"),
    end: str = Query(..., description="End date YYYY-MM-DD")
):
    """Return aggregated analytics for selected campaigns and date range."""
    campaign_list = [c.strip() for c in campaigns.split(",")]
    query = text("""
        SELECT campaign, date, avg_sentiment, tweet_count, pos_count, neu_count, neg_count
        FROM tweet_analytics
        WHERE campaign IN :campaigns
          AND date BETWEEN :start AND :end
        ORDER BY campaign, date;
    """)
    df = pd.read_sql(query, engine, params={"campaigns": tuple(campaign_list), "start": start, "end": end})
    return df.to_dict(orient="records")

@app.get("/latest_tweets/")
def latest_tweets(
    campaigns: str = Query(..., description="Comma-separated campaign names"),
    start: str = Query(..., description="Start date YYYY-MM-DD"),
    end: str = Query(..., description="End date YYYY-MM-DD"),
    limit: int = Query(100, description="Max tweets")
):
    """Return latest tweets (with sentiment) for colored table/word cloud."""
    campaign_list = [c.strip() for c in campaigns.split(",")]
    query_mongo = {
        "matched_term": {"$in": campaign_list},
        "created_at": {"$gte": start, "$lte": end}
    }
    fields = {"text": 1, "sentiment_label": 1, "created_at": 1}
    tweets = list(tweets_collection.find(query_mongo, fields).sort("created_at", -1).limit(limit))
    # Convert datetime for JSON serialization
    for t in tweets:
        if "created_at" in t:
            t["created_at"] = str(t["created_at"])
        if "_id" in t:
            t.pop("_id")  # Remove MongoDB's ObjectId from response
    return tweets
