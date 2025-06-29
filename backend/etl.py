from dotenv import load_dotenv
load_dotenv()  # Loads environment variables from .env

import os
import pandas as pd
from pymongo import MongoClient
from sqlalchemy import create_engine, text
from transformers import pipeline
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# --- Load and clean database URIs from .env ---
MONGO_URI = clean_env_var(os.environ.get("MONGO_URI"))
MONGO_DB = clean_env_var(os.environ.get("MONGO_DB"))
MONGO_COLL = clean_env_var(os.environ.get("MONGO_COLL"))
POSTGRES_URI = clean_env_var(os.environ.get("POSTGRES_URI"))

# Log configuration (without exposing sensitive data)
logger.info(f"MongoDB Database: {MONGO_DB}")
logger.info(f"MongoDB Collection: {MONGO_COLL}")
logger.info(f"PostgreSQL URI: {'Set' if POSTGRES_URI else 'Not Set'}")
logger.info(f"MongoDB URI: {'Set' if MONGO_URI else 'Not Set'}")

# Validate required environment variables
if not all([MONGO_URI, MONGO_DB, MONGO_COLL, POSTGRES_URI]):
    logger.error("Missing required environment variables!")
    logger.error("Please ensure MONGO_URI, MONGO_DB, MONGO_COLL, and POSTGRES_URI are set in .env")
    exit(1)

# --- Connect to MongoDB ---
try:
    mongo_client = MongoClient(MONGO_URI)
    # Test connection
    mongo_client.server_info()
    mongo_db = mongo_client[MONGO_DB]
    tweets_collection = mongo_db[MONGO_COLL]  # This was missing in your original code!
    logger.info(f"Connected to MongoDB - Database: {MONGO_DB}, Collection: {MONGO_COLL}")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    exit(1)

# --- Connect to PostgreSQL ---
try:
    engine = create_engine(POSTGRES_URI)
    # Test connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Connected to PostgreSQL")
except Exception as e:
    logger.error(f"Failed to connect to PostgreSQL: {e}")
    exit(1)

# --- HuggingFace RoBERTa sentiment pipeline ---
logger.info("Loading sentiment analysis model...")
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    logger.info("Sentiment model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load sentiment model: {e}")
    exit(1)

# --- Ensure analytics table exists ---
create_table_sql = """
CREATE TABLE IF NOT EXISTS tweet_analytics (
    id SERIAL PRIMARY KEY,
    campaign VARCHAR(255),
    date DATE,
    avg_sentiment FLOAT,
    tweet_count INT,
    pos_count INT,
    neu_count INT,
    neg_count INT,
    UNIQUE(campaign, date)  -- Prevent duplicate entries
);
"""
try:
    with engine.connect() as conn:
        conn.execute(text(create_table_sql))
        conn.commit()
    logger.info("Analytics table ready")
except Exception as e:
    logger.error(f"Failed to create table: {e}")
    exit(1)

# --- Find last 3 months of tweets missing sentiment, then label them ---
start_date = datetime.utcnow() - timedelta(days=90)
logger.info(f"Processing tweets from {start_date.isoformat()} onwards")

# Count tweets to process
count_missing = tweets_collection.count_documents({
    "created_at": {"$gte": start_date.isoformat()},
    "sentiment_label": {"$exists": False}
})
logger.info(f"Found {count_missing} tweets without sentiment labels")

# Process tweets missing sentiment
tweets_cursor = tweets_collection.find({
    "created_at": {"$gte": start_date.isoformat()},
    "sentiment_label": {"$exists": False}
}).limit(1000)  # Process in batches to avoid memory issues

tweet_docs = []
processed_count = 0

for tweet in tweets_cursor:
    try:
        text = tweet.get('text', '')
        if not text:
            continue
            
        # Score tweet with RoBERTa model (truncate at 512 chars)
        result = sentiment_analyzer(text[:512])[0]
        
        # Map model labels to expected format
        label_mapping = {
            'LABEL_0': 'NEGATIVE',
            'LABEL_1': 'NEUTRAL',
            'LABEL_2': 'POSITIVE',
            'negative': 'NEGATIVE',
            'neutral': 'NEUTRAL',
            'positive': 'POSITIVE'
        }
        
        sentiment_label = label_mapping.get(result['label'], result['label'].upper())
        
        # Update MongoDB with sentiment results
        tweets_collection.update_one(
            {"_id": tweet["_id"]},
            {"$set": {
                "sentiment_label": sentiment_label,
                "sentiment_score": float(result['score'])
            }}
        )
        
        tweet['sentiment_label'] = sentiment_label
        tweet['sentiment_score'] = float(result['score'])
        tweet_docs.append(tweet)
        
        processed_count += 1
        if processed_count % 100 == 0:
            logger.info(f"Processed {processed_count} tweets")
            
    except Exception as e:
        logger.error(f"Error scoring tweet {tweet.get('_id')}: {e}")

logger.info(f"Labeled {processed_count} new tweets")

# Also fetch already-scored tweets in the 3-month window
already_scored_cursor = tweets_collection.find({
    "created_at": {"$gte": start_date.isoformat()},
    "sentiment_label": {"$exists": True}
})
tweet_docs.extend(list(already_scored_cursor))

if not tweet_docs:
    logger.warning("No tweets found in time range.")
    exit()

logger.info(f"Total tweets to aggregate: {len(tweet_docs)}")

# Create dataframe
df = pd.DataFrame(tweet_docs)

# --- Extract campaign from matched_term for analytics ---
df['campaign'] = df['matched_term'].fillna('unknown')
df['date'] = pd.to_datetime(df['created_at']).dt.date

# Handle missing sentiment data
df['sentiment_label'] = df['sentiment_label'].fillna('UNKNOWN')
df['sentiment_score'] = df['sentiment_score'].fillna(0.0)

# --- Aggregate by campaign, date, and sentiment label counts ---
agg = df.groupby(['campaign', 'date']).agg(
    avg_sentiment=('sentiment_score', 'mean'),
    tweet_count=('tweet_id', 'count'),
    pos_count=('sentiment_label', lambda x: (x == 'POSITIVE').sum()),
    neu_count=('sentiment_label', lambda x: (x == 'NEUTRAL').sum()),
    neg_count=('sentiment_label', lambda x: (x == 'NEGATIVE').sum())
).reset_index()

logger.info(f"Aggregated data into {len(agg)} campaign-date combinations")

# --- Write aggregates to PostgreSQL ---
try:
    # Delete existing data for the date range to avoid duplicates
    with engine.connect() as conn:
        delete_sql = text("""
            DELETE FROM tweet_analytics 
            WHERE date >= :start_date
        """)
        conn.execute(delete_sql, {"start_date": start_date.date()})
        conn.commit()
    
    # Insert new aggregated data
    agg.to_sql('tweet_analytics', engine, if_exists='append', index=False)
    logger.info(f"Successfully wrote {len(agg)} rows to PostgreSQL")
    
except Exception as e:
    logger.error(f"Failed to write to PostgreSQL: {e}")
    exit(1)

# Close connections
mongo_client.close()
engine.dispose()

logger.info("ETL with RoBERTa sentiment and full aggregation to PostgreSQL completed.")