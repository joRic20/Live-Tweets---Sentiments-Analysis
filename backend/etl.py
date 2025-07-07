from dotenv import load_dotenv
load_dotenv()  # Loads environment variables from .env

import os
import pandas as pd
from pymongo import MongoClient
from sqlalchemy import create_engine, text
from transformers import pipeline
from datetime import datetime, timedelta
import logging
import re  # Added for text cleaning

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

# Text cleaning function for better sentiment analysis
def clean_text_for_sentiment(text):
    """
    Clean tweet text for more accurate sentiment analysis
    
    Args:
        text (str): Raw tweet text
    
    Returns:
        str: Cleaned tweet text
    """
    if not text:
        return ""
    
    # 1. Remove URLs (http, https, www)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # 2. Remove RT prefix and the username after it
    text = re.sub(r'^RT\s*@\w+\s*:\s*', '', text, flags=re.IGNORECASE)
    
    # 3. Handle @mentions
    # If there are 3 or more mentions, replace with a single placeholder
    if len(re.findall(r'@\w+', text)) >= 3:
        text = re.sub(r'(@\w+\s*)+', '@users ', text)
    
    # 4. Clean hashtags - keep them but remove excessive hashtag spam
    hashtags = re.findall(r'#\w+', text)
    if len(hashtags) > 5:  # Too many hashtags is likely spam
        # Remove all hashtags if there are too many
        text = re.sub(r'#\w+', '', text)
    
    # 5. Remove extra whitespace
    text = ' '.join(text.split())
    
    # 6. Remove leading/trailing whitespace
    text = text.strip()
    
    return text

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
    tweets_collection = mongo_db[MONGO_COLL]
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

# --- Create tables in PostgreSQL ---
# Table for aggregated analytics
create_analytics_table_sql = """
CREATE TABLE IF NOT EXISTS tweet_analytics (
    id SERIAL PRIMARY KEY,
    campaign VARCHAR(255),
    date DATE,
    avg_sentiment FLOAT,
    tweet_count INT,
    pos_count INT,
    neu_count INT,
    neg_count INT,
    UNIQUE(campaign, date)
);
"""

# Table for individual cleaned tweets
create_tweets_table_sql = """
CREATE TABLE IF NOT EXISTS cleaned_tweets (
    id SERIAL PRIMARY KEY,
    tweet_id VARCHAR(255) UNIQUE,
    campaign VARCHAR(255),
    original_text TEXT,
    cleaned_text TEXT,
    sentiment_label VARCHAR(20),
    sentiment_score FLOAT,
    created_at TIMESTAMP,
    user_id VARCHAR(255),
    hashtags TEXT[],
    created_date DATE,
    INDEX idx_campaign_date (campaign, created_date),
    INDEX idx_sentiment (sentiment_label),
    INDEX idx_created (created_at)
);
"""

try:
    with engine.connect() as conn:
        conn.execute(text(create_analytics_table_sql))
        conn.execute(text(create_tweets_table_sql))
        conn.commit()
    logger.info("PostgreSQL tables ready")
except Exception as e:
    logger.error(f"Failed to create tables: {e}")
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
cleaned_examples = []  # Store some examples of cleaning for logging

for tweet in tweets_cursor:
    try:
        text = tweet.get('text', '')
        if not text:
            continue
        
        # Clean the text for sentiment analysis
        cleaned_text = clean_text_for_sentiment(text)
        
        # Skip if cleaning removed all content
        if not cleaned_text or len(cleaned_text) < 5:
            logger.warning(f"Tweet {tweet.get('_id')} became too short after cleaning, skipping")
            continue
        
        # Log some examples of cleaning (first 5)
        if len(cleaned_examples) < 5 and text != cleaned_text:
            cleaned_examples.append({
                'original': text[:100] + '...' if len(text) > 100 else text,
                'cleaned': cleaned_text[:100] + '...' if len(cleaned_text) > 100 else cleaned_text
            })
        
        # Score CLEANED tweet with RoBERTa model (truncate at 512 chars)
        result = sentiment_analyzer(cleaned_text[:512])[0]
        
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
        
        # Update MongoDB with ONLY sentiment results (NOT cleaned text)
        tweets_collection.update_one(
            {"_id": tweet["_id"]},
            {"$set": {
                "sentiment_label": sentiment_label,
                "sentiment_score": float(result['score'])
                # NOT storing cleaned_text in MongoDB - keeping it raw!
            }}
        )
        
        # Add all data to tweet document for PostgreSQL storage
        tweet['sentiment_label'] = sentiment_label
        tweet['sentiment_score'] = float(result['score'])
        tweet['cleaned_text'] = cleaned_text  # Will be stored in PostgreSQL
        tweet_docs.append(tweet)
        
        processed_count += 1
        if processed_count % 100 == 0:
            logger.info(f"Processed {processed_count} tweets")
            
    except Exception as e:
        logger.error(f"Error scoring tweet {tweet.get('_id')}: {e}")

logger.info(f"Labeled {processed_count} new tweets")

# Log some cleaning examples
if cleaned_examples:
    logger.info("Text cleaning examples:")
    for i, example in enumerate(cleaned_examples, 1):
        logger.info(f"Example {i}:")
        logger.info(f"  Original: {example['original']}")
        logger.info(f"  Cleaned:  {example['cleaned']}")

# Also fetch already-scored tweets in the 3-month window
already_scored_cursor = tweets_collection.find({
    "created_at": {"$gte": start_date.isoformat()},
    "sentiment_label": {"$exists": True}
})

# Process already-scored tweets to add cleaned text
for tweet in already_scored_cursor:
    # Clean the text for consistency
    original_text = tweet.get('text', '')
    if original_text:
        tweet['cleaned_text'] = clean_text_for_sentiment(original_text)
    else:
        tweet['cleaned_text'] = ''
    tweet_docs.append(tweet)

if not tweet_docs:
    logger.warning("No tweets found in time range.")
    exit()

logger.info(f"Total tweets to process: {len(tweet_docs)}")

# Create dataframe
df = pd.DataFrame(tweet_docs)

# --- Data quality checks ---
# Remove tweets with very low quality
df = df[df['cleaned_text'].str.len() > 10]
logger.info(f"Tweets after quality filter: {len(df)}")

# Remove duplicate tweets (based on original text)
original_count = len(df)
df = df.drop_duplicates(subset=['text'], keep='first')
if original_count > len(df):
    logger.info(f"Removed {original_count - len(df)} duplicate tweets")

# --- Prepare data for PostgreSQL ---
# Extract campaign from matched_term
df['campaign'] = df['matched_term'].fillna('unknown')
df['created_date'] = pd.to_datetime(df['created_at']).dt.date

# Handle missing sentiment data
df['sentiment_label'] = df['sentiment_label'].fillna('UNKNOWN')
df['sentiment_score'] = df['sentiment_score'].fillna(0.0)

# --- Store individual cleaned tweets in PostgreSQL ---
logger.info("Storing cleaned tweets in PostgreSQL...")
try:
    # Prepare tweets dataframe for PostgreSQL
    tweets_for_postgres = df[[
        'tweet_id', 'campaign', 'text', 'cleaned_text',
        'sentiment_label', 'sentiment_score', 'created_at',
        'user_id', 'hashtags'
    ]].copy()
    
    # Rename columns for PostgreSQL
    tweets_for_postgres.columns = [
        'tweet_id', 'campaign', 'original_text', 'cleaned_text',
        'sentiment_label', 'sentiment_score', 'created_at',
        'user_id', 'hashtags'
    ]
    
    # Convert hashtags list to PostgreSQL array format
    tweets_for_postgres['hashtags'] = tweets_for_postgres['hashtags'].apply(
        lambda x: x if isinstance(x, list) else []
    )
    
    # Add created_date for indexing
    tweets_for_postgres['created_date'] = pd.to_datetime(tweets_for_postgres['created_at']).dt.date
    
    # Delete existing tweets for this date range to avoid duplicates
    with engine.connect() as conn:
        delete_tweets_sql = text("""
            DELETE FROM cleaned_tweets 
            WHERE created_date >= :start_date
        """)
        conn.execute(delete_tweets_sql, {"start_date": start_date.date()})
        conn.commit()
    
    # Insert cleaned tweets
    tweets_for_postgres.to_sql('cleaned_tweets', engine, if_exists='append', index=False)
    logger.info(f"Stored {len(tweets_for_postgres)} cleaned tweets in PostgreSQL")
    
except Exception as e:
    logger.error(f"Failed to store cleaned tweets: {e}")
    # Continue with aggregation even if individual tweets fail

# --- Aggregate by campaign and date for analytics ---
agg = df.groupby(['campaign', 'created_date']).agg(
    avg_sentiment=('sentiment_score', 'mean'),
    tweet_count=('tweet_id', 'count'),
    pos_count=('sentiment_label', lambda x: (x == 'POSITIVE').sum()),
    neu_count=('sentiment_label', lambda x: (x == 'NEUTRAL').sum()),
    neg_count=('sentiment_label', lambda x: (x == 'NEGATIVE').sum())
).reset_index()

# Rename created_date to date for analytics table
agg.rename(columns={'created_date': 'date'}, inplace=True)

logger.info(f"Aggregated data into {len(agg)} campaign-date combinations")

# --- Write aggregates to PostgreSQL ---
try:
    # Delete existing aggregates for the date range
    with engine.connect() as conn:
        delete_analytics_sql = text("""
            DELETE FROM tweet_analytics 
            WHERE date >= :start_date
        """)
        conn.execute(delete_analytics_sql, {"start_date": start_date.date()})
        conn.commit()
    
    # Insert new aggregated data
    agg.to_sql('tweet_analytics', engine, if_exists='append', index=False)
    logger.info(f"Successfully wrote {len(agg)} rows to tweet_analytics table")
    
except Exception as e:
    logger.error(f"Failed to write aggregates to PostgreSQL: {e}")
    exit(1)

# Close connections
mongo_client.close()
engine.dispose()

logger.info("ETL completed successfully!")
logger.info(f"- Processed {processed_count} new tweets")
logger.info(f"- Stored {len(tweets_for_postgres)} cleaned tweets in PostgreSQL")
logger.info(f"- Created {len(agg)} aggregated analytics records")