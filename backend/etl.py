from dotenv import load_dotenv
load_dotenv()  # Loads environment variables from .env

import os
import pandas as pd
from pymongo import MongoClient
from sqlalchemy import create_engine
from transformers import pipeline
from datetime import datetime, timedelta

# --- Load database URIs from .env ---
MONGO_URI = os.environ.get("MONGO_URI")           # MongoDB connection string
MONGO_DB = os.environ.get("MONGO_DB")             # MongoDB database name
MONGO_COLL = os.environ.get("MONGO_COLL")         # MongoDB collection name
POSTGRES_URI = os.environ.get("POSTGRES_URI")     # PostgreSQL connection string

# --- Connect to MongoDB ---
mongo_client = MongoClient(MONGO_URI)              # Create MongoDB client
mongo_db = mongo_client[MONGO_DB]                  # Select database
tweets_collection = mongo_db[MONGO_COLL]           # Select tweets collection

# --- HuggingFace RoBERTa sentiment pipeline ---
sentiment_analyzer = pipeline(
    "sentiment-analysis", 
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)  # Loads the RoBERTa model for sentiment analysis

# --- Find last 3 months of tweets missing sentiment, then label them ---
start_date = datetime.utcnow() - timedelta(days=90)    # Define 3-month window
tweets_cursor = tweets_collection.find({
    "created_at": {"$gte": start_date.isoformat()},    # Only tweets in time window
    "sentiment_label": {"$exists": False}              # Only those NOT already scored
})

tweet_docs = []
for tweet in tweets_cursor:
    try:
        text = tweet['text']                                       # Get tweet text
        # Score tweet with RoBERTa model (truncate at 512 chars for safety)
        result = sentiment_analyzer(text[:512])[0]
        # Update MongoDB with sentiment results
        tweets_collection.update_one(
            {"_id": tweet["_id"]},
            {"$set": {
                "sentiment_label": result['label'],
                "sentiment_score": float(result['score'])
            }}
        )
        tweet['sentiment_label'] = result['label']                 # Save locally too
        tweet['sentiment_score'] = float(result['score'])
        tweet_docs.append(tweet)
    except Exception as e:
        print(f"Error scoring tweet: {e}")                         # Error handling

# Also fetch already-scored tweets in the 3-month window
already_scored_cursor = tweets_collection.find({
    "created_at": {"$gte": start_date.isoformat()},
    "sentiment_label": {"$exists": True}
})
tweet_docs.extend(list(already_scored_cursor))                     # Add to list

if not tweet_docs:
    print("No tweets found in time range.")                        # Exit if nothing to process
    exit()

df = pd.DataFrame(tweet_docs)                                      # Create dataframe

# --- Extract campaign from matched_term for analytics (or improve as needed) ---
df['campaign'] = df['matched_term'].fillna('unknown')              # Campaign = matched_term or 'unknown'
df['date'] = pd.to_datetime(df['created_at']).dt.date              # Normalize to date

df['sentiment_label'] = df['sentiment_label'].fillna('UNKNOWN')    # Fill missing
df['sentiment_score'] = df['sentiment_score'].fillna(0.0)

# --- Aggregate by campaign, date, and sentiment label counts ---
agg = df.groupby(['campaign', 'date']).agg(
    avg_sentiment=('sentiment_score', 'mean'),                     # Mean sentiment
    tweet_count=('tweet_id', 'count'),                             # Number of tweets
    pos_count=('sentiment_label', lambda x: (x == 'POSITIVE').sum()),
    neu_count=('sentiment_label', lambda x: (x == 'NEUTRAL').sum()),
    neg_count=('sentiment_label', lambda x: (x == 'NEGATIVE').sum())
).reset_index()

# --- Connect to PostgreSQL ---
engine = create_engine(POSTGRES_URI)                               # Create SQLAlchemy engine

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
    neg_count INT
);
"""
with engine.connect() as conn:
    conn.execute(create_table_sql)                                 # Create table if missing

# --- Write aggregates to PostgreSQL ---
agg.to_sql('tweet_analytics', engine, if_exists='append', index=False)  # Append new aggregates

print("ETL with RoBERTa sentiment and full aggregation to PostgreSQL completed.")
