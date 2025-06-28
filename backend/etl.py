# etl_roberta_to_pg.py
# Reads tweets from MongoDB, applies RoBERTa sentiment, aggregates by campaign/date/label, writes to PostgreSQL.

from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
from pymongo import MongoClient
from sqlalchemy import create_engine
from transformers import pipeline
from datetime import datetime, timedelta

# --- Load database URIs from .env ---
MONGO_URI = os.environ.get("MONGO_URI")
MONGO_DB = os.environ.get("MONGO_DB")
MONGO_COLL = os.environ.get("MONGO_COLL")
POSTGRES_URI = os.environ.get("POSTGRES_URI")

# --- Connect to MongoDB ---
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB]
tweets_collection = mongo_db[MONGO_COLL]

# --- HuggingFace RoBERTa sentiment pipeline ---
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# --- Find last 3 months of tweets missing sentiment, then label them ---
start_date = datetime.utcnow() - timedelta(days=90)
tweets_cursor = tweets_collection.find({
    "created_at": {"$gte": start_date.isoformat()},
    "sentiment_label": {"$exists": False}
})

tweet_docs = []
for tweet in tweets_cursor:
    try:
        text = tweet['text']
        # Score tweet with RoBERTa model
        result = sentiment_analyzer(text[:512])[0]
        # Update MongoDB with sentiment
        tweets_collection.update_one(
            {"_id": tweet["_id"]},
            {"$set": {
                "sentiment_label": result['label'],
                "sentiment_score": float(result['score'])
            }}
        )
        tweet['sentiment_label'] = result['label']
        tweet['sentiment_score'] = float(result['score'])
        tweet_docs.append(tweet)
    except Exception as e:
        print(f"Error scoring tweet: {e}")

# Also fetch already-scored tweets in the 3-month window
already_scored_cursor = tweets_collection.find({
    "created_at": {"$gte": start_date.isoformat()},
    "sentiment_label": {"$exists": True}
})
tweet_docs.extend(list(already_scored_cursor))

if not tweet_docs:
    print("No tweets found in time range.")
    exit()

df = pd.DataFrame(tweet_docs)

# --- Extract campaign from matched_term for analytics (or improve as needed) ---
df['campaign'] = df['matched_term'].fillna('unknown')
df['date'] = pd.to_datetime(df['created_at']).dt.date

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

# --- Connect to PostgreSQL ---
engine = create_engine(POSTGRES_URI)

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
    conn.execute(create_table_sql)

# --- Write aggregates to PostgreSQL ---
agg.to_sql('tweet_analytics', engine, if_exists='append', index=False)

print("ETL with RoBERTa sentiment and full aggregation to PostgreSQL completed.")
