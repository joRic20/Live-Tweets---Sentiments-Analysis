# etl.py
# FIXED ETL script - Process ALL tweets without date restriction
# Uses REAL user data from MongoDB (follower counts, etc.)
#
# Sentiment Scoring System:
# - Uses cardiffnlp/twitter-roberta-base-sentiment-latest model
# - Model outputs probabilities for 3 classes: negative, neutral, positive
# - Sentiment Score: Continuous value from -1 to +1
#   * -1.0 = Completely negative
#   *  0.0 = Neutral
#   * +1.0 = Completely positive
#   * Calculated as: P(positive) - P(negative)
# - Sentiment Label: Categorical (POSITIVE, NEUTRAL, NEGATIVE)
#   * Based on probability thresholds
#   * NEUTRAL if P(neutral) > 0.5
#   * Otherwise, highest probability class with minimum threshold of 0.4
# - Confidence Score: Maximum probability among all classes (0-1)

from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
from pymongo import MongoClient
from sqlalchemy import create_engine, text
from transformers import pipeline
from datetime import datetime, timedelta
import logging
import re
import traceback
import requests
import pycountry
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track processing time for performance monitoring
start_time = time.time()

def clean_env_var(value):
    """Remove inline comments and extra quotes from environment variables"""
    if value:
        if '#' in value:
            value = value.split('#')[0]
        value = value.strip().strip('"\'')
    return value

def clean_text_for_sentiment(text):
    """Clean tweet text for more accurate sentiment analysis"""
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove RT prefix
    text = re.sub(r'^RT\s*@\w+\s*:\s*', '', text, flags=re.IGNORECASE)
    # Handle excessive mentions
    if len(re.findall(r'@\w+', text)) >= 3:
        text = re.sub(r'(@\w+\s*)+', '@users ', text)
    # Clean excessive hashtags
    hashtags = re.findall(r'#\w+', text)
    if len(hashtags) > 5:
        text = re.sub(r'#\w+', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split()).strip()
    
    return text

def extract_location_info(tweet_data):
    """Extract location information from tweet data"""
    location_info = {
        'country': None,
        'country_code': None,
        'region': None,
        'city': None
    }
    
    # Try to extract from various location fields
    if 'geo' in tweet_data and tweet_data['geo']:
        # Handle coordinate-based location
        pass  # Would need geocoding service
    
    if 'place' in tweet_data and tweet_data['place']:
        place = tweet_data['place']
        if 'country' in place:
            location_info['country'] = place['country']
        if 'country_code' in place:
            location_info['country_code'] = place['country_code']
        if 'name' in place:
            location_info['city'] = place['name']
    
    # Try to extract from user location string
    user_location = tweet_data.get('user_location', '')
    
    # Also check in user object if available
    if not user_location:
        if 'user' in tweet_data and tweet_data['user']:
            user_location = tweet_data['user'].get('location', '')
        elif 'author' in tweet_data and tweet_data['author']:
            user_location = tweet_data['author'].get('location', '')
    
    if user_location and not location_info['country']:
        location_info.update(parse_location_string(user_location))
    
    return location_info

def parse_location_string(location_str):
    """Parse location string to extract country information"""
    location_info = {'country': None, 'country_code': None}
    
    if not location_str:
        return location_info
    
    # Common country mappings
    country_mappings = {
        'usa': 'United States',
        'us': 'United States', 
        'america': 'United States',
        'uk': 'United Kingdom',
        'england': 'United Kingdom',
        'britain': 'United Kingdom',
        'canada': 'Canada',
        'australia': 'Australia',
        'germany': 'Germany',
        'france': 'France',
        'spain': 'Spain',
        'italy': 'Italy',
        'japan': 'Japan',
        'china': 'China',
        'india': 'India',
        'brazil': 'Brazil',
        'mexico': 'Mexico'
    }
    
    location_lower = location_str.lower()
    
    for key, country in country_mappings.items():
        if key in location_lower:
            location_info['country'] = country
            try:
                country_obj = pycountry.countries.lookup(country)
                location_info['country_code'] = country_obj.alpha_2
            except:
                pass
            break
    
    return location_info

def get_user_metrics_from_tweet(tweet_data):
    """Extract real user metrics from tweet data in MongoDB"""
    user_metrics = {
        'follower_count': 0,
        'following_count': 0,
        'tweet_count': 0,
        'verified': False
    }
    
    if not tweet_data:
        return user_metrics
    
    # Try different possible locations for user data in the tweet document
    user_data = None
    
    # Check common Twitter API response structures
    if 'user' in tweet_data and tweet_data['user']:
        user_data = tweet_data['user']
    elif 'author' in tweet_data and tweet_data['author']:
        user_data = tweet_data['author']
    elif 'includes' in tweet_data and 'users' in tweet_data['includes']:
        # Twitter API v2 structure
        users = tweet_data['includes']['users']
        if users and len(users) > 0:
            user_data = users[0]
    
    # Extract metrics from user data
    if user_data and isinstance(user_data, dict):
        # Common field names for follower count (handle both int and string)
        follower_count = (
            user_data.get('followers_count', 0) or
            user_data.get('public_metrics', {}).get('followers_count', 0) or
            user_data.get('follower_count', 0) or
            0
        )
        user_metrics['follower_count'] = int(follower_count) if follower_count else 0
        
        # Common field names for following/friends count
        following_count = (
            user_data.get('friends_count', 0) or
            user_data.get('public_metrics', {}).get('following_count', 0) or
            user_data.get('following_count', 0) or
            0
        )
        user_metrics['following_count'] = int(following_count) if following_count else 0
        
        # Common field names for tweet count
        tweet_count = (
            user_data.get('statuses_count', 0) or
            user_data.get('public_metrics', {}).get('tweet_count', 0) or
            user_data.get('tweet_count', 0) or
            0
        )
        user_metrics['tweet_count'] = int(tweet_count) if tweet_count else 0
        
        # Verified status
        user_metrics['verified'] = bool(
            user_data.get('verified', False) or
            user_data.get('verified_type', None) is not None
        )
    else:
        # Log if we couldn't find user data
        logger.debug(f"No user data found in tweet document")
    
    return user_metrics

# Load environment variables
MONGO_URI = clean_env_var(os.environ.get("MONGO_URI"))
MONGO_DB = clean_env_var(os.environ.get("MONGO_DB"))
MONGO_COLL = clean_env_var(os.environ.get("MONGO_COLL"))
POSTGRES_URI = clean_env_var(os.environ.get("POSTGRES_URI"))

logger.info(f"MongoDB Database: {MONGO_DB}")
logger.info(f"MongoDB Collection: {MONGO_COLL}")

# Validate environment variables
if not all([MONGO_URI, MONGO_DB, MONGO_COLL, POSTGRES_URI]):
    logger.error("Missing required environment variables!")
    exit(1)

# Connect to MongoDB
try:
    mongo_client = MongoClient(MONGO_URI)
    mongo_client.server_info()
    mongo_db = mongo_client[MONGO_DB]
    tweets_collection = mongo_db[MONGO_COLL]
    logger.info(f"Connected to MongoDB")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    exit(1)

# Connect to PostgreSQL
try:
    engine = create_engine(POSTGRES_URI)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Connected to PostgreSQL")
except Exception as e:
    logger.error(f"Failed to connect to PostgreSQL: {e}")
    exit(1)

# Load sentiment analysis model
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

# Create enhanced database tables with proper constraints
create_analytics_table_sql = """
CREATE TABLE IF NOT EXISTS tweet_analytics (
    id SERIAL PRIMARY KEY,
    campaign VARCHAR(255),
    date DATE,
    avg_sentiment FLOAT,  -- Average sentiment score (-1 to +1)
    tweet_count INT,
    pos_count INT,
    neu_count INT,
    neg_count INT,
    UNIQUE(campaign, date)
);
"""

create_hourly_analytics_table_sql = """
CREATE TABLE IF NOT EXISTS tweet_analytics_hourly (
    id SERIAL PRIMARY KEY,
    campaign VARCHAR(255),
    datetime TIMESTAMP,
    hour INT,
    avg_sentiment FLOAT,  -- Average sentiment score (-1 to +1)
    tweet_count INT,
    pos_count INT,
    neu_count INT,
    neg_count INT,
    UNIQUE(campaign, datetime)
);
"""

create_tweets_table_sql = """
CREATE TABLE IF NOT EXISTS cleaned_tweets (
    id SERIAL PRIMARY KEY,
    tweet_id VARCHAR(255) UNIQUE,
    campaign VARCHAR(255),
    original_text TEXT,
    cleaned_text TEXT,
    sentiment_label VARCHAR(20),
    sentiment_score FLOAT,  -- Range: -1 (negative) to +1 (positive)
    confidence_score FLOAT,  -- Range: 0 to 1
    created_at TIMESTAMP,
    user_id VARCHAR(255),
    username VARCHAR(255),
    hashtags TEXT[],
    country VARCHAR(100),
    country_code VARCHAR(2),
    region VARCHAR(100),
    city VARCHAR(100),
    created_date DATE,
    created_hour TIMESTAMP
);
"""

create_user_metrics_table_sql = """
CREATE TABLE IF NOT EXISTS user_metrics (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) UNIQUE,
    username VARCHAR(255),
    follower_count INT,
    following_count INT,
    tweet_count INT,
    verified BOOLEAN,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

create_indexes_sql = """
CREATE INDEX IF NOT EXISTS idx_campaign_date ON cleaned_tweets (campaign, created_date);
CREATE INDEX IF NOT EXISTS idx_campaign_hour ON cleaned_tweets (campaign, created_hour);
CREATE INDEX IF NOT EXISTS idx_sentiment ON cleaned_tweets (sentiment_label);
CREATE INDEX IF NOT EXISTS idx_created ON cleaned_tweets (created_at);
CREATE INDEX IF NOT EXISTS idx_user_id ON cleaned_tweets (user_id);
CREATE INDEX IF NOT EXISTS idx_country ON cleaned_tweets (country);
CREATE INDEX IF NOT EXISTS idx_user_metrics_user_id ON user_metrics (user_id);
CREATE INDEX IF NOT EXISTS idx_user_metrics_followers ON user_metrics (follower_count);
CREATE INDEX IF NOT EXISTS idx_hourly_campaign_datetime ON tweet_analytics_hourly (campaign, datetime);
"""

try:
    with engine.connect() as conn:
        conn.execute(text(create_analytics_table_sql))
        conn.execute(text(create_hourly_analytics_table_sql))
        conn.execute(text(create_tweets_table_sql))
        conn.execute(text(create_user_metrics_table_sql))
        conn.execute(text(create_indexes_sql))
        conn.commit()
    logger.info("PostgreSQL tables ready")
except Exception as e:
    logger.error(f"Failed to create tables: {e}")
    exit(1)

# UPDATED: Process ALL tweets without date restriction
logger.info(f"Processing ALL tweets from MongoDB (no date restrictions)")

# Count ALL unprocessed tweets in MongoDB
total_tweets_in_mongo = tweets_collection.count_documents({})
logger.info(f"Total tweets in MongoDB: {total_tweets_in_mongo}")

# Count tweets to process (without sentiment labels)
count_missing = tweets_collection.count_documents({
    "sentiment_label": {"$exists": False}
})
logger.info(f"Found {count_missing} tweets without sentiment labels")

# FIXED: Process in batches to handle large volumes
BATCH_SIZE = 5000
tweet_docs = []
processed_count = 0
user_metrics_cache = {}

# Process unscored tweets in batches
total_batches = (count_missing // BATCH_SIZE) + 1
logger.info(f"Processing {count_missing} tweets in {total_batches} batches")

# Debug: Log structure of first tweet to understand data format
if count_missing > 0:
    sample_tweet = tweets_collection.find_one({"sentiment_label": {"$exists": False}})
    if sample_tweet:
        logger.info("Sample tweet structure (keys): " + ", ".join(sample_tweet.keys()))
        if 'user' in sample_tweet and sample_tweet['user']:
            logger.info("User data keys: " + ", ".join(sample_tweet['user'].keys()))
            logger.info(f"Sample follower count: {sample_tweet['user'].get('followers_count', 'NOT FOUND')}")
        elif 'author' in sample_tweet and sample_tweet['author']:
            logger.info("Author data keys: " + ", ".join(sample_tweet['author'].keys()))
            logger.info(f"Sample follower count: {sample_tweet['author'].get('public_metrics', {}).get('followers_count', 'NOT FOUND')}")
        else:
            logger.warning("No user/author data found in tweet document!")

for batch_num in range(total_batches):
    skip_count = batch_num * BATCH_SIZE
    
    # Get batch of unprocessed tweets (no date filter)
    tweets_cursor = tweets_collection.find({
        "sentiment_label": {"$exists": False}
    }).skip(skip_count).limit(BATCH_SIZE)
    
    batch_tweets = list(tweets_cursor)
    if not batch_tweets:
        break
        
    logger.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_tweets)} tweets)")
    
    for tweet in batch_tweets:
        try:
            text = tweet.get('text', '')
            if not text:
                continue
            
            # Clean the text for sentiment analysis
            cleaned_text = clean_text_for_sentiment(text)
            
            if not cleaned_text or len(cleaned_text) < 5:
                continue
            
            # Score CLEANED tweet with RoBERTa model
            # Get full probability distribution for all classes
            results = sentiment_analyzer(cleaned_text[:512], top_k=None)
            
            # Initialize probabilities
            probs = {'negative': 0.0, 'neutral': 0.0, 'positive': 0.0}
            
            # Extract probabilities for each class
            if isinstance(results, list):
                # Multiple results returned
                for result in results:
                    label = result['label'].lower()
                    if label == 'label_0' or label == 'negative':
                        probs['negative'] = float(result['score'])
                    elif label == 'label_1' or label == 'neutral':
                        probs['neutral'] = float(result['score'])
                    elif label == 'label_2' or label == 'positive':
                        probs['positive'] = float(result['score'])
            else:
                # Single result - need to run again for all probabilities
                results = sentiment_analyzer(cleaned_text[:512], top_k=3)
                for result in results:
                    label = result['label'].lower()
                    if label == 'label_0' or label == 'negative':
                        probs['negative'] = float(result['score'])
                    elif label == 'label_1' or label == 'neutral':
                        probs['neutral'] = float(result['score'])
                    elif label == 'label_2' or label == 'positive':
                        probs['positive'] = float(result['score'])
            
            # Calculate weighted sentiment score (-1 to +1)
            # This gives a continuous score where:
            # -1 = completely negative, 0 = neutral, +1 = completely positive
            sentiment_score = probs['positive'] - probs['negative']
            
            # Determine sentiment label based on probabilities
            # Using thresholds to handle cases where probabilities are close
            max_prob = max(probs.values())
            confidence_score = max_prob
            
            # Determine label with some threshold logic
            if probs['neutral'] > 0.5:
                sentiment_label = 'NEUTRAL'
            elif probs['positive'] > probs['negative'] and probs['positive'] > 0.4:
                sentiment_label = 'POSITIVE'
            elif probs['negative'] > probs['positive'] and probs['negative'] > 0.4:
                sentiment_label = 'NEGATIVE'
            else:
                # If no clear winner, check which is highest
                if probs['positive'] == max_prob:
                    sentiment_label = 'POSITIVE'
                elif probs['negative'] == max_prob:
                    sentiment_label = 'NEGATIVE'
                else:
                    sentiment_label = 'NEUTRAL'
            
            # Update MongoDB with sentiment results
            tweets_collection.update_one(
                {"_id": tweet["_id"]},
                {"$set": {
                    "sentiment_label": sentiment_label,
                    "sentiment_score": sentiment_score,
                    "confidence_score": confidence_score,
                    "sentiment_probs": probs  # Store full probability distribution
                }}
            )
            
            # Extract location information
            location_info = extract_location_info(tweet)
            
            # Get user metrics from the actual tweet data
            # Extract user_id from various possible locations
            user_id = str(tweet.get('user_id', ''))
            if not user_id and 'user' in tweet and tweet['user']:
                user_id = str(tweet['user'].get('id', '') or tweet['user'].get('id_str', ''))
            elif not user_id and 'author_id' in tweet:
                user_id = str(tweet['author_id'])
            
            username = tweet.get('username', '')
            
            # Also try to get username from user object if not at top level
            if not username and 'user' in tweet and tweet['user']:
                username = tweet['user'].get('screen_name', '') or tweet['user'].get('username', '')
            elif not username and 'author' in tweet and tweet['author']:
                username = tweet['author'].get('username', '')
            
            if user_id and user_id not in user_metrics_cache:
                user_metrics_cache[user_id] = get_user_metrics_from_tweet(tweet)
            
            # Enhance tweet document
            tweet['sentiment_label'] = sentiment_label
            tweet['sentiment_score'] = sentiment_score
            tweet['confidence_score'] = confidence_score
            tweet['cleaned_text'] = cleaned_text
            tweet['country'] = location_info.get('country')
            tweet['country_code'] = location_info.get('country_code')
            tweet['region'] = location_info.get('region')
            tweet['city'] = location_info.get('city')
            tweet['username'] = username
            tweet['user_id'] = user_id  # Ensure user_id is stored
            
            tweet_docs.append(tweet)
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error scoring tweet {tweet.get('_id')}: {e}")

logger.info(f"Labeled {processed_count} new tweets")

# Log sentiment distribution of newly processed tweets
if processed_count > 0:
    sentiment_counts = {}
    for tweet in tweet_docs[:processed_count]:
        label = tweet.get('sentiment_label', 'UNKNOWN')
        sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
    
    logger.info(f"Sentiment distribution of new tweets:")
    for label, count in sentiment_counts.items():
        percentage = (count / processed_count) * 100
        logger.info(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Log sample of user metrics found
    sample_users = list(user_metrics_cache.items())[:5]
    logger.info(f"Sample user metrics extracted (first 5):")
    for user_id, metrics in sample_users:
        logger.info(f"  User {user_id}: {metrics['follower_count']:,} followers")

# UPDATED: Fetch ALL already-scored tweets (no date filter)
already_scored_count = tweets_collection.count_documents({
    "sentiment_label": {"$exists": True}
})

logger.info(f"Found {already_scored_count} already scored tweets")

# Process already scored tweets in batches
for batch_num in range((already_scored_count // BATCH_SIZE) + 1):
    skip_count = batch_num * BATCH_SIZE
    
    already_scored_cursor = tweets_collection.find({
        "sentiment_label": {"$exists": True}
    }).skip(skip_count).limit(BATCH_SIZE)
    
    batch_tweets = list(already_scored_cursor)
    if not batch_tweets:
        break
        
    logger.info(f"Processing already scored batch {batch_num + 1} ({len(batch_tweets)} tweets)")
    
    for tweet in batch_tweets:
        original_text = tweet.get('text', '')
        if original_text:
            tweet['cleaned_text'] = clean_text_for_sentiment(original_text)
            
            # Fix existing sentiment scores if needed
            existing_score = tweet.get('sentiment_score', 0)
            sentiment_label = tweet.get('sentiment_label', 'NEUTRAL')
            
            # Check if score needs recalculation (old format was 0-1 range)
            if 'sentiment_score' in tweet and -1 <= existing_score <= 1:
                # Score is already in correct range (-1 to +1)
                tweet['sentiment_score'] = existing_score
            else:
                # Old format or missing score - recalculate if possible
                if 'confidence_score' in tweet:
                    conf = tweet['confidence_score']
                    if sentiment_label == 'POSITIVE':
                        tweet['sentiment_score'] = conf  # Maps to 0 to +1
                    elif sentiment_label == 'NEGATIVE':
                        tweet['sentiment_score'] = -conf  # Maps to -1 to 0
                    else:  # NEUTRAL
                        tweet['sentiment_score'] = 0.0  # Neutral is 0
                else:
                    # Default scores based on label only
                    if sentiment_label == 'POSITIVE':
                        tweet['sentiment_score'] = 0.6
                    elif sentiment_label == 'NEGATIVE':
                        tweet['sentiment_score'] = -0.6
                    else:
                        tweet['sentiment_score'] = 0.0
                    tweet['confidence_score'] = 0.6  # Default confidence
        else:
            tweet['cleaned_text'] = ''
        
        # Extract location and user data
        location_info = extract_location_info(tweet)
        tweet['country'] = location_info.get('country')
        tweet['country_code'] = location_info.get('country_code')
        
        # Extract user_id from various possible locations
        user_id = str(tweet.get('user_id', ''))
        if not user_id and 'user' in tweet and tweet['user']:
            user_id = str(tweet['user'].get('id', '') or tweet['user'].get('id_str', ''))
        elif not user_id and 'author_id' in tweet:
            user_id = str(tweet['author_id'])
            
        username = tweet.get('username', '')
        
        # Also try to get username from user object if not at top level
        if not username and 'user' in tweet and tweet['user']:
            username = tweet['user'].get('screen_name', '') or tweet['user'].get('username', '')
        elif not username and 'author' in tweet and tweet['author']:
            username = tweet['author'].get('username', '')
            
        if user_id and user_id not in user_metrics_cache:
            user_metrics_cache[user_id] = get_user_metrics_from_tweet(tweet)
        tweet['username'] = username
        
        tweet_docs.append(tweet)

if not tweet_docs:
    logger.warning("No tweets found to process.")
    mongo_client.close()
    engine.dispose()
    exit(0)

logger.info(f"Total tweets to process: {len(tweet_docs)}")

# Create dataframe
df = pd.DataFrame(tweet_docs)

# Data quality checks
if 'cleaned_text' not in df.columns:
    df['cleaned_text'] = df['text'].apply(clean_text_for_sentiment)

df = df[df['cleaned_text'].str.len() > 10]
logger.info(f"Tweets after quality filter: {len(df)}")

# Validate sentiment scores are in correct range
if 'sentiment_score' in df.columns:
    out_of_range = df[(df['sentiment_score'] < -1) | (df['sentiment_score'] > 1)]
    if len(out_of_range) > 0:
        logger.warning(f"Found {len(out_of_range)} tweets with sentiment scores outside [-1, 1] range")
        df.loc[(df['sentiment_score'] < -1) | (df['sentiment_score'] > 1), 'sentiment_score'] = 0.0
    
    # Log sentiment score distribution
    logger.info(f"Sentiment score statistics:")
    logger.info(f"  Mean: {df['sentiment_score'].mean():.3f}")
    logger.info(f"  Std: {df['sentiment_score'].std():.3f}")
    logger.info(f"  Min: {df['sentiment_score'].min():.3f}")
    logger.info(f"  Max: {df['sentiment_score'].max():.3f}")

# Remove duplicates
original_count = len(df)
df = df.drop_duplicates(subset=['text'], keep='first')
if original_count > len(df):
    logger.info(f"Removed {original_count - len(df)} duplicate tweets")

# Prepare data for PostgreSQL
df['campaign'] = df['matched_term'].fillna('unknown')

# Handle created_at conversion with HOURLY buckets
if 'created_at' in df.columns:
    if df['created_at'].dtype == 'object':
        df['created_at'] = pd.to_datetime(df['created_at'])
    df['created_date'] = df['created_at'].dt.date
    df['created_hour'] = df['created_at'].dt.floor('H')
else:
    df['created_date'] = datetime.now().date()
    df['created_hour'] = datetime.now().replace(minute=0, second=0, microsecond=0)

# Handle missing sentiment data
df['sentiment_label'] = df['sentiment_label'].fillna('UNKNOWN')
df['sentiment_score'] = df['sentiment_score'].fillna(0.0)

# FIXED: Store user metrics using UPSERT
logger.info("Storing user metrics...")
try:
    if user_metrics_cache:
        stored_count = 0
        for user_id, metrics in user_metrics_cache.items():
            if user_id and metrics['follower_count'] > 0:  # Only store users with real data
                # Find username from tweets
                username = ''
                user_tweets = df[df['user_id'] == user_id]
                if len(user_tweets) > 0:
                    username = user_tweets.iloc[0].get('username', '')
                    if not username:
                        # Try to get from original tweet data
                        for tweet in tweet_docs:
                            if str(tweet.get('user_id', '')) == user_id:
                                username = tweet.get('username', '') or tweet.get('screen_name', '') or user_id
                                break
                
                upsert_user_sql = text("""
                    INSERT INTO user_metrics (
                        user_id, username, follower_count, 
                        following_count, tweet_count, verified
                    ) VALUES (
                        :user_id, :username, :follower_count,
                        :following_count, :tweet_count, :verified
                    )
                    ON CONFLICT (user_id) DO UPDATE SET
                        follower_count = EXCLUDED.follower_count,
                        following_count = EXCLUDED.following_count,
                        tweet_count = EXCLUDED.tweet_count,
                        verified = EXCLUDED.verified,
                        username = EXCLUDED.username,
                        last_updated = CURRENT_TIMESTAMP
                """)
                
                with engine.connect() as conn:
                    conn.execute(upsert_user_sql, {
                        'user_id': user_id,
                        'username': username or user_id,
                        'follower_count': metrics['follower_count'],
                        'following_count': metrics['following_count'],
                        'tweet_count': metrics['tweet_count'],
                        'verified': metrics['verified']
                    })
                    conn.commit()
                    stored_count += 1
        
        logger.info(f"Stored/updated {stored_count} user metrics with real data")

except Exception as e:
    logger.error(f"Failed to store user metrics: {e}")

# FIXED: Store cleaned tweets using UPSERT instead of DELETE/INSERT
logger.info("Storing cleaned tweets in PostgreSQL...")
try:
    required_columns = ['tweet_id', 'campaign', 'text', 'cleaned_text',
                       'sentiment_label', 'sentiment_score', 'confidence_score', 'created_at',
                       'user_id', 'username', 'hashtags', 'country', 'country_code',
                       'created_hour']
    
    # Ensure tweet_id exists in df first
    if 'tweet_id' not in df.columns:
        if 'id' in df.columns:
            df['tweet_id'] = df['id'].astype(str)
        elif 'id_str' in df.columns:
            df['tweet_id'] = df['id_str']
        elif '_id' in df.columns:
            df['tweet_id'] = df['_id'].astype(str)
        else:
            logger.error("No tweet ID column found!")
            df['tweet_id'] = pd.Series(range(len(df))).astype(str)  # Fallback
    
    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"Missing columns: {missing_columns}")
        for col in missing_columns:
            if col == 'hashtags':
                # Try to extract hashtags from text if not present
                if 'text' in df.columns:
                    import re
                    df[col] = df['text'].apply(lambda t: re.findall(r'#\w+', str(t)) if t else [])
                else:
                    df[col] = [[] for _ in range(len(df))]
            elif col in ['country', 'country_code', 'username']:
                df[col] = None
            elif col == 'confidence_score':
                df[col] = df['sentiment_score'].abs()
            else:
                df[col] = None
    
    tweets_for_postgres = df[required_columns].copy()
    
    # Ensure we have tweet_id
    if 'tweet_id' not in tweets_for_postgres.columns:
        if 'id' in df.columns:
            tweets_for_postgres['tweet_id'] = df['id'].astype(str)
        elif 'id_str' in df.columns:
            tweets_for_postgres['tweet_id'] = df['id_str']
        elif '_id' in df.columns:
            tweets_for_postgres['tweet_id'] = df['_id'].astype(str)
        else:
            logger.error("No tweet ID column found!")
            tweets_for_postgres['tweet_id'] = pd.Series(range(len(df))).astype(str)  # Fallback
    
    # Process hashtags
    def process_hashtags(x):
        if isinstance(x, list):
            return x
        elif isinstance(x, str):
            # Extract hashtags from text if stored as string
            import re
            hashtags = re.findall(r'#\w+', x)
            return hashtags
        elif pd.isna(x):
            return []
        else:
            return []
    
    tweets_for_postgres['hashtags'] = tweets_for_postgres['hashtags'].apply(process_hashtags)
    
    # Ensure we have tweet_id
    if 'tweet_id' not in df.columns:
        if 'id' in df.columns:
            df['tweet_id'] = df['id'].astype(str)
        elif 'id_str' in df.columns:
            df['tweet_id'] = df['id_str']
        elif '_id' in df.columns:
            df['tweet_id'] = df['_id'].astype(str)
        else:
            logger.error("No tweet ID column found!")
            df['tweet_id'] = pd.Series(range(len(df))).astype(str)  # Fallback
        tweets_for_postgres['created_at'] = pd.to_datetime(tweets_for_postgres['created_at'])
    tweets_for_postgres['created_date'] = tweets_for_postgres['created_at'].dt.date
    
    # FIXED: Use UPSERT for each tweet
    inserted_count = 0
    updated_count = 0
    
    # Process in smaller chunks for better performance
    UPSERT_BATCH_SIZE = 1000
    total_rows = len(tweets_for_postgres)
    
    for chunk_start in range(0, total_rows, UPSERT_BATCH_SIZE):
        chunk_end = min(chunk_start + UPSERT_BATCH_SIZE, total_rows)
        chunk = tweets_for_postgres.iloc[chunk_start:chunk_end]
        
        logger.info(f"Upserting tweets {chunk_start + 1} to {chunk_end} of {total_rows}")
        
        for _, tweet_row in chunk.iterrows():
            try:
                upsert_sql = text("""
                    INSERT INTO cleaned_tweets (
                        tweet_id, campaign, original_text, cleaned_text,
                        sentiment_label, sentiment_score, confidence_score,
                        created_at, user_id, username, hashtags,
                        country, country_code, created_date, created_hour
                    ) VALUES (
                        :tweet_id, :campaign, :original_text, :cleaned_text,
                        :sentiment_label, :sentiment_score, :confidence_score,
                        :created_at, :user_id, :username, :hashtags,
                        :country, :country_code, :created_date, :created_hour
                    )
                    ON CONFLICT (tweet_id) DO UPDATE SET
                        sentiment_label = EXCLUDED.sentiment_label,
                        sentiment_score = EXCLUDED.sentiment_score,
                        confidence_score = EXCLUDED.confidence_score,
                        cleaned_text = EXCLUDED.cleaned_text,
                        country = EXCLUDED.country,
                        country_code = EXCLUDED.country_code,
                        username = EXCLUDED.username,
                        created_hour = EXCLUDED.created_hour
                    RETURNING (xmax = 0) AS inserted
                """)
                
                with engine.connect() as conn:
                    result = conn.execute(upsert_sql, {
                        'tweet_id': str(tweet_row['tweet_id']),
                        'campaign': str(tweet_row['campaign']),
                        'original_text': tweet_row['text'],
                        'cleaned_text': tweet_row['cleaned_text'],
                        'sentiment_label': tweet_row['sentiment_label'],
                        'sentiment_score': float(tweet_row['sentiment_score']),
                        'confidence_score': float(tweet_row['confidence_score']),
                        'created_at': tweet_row['created_at'],
                        'user_id': str(tweet_row['user_id']) if tweet_row['user_id'] else '',
                        'username': str(tweet_row['username']) if tweet_row['username'] else '',
                        'hashtags': tweet_row['hashtags'],
                        'country': tweet_row['country'],
                        'country_code': tweet_row['country_code'],
                        'created_date': tweet_row['created_date'],
                        'created_hour': tweet_row['created_hour']
                    })
                    
                    # Check if it was an insert or update
                    if result.fetchone()[0]:
                        inserted_count += 1
                    else:
                        updated_count += 1
                        
                    conn.commit()
                    
            except Exception as e:
                logger.error(f"Failed to upsert tweet {tweet_row.get('tweet_id')}: {e}")
    
    logger.info(f"Inserted {inserted_count} new tweets, updated {updated_count} existing tweets")
    
except Exception as e:
    logger.error(f"Failed to store cleaned tweets: {e}")
    logger.error(f"Error details: {traceback.format_exc()}")

# FIXED: Aggregate analytics data with proper updates
try:
    # Daily aggregation
    agg_daily = df.groupby(['campaign', 'created_date']).agg(
        avg_sentiment=('sentiment_score', 'mean'),
        tweet_count=('tweet_id', 'count'),
        pos_count=('sentiment_label', lambda x: (x == 'POSITIVE').sum()),
        neu_count=('sentiment_label', lambda x: (x == 'NEUTRAL').sum()),
        neg_count=('sentiment_label', lambda x: (x == 'NEGATIVE').sum())
    ).reset_index()
    
    agg_daily.rename(columns={'created_date': 'date'}, inplace=True)
    
    logger.info(f"Aggregated data into {len(agg_daily)} campaign-date combinations")
    
    # FIXED: Update only specific campaign-date combinations
    for _, row in agg_daily.iterrows():
        try:
            upsert_analytics_sql = text("""
                INSERT INTO tweet_analytics (
                    campaign, date, avg_sentiment, tweet_count,
                    pos_count, neu_count, neg_count
                ) VALUES (
                    :campaign, :date, :avg_sentiment, :tweet_count,
                    :pos_count, :neu_count, :neg_count
                )
                ON CONFLICT (campaign, date) DO UPDATE SET
                    avg_sentiment = EXCLUDED.avg_sentiment,
                    tweet_count = EXCLUDED.tweet_count,
                    pos_count = EXCLUDED.pos_count,
                    neu_count = EXCLUDED.neu_count,
                    neg_count = EXCLUDED.neg_count
            """)
            
            with engine.connect() as conn:
                conn.execute(upsert_analytics_sql, row.to_dict())
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update analytics for {row['campaign']} on {row['date']}: {e}")
    
    logger.info(f"Successfully updated {len(agg_daily)} daily analytics records")
    
except Exception as e:
    logger.error(f"Failed to aggregate daily data: {e}")

# FIXED: Hourly aggregation with proper updates
try:
    # Hourly aggregation
    agg_hourly = df.groupby(['campaign', 'created_hour']).agg(
        avg_sentiment=('sentiment_score', 'mean'),
        tweet_count=('tweet_id', 'count'),
        pos_count=('sentiment_label', lambda x: (x == 'POSITIVE').sum()),
        neu_count=('sentiment_label', lambda x: (x == 'NEUTRAL').sum()),
        neg_count=('sentiment_label', lambda x: (x == 'NEGATIVE').sum())
    ).reset_index()
    
    # Add hour column
    agg_hourly['hour'] = agg_hourly['created_hour'].dt.hour
    agg_hourly.rename(columns={'created_hour': 'datetime'}, inplace=True)
    
    logger.info(f"Aggregated data into {len(agg_hourly)} campaign-hour combinations")
    
    # Update hourly analytics
    for _, row in agg_hourly.iterrows():
        try:
            upsert_hourly_sql = text("""
                INSERT INTO tweet_analytics_hourly (
                    campaign, datetime, hour, avg_sentiment,
                    tweet_count, pos_count, neu_count, neg_count
                ) VALUES (
                    :campaign, :datetime, :hour, :avg_sentiment,
                    :tweet_count, :pos_count, :neu_count, :neg_count
                )
                ON CONFLICT (campaign, datetime) DO UPDATE SET
                    avg_sentiment = EXCLUDED.avg_sentiment,
                    tweet_count = EXCLUDED.tweet_count,
                    pos_count = EXCLUDED.pos_count,
                    neu_count = EXCLUDED.neu_count,
                    neg_count = EXCLUDED.neg_count
            """)
            
            with engine.connect() as conn:
                conn.execute(upsert_hourly_sql, row.to_dict())
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update hourly analytics: {e}")
    
    logger.info(f"Successfully updated {len(agg_hourly)} hourly analytics records")
    
except Exception as e:
    logger.error(f"Failed to aggregate hourly data: {e}")

# Close connections
mongo_client.close()
engine.dispose()

# Calculate total processing time
end_time = time.time()
processing_time = end_time - start_time

logger.info("ETL completed successfully!")
logger.info(f"- Processing time: {processing_time:.2f} seconds")
logger.info(f"- Processed {processed_count} new tweets")
logger.info(f"- Total tweets handled: {len(tweet_docs)}")
logger.info(f"- User metrics stored: {len(user_metrics_cache)}")

# Log how many users had real follower data
if user_metrics_cache:
    users_with_real_data = sum(1 for metrics in user_metrics_cache.values() if metrics['follower_count'] > 0)
    logger.info(f"- Users with real follower data: {users_with_real_data}/{len(user_metrics_cache)}")

logger.info(f"- Daily analytics updated: {len(agg_daily) if 'agg_daily' in locals() else 0}")
logger.info(f"- Hourly analytics updated: {len(agg_hourly) if 'agg_hourly' in locals() else 0}")