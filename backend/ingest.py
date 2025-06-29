# ingest.py
# Streams tweets for a user-specified search term into MongoDB, with a max tweet limit.

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env

import os
import tweepy
from pymongo import MongoClient, errors
from datetime import datetime
import logging
import sys
import time

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

# --- Load and clean secrets from .env ---
TWITTER_BEARER_TOKEN = clean_env_var(os.environ.get("TWITTER_BEARER_TOKEN"))
MONGO_URI = clean_env_var(os.environ.get("MONGO_URI"))
MONGO_DB = clean_env_var(os.environ.get("MONGO_DB"))
MONGO_COLL = clean_env_var(os.environ.get("MONGO_COLL"))

# --- Load maximum tweets limit (default: 100) ---
MAX_TWEETS = int(os.environ.get("MAX_TWEETS", "80"))
tweet_counter = 0  # Track number of tweets ingested

# --- Set up logging ---
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s", 
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ingest.log')
    ]
)
logger = logging.getLogger(__name__)

# Validate required environment variables
if not all([TWITTER_BEARER_TOKEN, MONGO_URI, MONGO_DB, MONGO_COLL]):
    logger.error("Missing required environment variables!")
    logger.error("Please ensure TWITTER_BEARER_TOKEN, MONGO_URI, MONGO_DB, and MONGO_COLL are set in .env")
    sys.exit(1)

# Log configuration (without exposing sensitive data)
logger.info(f"MongoDB Database: {MONGO_DB}")
logger.info(f"MongoDB Collection: {MONGO_COLL}")
logger.info(f"Max Tweets: {MAX_TWEETS}")
logger.info(f"Bearer Token: {'Set' if TWITTER_BEARER_TOKEN else 'Not Set'}")

# --- Accept product/service/campaign/brand name as CLI argument ---
if len(sys.argv) > 1:
    TRACK_TERM = sys.argv[1]
    logger.info(f"Tracking term: '{TRACK_TERM}'")
else:
    logger.error("Error: Please provide a product/service/campaign/brand name as a search term.")
    logger.error("Usage: python ingest.py <search_term>")
    sys.exit(1)

# --- Set up MongoDB client and collection ---
try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # Test connection
    mongo_client.server_info()
    mongo_db = mongo_client[MONGO_DB]
    tweets_collection = mongo_db[MONGO_COLL]
    logger.info(f"Connected to MongoDB - Database: {MONGO_DB}, Collection: {MONGO_COLL}")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    logger.error("Please check your MONGO_URI in the .env file")
    sys.exit(1)

# --- Ensure indexes are created ---
try:
    # Create unique index on tweet_id
    tweets_collection.create_index("tweet_id", unique=True)
    # Create TTL index for auto-deletion after 30 days
    tweets_collection.create_index("created_at", expireAfterSeconds=30 * 24 * 3600)
    # Create index on matched_term for faster queries
    tweets_collection.create_index("matched_term")
    logger.info("MongoDB indexes created/verified")
except Exception as e:
    logger.error(f"Failed to create indexes: {e}")

# --- Tweepy Streaming Listener class ---
class MongoTweetListener(tweepy.StreamingClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()
        self.error_count = 0
        self.max_errors = 10
    
    def on_tweet(self, tweet):
        global tweet_counter
        
        try:
            # Extract user data if available
            user_data = {}
            if hasattr(tweet, 'includes') and 'users' in tweet.includes:
                user = tweet.includes['users'][0]
                user_data = {
                    "user_id": user.id,
                    "username": user.username,
                    "user_name": user.name
                }
            
            # Prepare MongoDB document for each tweet
            tweet_doc = {
                "tweet_id": str(tweet.id),  # Store as string to avoid int64 issues
                "text": tweet.text,
                "author_id": str(tweet.author_id) if tweet.author_id else None,
                **user_data,  # Include user data if available
                "created_at": tweet.created_at.isoformat() if tweet.created_at else datetime.utcnow().isoformat(),
                "hashtags": [],
                "mentions": [],
                "urls": [],
                "lang": tweet.lang if hasattr(tweet, 'lang') else None,
                "matched_term": TRACK_TERM,
                "ingested_at": datetime.utcnow().isoformat(),
                "raw": tweet.data
            }
            
            # Extract entities if available
            if hasattr(tweet, 'entities') and tweet.entities:
                if 'hashtags' in tweet.entities:
                    tweet_doc['hashtags'] = [tag['tag'] for tag in tweet.entities['hashtags']]
                if 'mentions' in tweet.entities:
                    tweet_doc['mentions'] = [mention['username'] for mention in tweet.entities['mentions']]
                if 'urls' in tweet.entities:
                    tweet_doc['urls'] = [url['expanded_url'] for url in tweet.entities['urls']]
            
            # Store tweet in MongoDB
            tweets_collection.insert_one(tweet_doc)
            tweet_counter += 1
            
            # Log progress
            logger.info(f"Tweet {tweet_counter}/{MAX_TWEETS} stored: ID={tweet_doc['tweet_id']}, "
                       f"User=@{tweet_doc.get('username', 'unknown')}, "
                       f"Text={tweet.text[:50]}...")
            
            # Stop streaming if max limit is reached
            if tweet_counter >= MAX_TWEETS:
                elapsed_time = time.time() - self.start_time
                logger.info(f"Reached max tweet limit ({MAX_TWEETS} tweets) in {elapsed_time:.1f} seconds")
                logger.info("Disconnecting stream...")
                self.disconnect()
                
        except errors.DuplicateKeyError:
            logger.debug(f"Duplicate tweet: {tweet.id}")
        except Exception as e:
            logger.error(f"Error inserting tweet: {e}")
            self.error_count += 1
            if self.error_count > self.max_errors:
                logger.error(f"Too many errors ({self.error_count}), disconnecting...")
                self.disconnect()

    def on_connect(self):
        logger.info("✅ Connected to Twitter Streaming API")
        logger.info(f"Streaming tweets for: '{TRACK_TERM}'")
        logger.info(f"Will collect up to {MAX_TWEETS} tweets")

    def on_connection_error(self):
        logger.error("❌ Connection error. Will attempt to reconnect...")

    def on_disconnect(self):
        logger.warning("🔌 Disconnected from Twitter Stream")
        elapsed_time = time.time() - self.start_time
        logger.info(f"Total runtime: {elapsed_time:.1f} seconds")
        logger.info(f"Total tweets collected: {tweet_counter}")

    def on_errors(self, errors):
        logger.error(f"Stream errors: {errors}")

    def on_exception(self, exception):
        logger.error(f"Stream exception: {exception}")
        return False  # Don't suppress the exception

    def on_request_error(self, status_code):
        logger.error(f"Request error with status code: {status_code}")
        if status_code == 420:
            logger.error("Rate limited! Waiting before reconnect...")
            time.sleep(60)  # Wait 1 minute
        elif status_code == 401:
            logger.error("Authentication failed! Check your bearer token.")
            return False  # Stop trying
        elif status_code == 403:
            logger.error("Access forbidden! Check your API access level.")
            return False  # Stop trying
        return True  # Continue trying

# --- Main function to start the streaming client ---
def main():
    try:
        # Create streaming client
        stream = MongoTweetListener(
            bearer_token=TWITTER_BEARER_TOKEN, 
            wait_on_rate_limit=True,
            max_retries=3
        )
        
        # Remove all existing stream rules
        logger.info("Checking existing stream rules...")
        existing_rules = stream.get_rules()
        if existing_rules and existing_rules.data:
            rule_ids = [rule.id for rule in existing_rules.data]
            stream.delete_rules(rule_ids)
            logger.info(f"Cleared {len(rule_ids)} old stream rules")
        
        # Add a rule for the user-specified search term
        # Add language filter and exclude retweets for better quality
        rule_value = f'{TRACK_TERM} -is:retweet lang:en'
        stream.add_rules(tweepy.StreamRule(value=rule_value, tag=TRACK_TERM))
        logger.info(f"Added stream rule: {rule_value}")
        
        # Start streaming tweets with all available fields
        logger.info("Starting tweet stream...")
        stream.filter(
            tweet_fields=["created_at", "lang", "entities", "author_id", "conversation_id", "public_metrics"],
            user_fields=["username", "name", "verified", "description"],
            expansions=["author_id"]
        )
        
    except KeyboardInterrupt:
        logger.info("Stream interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Cleanup
        if 'mongo_client' in locals():
            mongo_client.close()
            logger.info("MongoDB connection closed")
        logger.info(f"Final tweet count: {tweet_counter}")

# --- Run main() if this script is executed directly ---
if __name__ == "__main__":
    main()