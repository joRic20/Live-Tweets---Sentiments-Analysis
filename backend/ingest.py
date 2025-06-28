# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import os
import tweepy
from pymongo import MongoClient, errors
from datetime import datetime
import logging
import sys

# ------------- SECRETS AND CONFIGS -------------
# Get credentials and config from environment variables
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")
MONGO_URI = os.environ.get("MONGO_URI")
MONGO_DB = os.environ.get("MONGO_DB")
MONGO_COLL = os.environ.get("MONGO_COLL")

# Set up logging so you can see what's happening
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

# MongoDB connection setup
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB]
tweets_collection = mongo_db[MONGO_COLL]

# Ensure we don't store duplicate tweets and that old tweets are deleted after 30 days
tweets_collection.create_index("tweet_id", unique=True)
tweets_collection.create_index("created_at", expireAfterSeconds=30 * 24 * 3600)

# Accept tracking terms (search keyword) as a command line argument from the backend
if len(sys.argv) > 1:
    TRACK_TERMS = [sys.argv[1]]  # User-supplied search term
else:
    TRACK_TERMS = ["#YourHashtag", "brand", "campaign"]  # Default terms

# Custom Tweepy streaming client to handle and store tweets
class MongoTweetListener(tweepy.StreamingClient):
    def on_tweet(self, tweet):
        # Prepare tweet data as a document for MongoDB
        tweet_doc = {
            "tweet_id": tweet.id,  # Unique tweet ID
            "text": tweet.text,  # The tweet's content
            "author_id": tweet.author_id,  # Twitter user who posted
            "created_at": tweet.created_at.isoformat() if tweet.created_at else datetime.utcnow().isoformat(),
            "hashtags": [tag['tag'] for tag in tweet.entities['hashtags']] if tweet.entities and 'hashtags' in tweet.entities else [],
            "lang": getattr(tweet, 'lang', None),  # Language if available
            "raw": tweet.data  # Full tweet JSON for reference
        }
        # Store in MongoDB
        try:
            tweets_collection.insert_one(tweet_doc)
            logging.info(f"Tweet stored: {tweet_doc['tweet_id']}")
        except errors.DuplicateKeyError:
            logging.debug(f"Duplicate tweet: {tweet_doc['tweet_id']}")
        except Exception as e:
            logging.error(f"Error inserting tweet: {e}")

    def on_connect(self):
        logging.info("Connected to Twitter Streaming API.")

    def on_connection_error(self):
        logging.error("Connection error. Reconnecting...")
        self.disconnect()

    def on_errors(self, errors):
        logging.error(f"Stream error: {errors}")

    def on_exception(self, exception):
        logging.error(f"Exception: {exception}")

    def on_disconnect(self):
        logging.warning("Disconnected from Twitter Stream.")

def main():
    # Initialize the streaming client
    stream = MongoTweetListener(bearer_token=TWITTER_BEARER_TOKEN, wait_on_rate_limit=True)
    # Remove any old stream rules so we only track the new one
    existing_rules = stream.get_rules()
    if existing_rules and existing_rules.data:
        rule_ids = [rule.id for rule in existing_rules.data]
        stream.delete_rules(rule_ids)
        logging.info("Cleared old stream rules.")
    # Add the user-specified tracking term as the rule
    for term in TRACK_TERMS:
        stream.add_rules(tweepy.StreamRule(term))
        logging.info(f"Added stream rule: {term}")
    # Start the live tweet stream
    logging.info("Starting tweet stream...")
    stream.filter(
        tweet_fields=["created_at", "lang", "entities"],
        expansions=["author_id"]
    )

if __name__ == "__main__":
    main()
