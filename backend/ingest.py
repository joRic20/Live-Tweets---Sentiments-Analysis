# ingest.py
# Streams tweets based on user-inputted term (product, service, brand) into MongoDB.

from dotenv import load_dotenv
load_dotenv()

import os
import tweepy
from pymongo import MongoClient, errors
from datetime import datetime
import logging
import sys

# --- Load secrets from .env ---
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")
MONGO_URI = os.environ.get("MONGO_URI")
MONGO_DB = os.environ.get("MONGO_DB")
MONGO_COLL = os.environ.get("MONGO_COLL")

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

# --- MongoDB setup ---
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB]
tweets_collection = mongo_db[MONGO_COLL]

# Ensure tweet_id is unique; auto-delete old tweets after 30 days.
tweets_collection.create_index("tweet_id", unique=True)
tweets_collection.create_index("created_at", expireAfterSeconds=30 * 24 * 3600)

# Accept product/service/campaign/brand name as CLI argument.
# Only run if a search term is provided by the user
if len(sys.argv) > 1:
    TRACK_TERM = sys.argv[1]
else:
    print("Error: Please provide a product/service/campaign/brand name as a search term.")
    sys.exit(1)  # Exit the script with an error code

class MongoTweetListener(tweepy.StreamingClient):
    def on_tweet(self, tweet):
        # Prepare MongoDB document for each tweet
        tweet_doc = {
            "tweet_id": tweet.id,
            "text": tweet.text,
            "author_id": tweet.author_id,
            "created_at": tweet.created_at.isoformat() if tweet.created_at else datetime.utcnow().isoformat(),
            "hashtags": [tag['tag'] for tag in tweet.entities['hashtags']] if tweet.entities and 'hashtags' in tweet.entities else [],
            "lang": getattr(tweet, 'lang', None),
            "matched_term": TRACK_TERM,  # Store what user is tracking!
            "raw": tweet.data
        }
        try:
            tweets_collection.insert_one(tweet_doc)
            logging.info(f"Tweet stored: {tweet_doc['tweet_id']}")
        except errors.DuplicateKeyError:
            logging.debug(f"Duplicate tweet: {tweet_doc['tweet_id']}")
        except Exception as e:
            logging.error(f"Error inserting tweet: {e}")

    # Optional: log connection state
    def on_connect(self):
        logging.info("Connected to Twitter Streaming API.")

    def on_connection_error(self):
        logging.error("Connection error. Reconnecting...")
        self.disconnect()

    def on_disconnect(self):
        logging.warning("Disconnected from Twitter Stream.")

def main():
    stream = MongoTweetListener(bearer_token=TWITTER_BEARER_TOKEN, wait_on_rate_limit=True)
    # Remove old rules so we only track the new term
    existing_rules = stream.get_rules()
    if existing_rules and existing_rules.data:
        rule_ids = [rule.id for rule in existing_rules.data]
        stream.delete_rules(rule_ids)
        logging.info("Cleared old stream rules.")
    # Add rule for new term
    stream.add_rules(tweepy.StreamRule(TRACK_TERM))
    logging.info(f"Added stream rule: {TRACK_TERM}")
    # Start streaming
    stream.filter(tweet_fields=["created_at", "lang", "entities"], expansions=["author_id"])

if __name__ == "__main__":
    main()
