# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()  # Reads variables in .env and puts them into os.environ

# Import libraries for system interaction, Twitter API, MongoDB, datetime handling, and logging
import os
import tweepy
from pymongo import MongoClient, errors
from datetime import datetime
import logging

# ------------- SECRETS AND CONFIGS -------------

# Get Twitter Bearer Token from environment variables
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")
# Get MongoDB URI (connection string)
MONGO_URI = os.environ.get("MONGO_URI")
# Get MongoDB database name
MONGO_DB = os.environ.get("MONGO_DB")
# Get MongoDB collection name
MONGO_COLL = os.environ.get("MONGO_COLL")

# ------------- LOGGING -------------

# Set up logging so you can track what's happening in the script
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)

# ------------- MONGODB SETUP -------------

# Connect to MongoDB using the URI loaded from .env
mongo_client = MongoClient(MONGO_URI)
# Access the specified database
mongo_db = mongo_client[MONGO_DB]
# Access the tweets collection
tweets_collection = mongo_db[MONGO_COLL]

# Create a unique index on tweet_id to avoid duplicates
tweets_collection.create_index("tweet_id", unique=True)
# Create a TTL index on created_at to automatically delete tweets older than 30 days
tweets_collection.create_index("created_at", expireAfterSeconds=30 * 24 * 3600)

# ------------- TWITTER STREAM HANDLER -------------

# List of hashtags or keywords to track (edit as needed for your use case)
TRACK_TERMS = ["#YourHashtag", "brand", "campaign"]

# Define a custom listener class, subclassing Tweepy's StreamingClient
class MongoTweetListener(tweepy.StreamingClient):
    # Called each time a tweet matching your rules is received
    def on_tweet(self, tweet):
        # Prepare a document for MongoDB
        tweet_doc = {
            "tweet_id": tweet.id,  # Unique tweet ID from Twitter
            "text": tweet.text,  # The text content of the tweet
            "author_id": tweet.author_id,  # ID of the tweet's author
            # Tweet creation time, ISO formatted string
            "created_at": tweet.created_at.isoformat() if tweet.created_at else datetime.utcnow().isoformat(),
            # Extract hashtags if present
            "hashtags": [tag['tag'] for tag in tweet.entities['hashtags']] if tweet.entities and 'hashtags' in tweet.entities else [],
            # Tweet language (if available)
            "lang": getattr(tweet, 'lang', None),
            "raw": tweet.data  # Store the full raw tweet JSON
        }
        # Insert the tweet document into MongoDB
        try:
            tweets_collection.insert_one(tweet_doc)
            logging.info(f"Tweet stored: {tweet_doc['tweet_id']}")
        except errors.DuplicateKeyError:
            # Skip duplicate tweets
            logging.debug(f"Duplicate tweet: {tweet_doc['tweet_id']}")
        except Exception as e:
            # Log any insertion errors
            logging.error(f"Error inserting tweet: {e}")

    # Log connection events and errors
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

# ------------- MAIN FUNCTION -------------

def main():
    # Create a new streaming client with your bearer token
    stream = MongoTweetListener(bearer_token=TWITTER_BEARER_TOKEN, wait_on_rate_limit=True)

    # Remove old stream rules to avoid duplicate filters
    existing_rules = stream.get_rules()
    if existing_rules and existing_rules.data:
        rule_ids = [rule.id for rule in existing_rules.data]
        stream.delete_rules(rule_ids)
        logging.info("Cleared old stream rules.")

    # Add new rules for the hashtags/keywords you want to track
    for term in TRACK_TERMS:
        stream.add_rules(tweepy.StreamRule(term))
        logging.info(f"Added stream rule: {term}")

    logging.info("Starting tweet stream...")
    # Start streaming tweets that match your rules, requesting specific fields and expansions
    stream.filter(
        tweet_fields=["created_at", "lang", "entities"],
        expansions=["author_id"]
    )

# If this script is run directly (not imported), run the main function
if __name__ == "__main__":
    main()
