import os
import tweepy
from dotenv import load_dotenv
load_dotenv()

TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")

client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)

# Search for 2 most recent tweets matching the query
response = client.search_recent_tweets(query="Adidas", max_results=2)

for i, tweet in enumerate(response.data, 1):
    print(f"Tweet {i}: {tweet.text}\n")
