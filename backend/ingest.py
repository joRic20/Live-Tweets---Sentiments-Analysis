# ingest.py
# Enhanced tweet ingestion script for comprehensive sentiment analysis

from dotenv import load_dotenv
load_dotenv()

import os
import tweepy
from pymongo import MongoClient, errors
from datetime import datetime, timezone, timedelta
import logging
import sys
import time
import argparse
import re

def clean_env_var(value):
    """Remove inline comments and extra quotes from environment variables"""
    if value:
        if '#' in value:
            value = value.split('#')[0]
        value = value.strip().strip('"\'')
    return value

def extract_user_location(user_data):
    """Extract location information from user profile"""
    location_info = {
        'user_location': None,
        'user_description': None,
        'user_url': None
    }
    
    if hasattr(user_data, 'location') and user_data.location:
        location_info['user_location'] = user_data.location
    
    if hasattr(user_data, 'description') and user_data.description:
        location_info['user_description'] = user_data.description
    
    if hasattr(user_data, 'url') and user_data.url:
        location_info['user_url'] = user_data.url
    
    return location_info

def extract_engagement_metrics(tweet):
    """Extract engagement metrics that could indicate influence"""
    metrics = {
        'retweet_count': 0,
        'like_count': 0,
        'reply_count': 0,
        'quote_count': 0
    }
    
    if hasattr(tweet, 'public_metrics') and tweet.public_metrics:
        metrics.update({
            'retweet_count': tweet.public_metrics.get('retweet_count', 0),
            'like_count': tweet.public_metrics.get('like_count', 0),
            'reply_count': tweet.public_metrics.get('reply_count', 0),
            'quote_count': tweet.public_metrics.get('quote_count', 0)
        })
    
    return metrics

def detect_product_mentions(text):
    """Detect potential product/feature mentions in tweet text"""
    product_keywords = {
        'customer_service': ['support', 'service', 'help', 'customer service', 'helpdesk', 'assistance'],
        'product_quality': ['quality', 'build', 'materials', 'durability', 'craftsmanship', 'solid', 'cheap', 'flimsy'],
        'user_interface': ['ui', 'interface', 'design', 'usability', 'user experience', 'ux', 'layout', 'navigation'],
        'performance': ['speed', 'fast', 'slow', 'performance', 'efficiency', 'responsive', 'lag', 'quick'],
        'pricing': ['price', 'cost', 'expensive', 'cheap', 'value', 'affordable', 'overpriced', 'worth'],
        'features': ['feature', 'functionality', 'capabilities', 'options', 'tools', 'works', 'function'],
        'delivery': ['shipping', 'delivery', 'arrival', 'packaging', 'logistics', 'received', 'shipped']
    }
    
    detected_categories = []
    text_lower = text.lower()
    
    for category, keywords in product_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                detected_categories.append(category)
                break
    
    return detected_categories

# Load environment variables
TWITTER_BEARER_TOKEN = clean_env_var(os.environ.get("TWITTER_BEARER_TOKEN"))
MONGO_URI = clean_env_var(os.environ.get("MONGO_URI"))
MONGO_DB = clean_env_var(os.environ.get("MONGO_DB"))
MONGO_COLL = clean_env_var(os.environ.get("MONGO_COLL"))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Enhanced tweet search for sentiment analysis')
parser.add_argument('term', help='Search term to track')
parser.add_argument('--start', help='Start date (YYYY-MM-DD)', default=None)
parser.add_argument('--end', help='End date (YYYY-MM-DD)', default=None)
args = parser.parse_args()

TRACK_TERM = args.term
START_DATE = args.start
END_DATE = args.end
MAX_TWEETS = int(os.environ.get("MAX_TWEETS", "100"))

# Set up logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s", 
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ingest.log')
    ]
)
logger = logging.getLogger(__name__)

# Validate environment variables
if not all([TWITTER_BEARER_TOKEN, MONGO_URI, MONGO_DB, MONGO_COLL]):
    logger.error("Missing required environment variables!")
    sys.exit(1)

logger.info(f"Search Term: '{TRACK_TERM}'")
logger.info(f"Max Tweets: {MAX_TWEETS}")
if START_DATE:
    logger.info(f"Date Range: {START_DATE} to {END_DATE}")

# Set up MongoDB
try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo_client.server_info()
    mongo_db = mongo_client[MONGO_DB]
    tweets_collection = mongo_db[MONGO_COLL]
    logger.info(f"Connected to MongoDB - Database: {MONGO_DB}, Collection: {MONGO_COLL}")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    sys.exit(1)

# Create indexes
try:
    tweets_collection.create_index("tweet_id", unique=True)
    tweets_collection.create_index("created_at", expireAfterSeconds=30 * 24 * 3600)
    tweets_collection.create_index("matched_term")
    tweets_collection.create_index("user_id")  # New index for user analysis
    logger.info("MongoDB indexes created/verified")
except Exception as e:
    logger.error(f"Failed to create indexes: {e}")

def search_tweets():
    """Enhanced tweet search with additional metadata collection"""
    try:
        # Create Twitter client with enhanced options
        client = tweepy.Client(
            bearer_token=TWITTER_BEARER_TOKEN, 
            wait_on_rate_limit=True
        )
        logger.info("Connected to Twitter API v2")
        
        # Build enhanced search query
        query = f"{TRACK_TERM} -is:retweet lang:en"
        logger.info(f"Search query: {query}")
        
        # Handle date parameters
        start_time = None
        end_time = None
        
        if START_DATE:
            start_time = f"{START_DATE}T00:00:00Z"
            
            if END_DATE:
                end_date_obj = datetime.strptime(END_DATE, "%Y-%m-%d").date()
                today = datetime.now(timezone.utc).date()
                
                if end_date_obj >= today:
                    end_time = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
                else:
                    end_time = f"{END_DATE}T23:59:59Z"
            else:
                end_time = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        else:
            end_time = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
            start_time = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        
        logger.info(f"Search time range: {start_time} to {end_time}")
        
        tweet_count = 0
        
        # Enhanced paginator with more fields
        paginator = tweepy.Paginator(
            client.search_recent_tweets,
            query=query,
            start_time=start_time,
            end_time=end_time,
            max_results=100,
            tweet_fields=[
                'created_at', 'author_id', 'lang', 'entities', 'public_metrics',
                'context_annotations', 'geo', 'in_reply_to_user_id', 'referenced_tweets',
                'reply_settings', 'source', 'possibly_sensitive'
            ],
            user_fields=[
                'username', 'name', 'location', 'description', 'url', 'verified',
                'public_metrics', 'created_at', 'profile_image_url'
            ],
            expansions=[
                'author_id', 'geo.place_id', 'in_reply_to_user_id', 'referenced_tweets.id'
            ],
            place_fields=['country', 'country_code', 'full_name', 'geo', 'name', 'place_type']
        )
        
        for page in paginator:
            if not page.data:
                logger.info("No more tweets found")
                break
            
            # Create enhanced lookup dictionaries
            users_dict = {}
            places_dict = {}
            
            # Process included users
            if hasattr(page, 'includes'):
                if 'users' in page.includes:
                    for user in page.includes['users']:
                        user_metrics = {
                            'follower_count': 0,
                            'following_count': 0,
                            'tweet_count': 0,
                            'listed_count': 0
                        }
                        
                        if hasattr(user, 'public_metrics') and user.public_metrics:
                            user_metrics.update(user.public_metrics)
                        
                        users_dict[user.id] = {
                            'username': user.username,
                            'name': user.name,
                            'location': getattr(user, 'location', ''),
                            'description': getattr(user, 'description', ''),
                            'url': getattr(user, 'url', ''),
                            'verified': getattr(user, 'verified', False),
                            'profile_image_url': getattr(user, 'profile_image_url', ''),
                            'user_created_at': str(getattr(user, 'created_at', '')),
                            **user_metrics
                        }
                
                # Process included places
                if 'places' in page.includes:
                    for place in page.includes['places']:
                        places_dict[place.id] = {
                            'country': getattr(place, 'country', ''),
                            'country_code': getattr(place, 'country_code', ''),
                            'full_name': getattr(place, 'full_name', ''),
                            'name': getattr(place, 'name', ''),
                            'place_type': getattr(place, 'place_type', '')
                        }
            
            # Process each tweet with enhanced data
            for tweet in page.data:
                try:
                    # Get user data
                    user_data = users_dict.get(tweet.author_id, {})
                    
                    # Get place data
                    place_data = {}
                    if hasattr(tweet, 'geo') and tweet.geo and 'place_id' in tweet.geo:
                        place_data = places_dict.get(tweet.geo['place_id'], {})
                    
                    # Extract engagement metrics
                    engagement_metrics = extract_engagement_metrics(tweet)
                    
                    # Detect product mentions
                    product_categories = detect_product_mentions(tweet.text)
                    
                    # Enhanced MongoDB document
                    tweet_doc = {
                        # Basic tweet info
                        "tweet_id": str(tweet.id),
                        "text": tweet.text,
                        "author_id": str(tweet.author_id) if tweet.author_id else None,
                        "user_id": str(tweet.author_id) if tweet.author_id else None,
                        "created_at": tweet.created_at.isoformat() if tweet.created_at else datetime.utcnow().isoformat(),
                        "lang": getattr(tweet, 'lang', None),
                        "source": getattr(tweet, 'source', ''),
                        "possibly_sensitive": getattr(tweet, 'possibly_sensitive', False),
                        
                        # User information
                        "username": user_data.get('username', 'unknown'),
                        "user_name": user_data.get('name', ''),
                        "user_location": user_data.get('location', ''),
                        "user_description": user_data.get('description', ''),
                        "user_url": user_data.get('url', ''),
                        "user_verified": user_data.get('verified', False),
                        "user_followers_count": user_data.get('follower_count', 0),
                        "user_following_count": user_data.get('following_count', 0),
                        "user_tweet_count": user_data.get('tweet_count', 0),
                        "user_listed_count": user_data.get('listed_count', 0),
                        
                        # Geographic information
                        "place_country": place_data.get('country', ''),
                        "place_country_code": place_data.get('country_code', ''),
                        "place_full_name": place_data.get('full_name', ''),
                        "place_name": place_data.get('name', ''),
                        "place_type": place_data.get('place_type', ''),
                        
                        # Engagement metrics
                        "retweet_count": engagement_metrics['retweet_count'],
                        "like_count": engagement_metrics['like_count'],
                        "reply_count": engagement_metrics['reply_count'],
                        "quote_count": engagement_metrics['quote_count'],
                        
                        # Enhanced entity extraction
                        "hashtags": [],
                        "mentions": [],
                        "urls": [],
                        "cashtags": [],
                        
                        # Product/feature detection
                        "product_categories": product_categories,
                        
                        # Context and annotations
                        "context_annotations": [],
                        "in_reply_to_user_id": str(getattr(tweet, 'in_reply_to_user_id', '')) if getattr(tweet, 'in_reply_to_user_id', None) else None,
                        
                        # Metadata
                        "matched_term": TRACK_TERM,
                        "ingested_at": datetime.utcnow().isoformat(),
                        "search_date_range": f"{START_DATE} to {END_DATE}" if START_DATE else "last_7_days"
                    }
                    
                    # Extract enhanced entities
                    if hasattr(tweet, 'entities') and tweet.entities:
                        if 'hashtags' in tweet.entities:
                            tweet_doc['hashtags'] = [tag['tag'] for tag in tweet.entities['hashtags']]
                        if 'mentions' in tweet.entities:
                            tweet_doc['mentions'] = [
                                {
                                    'username': mention['username'],
                                    'id': mention.get('id', ''),
                                    'start': mention.get('start', 0),
                                    'end': mention.get('end', 0)
                                } for mention in tweet.entities['mentions']
                            ]
                        if 'urls' in tweet.entities:
                            tweet_doc['urls'] = [
                                {
                                    'url': url.get('url', ''),
                                    'expanded_url': url.get('expanded_url', ''),
                                    'display_url': url.get('display_url', ''),
                                    'title': url.get('title', ''),
                                    'description': url.get('description', '')
                                } for url in tweet.entities['urls']
                            ]
                        if 'cashtags' in tweet.entities:
                            tweet_doc['cashtags'] = [tag['tag'] for tag in tweet.entities['cashtags']]
                    
                    # Extract context annotations for topic detection
                    if hasattr(tweet, 'context_annotations') and tweet.context_annotations:
                        tweet_doc['context_annotations'] = [
                            {
                                'domain': ann.get('domain', {}),
                                'entity': ann.get('entity', {})
                            } for ann in tweet.context_annotations
                        ]
                    
                    # Store enhanced tweet in MongoDB
                    tweets_collection.insert_one(tweet_doc)
                    tweet_count += 1
                    
                    # Enhanced logging
                    logger.info(f"Tweet {tweet_count}/{MAX_TWEETS}: @{tweet_doc.get('username', 'unknown')} "
                               f"({tweet_doc.get('user_followers_count', 0)} followers) "
                               f"- {tweet.text[:50]}...")
                    
                    if product_categories:
                        logger.info(f"  └─ Product categories detected: {', '.join(product_categories)}")
                    
                    if place_data.get('country'):
                        logger.info(f"  └─ Location: {place_data['country']}")
                    
                    # Check if we've reached the limit
                    if tweet_count >= MAX_TWEETS:
                        logger.info(f"Reached max tweet limit ({MAX_TWEETS} tweets)")
                        return tweet_count
                        
                except errors.DuplicateKeyError:
                    logger.debug(f"Duplicate tweet: {tweet.id}")
                except Exception as e:
                    logger.error(f"Error inserting tweet: {e}")
            
            # Rate limiting courtesy pause
            time.sleep(0.5)
        
        logger.info(f"Search completed. Total tweets collected: {tweet_count}")
        return tweet_count
        
    except tweepy.TooManyRequests as e:
        logger.error("Rate limit exceeded! Please wait before trying again.")
    except tweepy.Unauthorized:
        logger.error("Authentication failed! Check your bearer token.")
    except Exception as e:
        logger.error(f"Error during search: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return 0

def main():
    """Main execution function"""
    try:
        start_time = time.time()
        logger.info(f"Starting enhanced tweet search for: '{TRACK_TERM}'")
        
        # Perform search
        tweet_count = search_tweets()
        
        # Calculate runtime
        elapsed_time = time.time() - start_time
        logger.info(f"Search completed in {elapsed_time:.1f} seconds")
        logger.info(f"Total tweets collected: {tweet_count}")
        
        # Summary statistics
        if tweet_count > 0:
            logger.info("Enhanced data captured:")
            logger.info("  ✓ User metrics (followers, engagement)")
            logger.info("  ✓ Geographic information")
            logger.info("  ✓ Product/feature mentions")
            logger.info("  ✓ Enhanced entity extraction")
            logger.info("  ✓ Context annotations")
        
    except KeyboardInterrupt:
        logger.info("Search interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Cleanup
        if 'mongo_client' in locals():
            mongo_client.close()
            logger.info("MongoDB connection closed")

if __name__ == "__main__":
    main()