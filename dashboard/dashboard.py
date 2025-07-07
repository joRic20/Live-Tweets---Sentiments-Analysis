import streamlit as st                          # Import Streamlit for dashboard UI
import requests                                 # For API requests to backend
import pandas as pd                             # For dataframe handling
import plotly.graph_objects as go               # For plotting charts
from wordcloud import WordCloud                 # For generating word clouds
import matplotlib.pyplot as plt                 # For plotting word cloud
from streamlit_autorefresh import st_autorefresh # For auto-refreshing dashboard
import os                                       # For environment variable access
import sys                                      # For system exit if config is bad

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

# Get and clean environment variables
API_BASE = clean_env_var(os.environ.get("API_BASE"))
API_KEY = clean_env_var(os.environ.get("API_KEY"))

# Validate configuration
if not API_BASE:
    st.error("❌ API_BASE environment variable is not set!")
    st.info("Please set API_BASE in your .env file (e.g., API_BASE=http://backend:8000)")
    st.stop()

if not API_KEY:
    st.warning("⚠️ API_KEY is not set. Some features may not work.")

# Log configuration for debugging (without exposing full API key)
st.sidebar.text(f"API Base: {API_BASE}")
st.sidebar.text(f"API Key: {'Set' if API_KEY else 'Not Set'}")

st.set_page_config(page_title="Campaign Sentiment Analytics", layout="wide")
st.title("📊 Campaign Sentiment Analytics Dashboard")

# --- Auto-refresh every 60 seconds ---
st_autorefresh(interval=60000, key="dashboard_autorefresh")

# --- Start Ingestion Section in Sidebar ---
st.sidebar.markdown("### 🔎 Track New Campaign")
new_term = st.sidebar.text_input("Enter a brand, company, or campaign name")

if st.sidebar.button("Start Tracking"):
    # When user clicks button, attempt to start backend ingestion for this term
    if not new_term.strip():
        st.warning("Please enter a search term to track.")
    else:
        try:
            resp = requests.post(
                f"{API_BASE}/start_ingest/",
                json={"term": new_term},
                headers={"access_token": API_KEY} if API_KEY else {},
                timeout=10  # Add timeout
            )
            if resp.status_code == 200:
                st.sidebar.success(f"Ingestion started for: {new_term}")
            else:
                st.sidebar.error(f"Failed to start ingestion. Status: {resp.status_code}, Response: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.sidebar.error(f"Cannot connect to backend at {API_BASE}. Is the backend service running?")
        except Exception as e:
            st.sidebar.error(f"Error contacting backend: {e}")

# --- Helper: Fetch campaign options for the multi-select ---
@st.cache_data(ttl=60)
def fetch_campaigns():
    """Fetch list of available campaigns from backend"""
    try:
        headers = {"access_token": API_KEY} if API_KEY else {}
        resp = requests.get(f"{API_BASE}/campaigns/", headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to backend at {API_BASE}")
        return []
    except Exception as e:
        st.error(f"Error fetching campaigns: {e}")
        return []

# --- Helper: Fetch date range for filtering ---
@st.cache_data(ttl=60)
def fetch_date_range():
    """Get min/max date in the analytics DB"""
    try:
        headers = {"access_token": API_KEY} if API_KEY else {}
        resp = requests.get(f"{API_BASE}/date_range/", headers=headers, timeout=10)
        resp.raise_for_status()
        dates = resp.json()
        return pd.to_datetime(dates['min_date']).date(), pd.to_datetime(dates['max_date']).date()
    except Exception as e:
        st.error(f"Error fetching date range: {e}")
        # Return default date range if API fails
        import datetime
        today = datetime.date.today()
        return today - datetime.timedelta(days=7), today

# --- Sidebar campaign/date filters ---
campaigns = fetch_campaigns()
if not campaigns:
    st.warning("No campaigns found. Please start tracking a campaign using the sidebar.")
    st.stop()

min_date, max_date = fetch_date_range()
selected_campaigns = st.sidebar.multiselect(
    "Select Campaign(s)", campaigns, default=campaigns[:2] if len(campaigns) > 1 else campaigns)

if not selected_campaigns:
    st.info("Please select at least one campaign from the sidebar.")
    st.stop()

date_range = st.sidebar.date_input(
    "Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
params = {"campaigns": ",".join(selected_campaigns), "start": str(date_range[0]), "end": str(date_range[1])}

# --- Display Options in Sidebar ---
st.sidebar.markdown("### 📊 Display Options")
show_original_tweets = st.sidebar.checkbox("Show original tweets (with URLs)", value=False)
st.sidebar.info("Cleaned tweets have URLs, RT prefixes, and spam removed for better analysis")

# --- Analytics Data from backend ---
try:
    headers = {"access_token": API_KEY} if API_KEY else {}
    analytics_response = requests.get(f"{API_BASE}/analytics/", params=params, headers=headers, timeout=10)
    analytics_response.raise_for_status()
    df = pd.DataFrame(analytics_response.json())
except Exception as e:
    st.error(f"Error fetching analytics: {e}")
    st.stop()

if df.empty:
    st.warning("No data found for selected filters.")
    st.stop()

# --- Top Hashtags Section ---
st.subheader("🏷️ Top Hashtags")
try:
    headers = {"access_token": API_KEY} if API_KEY else {}
    tags_resp = requests.get(f"{API_BASE}/top_hashtags/", params={**params, "limit": 10}, headers=headers, timeout=10)
    tags_resp.raise_for_status()
    tags_dict = tags_resp.json()
    if tags_dict:
        st.dataframe(pd.DataFrame(list(tags_dict.items()), columns=["Hashtag", "Count"]), use_container_width=True)
    else:
        st.info("No hashtags found for selected filters.")
except Exception as e:
    st.error(f"Error fetching hashtags: {e}")

# --- Top Users Section ---
st.subheader("👤 Top Users")
try:
    headers = {"access_token": API_KEY} if API_KEY else {}
    users_resp = requests.get(f"{API_BASE}/top_users/", params={**params, "limit": 10}, headers=headers, timeout=10)
    users_resp.raise_for_status()
    users_dict = users_resp.json()
    if users_dict:
        st.dataframe(pd.DataFrame(list(users_dict.items()), columns=["User ID", "Tweet Count"]), use_container_width=True)
    else:
        st.info("No users found for selected filters.")
except Exception as e:
    st.error(f"Error fetching users: {e}")

# --- Sentiment Distribution Pie Chart ---
st.subheader("Sentiment Distribution (Pie Chart)")
try:
    headers = {"access_token": API_KEY} if API_KEY else {}
    tweets_params = params.copy()
    tweets_params["limit"] = 1000  # For more representative pie/wordcloud
    tweets_response = requests.get(f"{API_BASE}/latest_tweets/", params=tweets_params, headers=headers, timeout=10)
    tweets_response.raise_for_status()
    tweet_list = tweets_response.json()
    tweet_df = pd.DataFrame(tweet_list)
    
    if not tweet_df.empty and 'sentiment_label' in tweet_df.columns:
        sentiment_counts = tweet_df['sentiment_label'].value_counts()
        pie_fig = go.Figure(
            data=[go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                marker=dict(colors=['green', 'gray', 'red']),
                hole=.3
            )]
        )
        st.plotly_chart(pie_fig, use_container_width=True)
    else:
        st.info("No tweets for sentiment pie chart.")
except Exception as e:
    st.error(f"Error fetching tweets: {e}")
    tweet_df = pd.DataFrame()  # Empty dataframe for later sections

# --- Data Cleaning Statistics (if we have cleaned data) ---
if not tweet_df.empty and 'cleaned_text' in tweet_df.columns and 'text' in tweet_df.columns:
    st.subheader("🧹 Data Cleaning Impact")
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate statistics
    original_lengths = tweet_df['text'].str.len()
    cleaned_lengths = tweet_df['cleaned_text'].str.len()
    
    with col1:
        avg_original = original_lengths.mean()
        st.metric("Avg Original Length", f"{avg_original:.0f} chars")
    
    with col2:
        avg_cleaned = cleaned_lengths.mean()
        st.metric("Avg Cleaned Length", f"{avg_cleaned:.0f} chars")
    
    with col3:
        reduction = ((avg_original - avg_cleaned) / avg_original) * 100
        st.metric("Avg Reduction", f"{reduction:.1f}%")
    
    with col4:
        # Count tweets with URLs
        url_count = tweet_df['text'].str.contains(r'http[s]?://|www\.', regex=True).sum()
        st.metric("Tweets with URLs", f"{url_count} ({url_count/len(tweet_df)*100:.1f}%)")

# --- Tweet Volume Over Time Chart (by Sentiment) ---
st.subheader("Tweet Volume Over Time (by Sentiment)")
if not df.empty:
    vol_df = df.groupby(['date']).agg(
        positive=('pos_count', 'sum'),
        neutral=('neu_count', 'sum'),
        negative=('neg_count', 'sum')
    ).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vol_df['date'], y=vol_df['positive'], mode='lines+markers', name='Positive', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=vol_df['date'], y=vol_df['neutral'], mode='lines+markers', name='Neutral', line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=vol_df['date'], y=vol_df['negative'], mode='lines+markers', name='Negative', line=dict(color='red')))
    fig.update_layout(title="Tweet Volume Over Time", xaxis_title="Date", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

# --- Most Positive/Negative Tweets Display ---
st.subheader("🌟 Example Positive/Negative Tweets")
if not tweet_df.empty and 'sentiment_label' in tweet_df.columns:
    # Get most positive and negative tweets
    pos_tweets = tweet_df[tweet_df['sentiment_label'] == 'POSITIVE'].sort_values("sentiment_score", ascending=False)
    neg_tweets = tweet_df[tweet_df['sentiment_label'] == 'NEGATIVE'].sort_values("sentiment_score", ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not pos_tweets.empty:
            st.success("**Most Positive Tweet:**")
            pos_tweet = pos_tweets.iloc[0]
            # Use cleaned text if available and not showing original
            if 'cleaned_text' in pos_tweet and not show_original_tweets:
                st.write(pos_tweet['cleaned_text'])
                st.caption(f"Sentiment Score: {pos_tweet.get('sentiment_score', 'N/A'):.3f}")
            else:
                st.write(pos_tweet['text'])
                st.caption(f"Sentiment Score: {pos_tweet.get('sentiment_score', 'N/A'):.3f}")
    
    with col2:
        if not neg_tweets.empty:
            st.error("**Most Negative Tweet:**")
            neg_tweet = neg_tweets.iloc[0]
            # Use cleaned text if available and not showing original
            if 'cleaned_text' in neg_tweet and not show_original_tweets:
                st.write(neg_tweet['cleaned_text'])
                st.caption(f"Sentiment Score: {neg_tweet.get('sentiment_score', 'N/A'):.3f}")
            else:
                st.write(neg_tweet['text'])
                st.caption(f"Sentiment Score: {neg_tweet.get('sentiment_score', 'N/A'):.3f}")

# --- Campaign comparison table ---
st.subheader("📊 Campaign Comparison")
comparison = df.groupby('campaign').agg(
    total_tweets=('tweet_count', 'sum'),
    avg_sentiment=('avg_sentiment', 'mean'),
    percent_positive=('pos_count', lambda x: sum(x) / max(df[df['campaign'] == x.name]['tweet_count'].sum(), 1)),
    percent_negative=('neg_count', lambda x: sum(x) / max(df[df['campaign'] == x.name]['tweet_count'].sum(), 1))
).reset_index()
st.dataframe(comparison.style.format({'avg_sentiment': '{:.2f}', 'percent_positive': '{:.1%}', 'percent_negative': '{:.1%}'}), use_container_width=True)

# --- Time series sentiment trend chart ---
st.subheader("📈 Sentiment Trend Over Time")
for campaign in selected_campaigns:
    data = df[df['campaign'] == campaign]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['avg_sentiment'], mode='lines+markers', name='Avg Sentiment'))
    fig.add_trace(go.Bar(x=data['date'], y=data['pos_count'], name='Positive', marker_color='green', opacity=0.4))
    fig.add_trace(go.Bar(x=data['date'], y=data['neu_count'], name='Neutral', marker_color='gray', opacity=0.4))
    fig.add_trace(go.Bar(x=data['date'], y=data['neg_count'], name='Negative', marker_color='red', opacity=0.4))
    fig.update_layout(
        title=f"Sentiment and Volume for '{campaign}'",
        xaxis_title="Date",
        yaxis_title="Count / Sentiment",
        barmode='stack',
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Download as CSV ---
st.subheader("📥 Download Data")
col1, col2 = st.columns(2)
with col1:
    st.download_button("Download Analytics Summary", df.to_csv(index=False), "campaign_analytics.csv", "text/csv")
with col2:
    if not tweet_df.empty:
        # Prepare tweet data for download
        download_df = tweet_df.copy()
        if 'cleaned_text' in download_df.columns and not show_original_tweets:
            download_df = download_df[['created_at', 'cleaned_text', 'sentiment_label', 'sentiment_score', 'campaign']]
            download_df.columns = ['Time', 'Tweet (Cleaned)', 'Sentiment', 'Score', 'Campaign']
        else:
            download_df = download_df[['created_at', 'text', 'sentiment_label', 'sentiment_score', 'campaign']]
            download_df.columns = ['Time', 'Tweet', 'Sentiment', 'Score', 'Campaign']
        st.download_button("Download Tweet Data", download_df.to_csv(index=False), "tweet_data.csv", "text/csv")

# --- Colored tweets table ---
st.subheader("🗃️ Latest Tweets (Colored by Sentiment)")

def sentiment_row_color(row):
    if row.get('Sentiment') == 'POSITIVE':
        return ['background-color: #e6ffe6'] * len(row)
    elif row.get('Sentiment') == 'NEGATIVE':
        return ['background-color: #ffe6e6'] * len(row)
    else:
        return [''] * len(row)

if not tweet_df.empty:
    tweet_df['created_at'] = pd.to_datetime(tweet_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Prepare display dataframe based on user preference and data availability
    if 'cleaned_text' in tweet_df.columns and not show_original_tweets:
        # Show cleaned tweets
        display_df = tweet_df[["created_at", "cleaned_text", "sentiment_label"]].copy()
        display_df.columns = ["Time", "Tweet (Cleaned)", "Sentiment"]
        st.caption("Showing cleaned tweets (URLs, mentions, and noise removed)")
    else:
        # Show original tweets
        display_df = tweet_df[["created_at", "text", "sentiment_label"]].copy()
        display_df.columns = ["Time", "Tweet (Original)", "Sentiment"]
        st.caption("Showing original tweets")
    
    # Apply coloring and display
    styled_df = display_df.style.apply(sentiment_row_color, axis=1)
    st.dataframe(styled_df, use_container_width=True)
else:
    st.info("No tweets available for selected campaigns.")

# --- Word Cloud ---
st.subheader("☁️ Word Cloud")
if not tweet_df.empty:
    # Use cleaned text if available and not showing original
    if 'cleaned_text' in tweet_df.columns and not show_original_tweets:
        text_corpus = " ".join(tweet_df['cleaned_text'].astype(str))
        st.caption("Word cloud generated from cleaned tweets (more meaningful without URLs and noise)")
    else:
        text_corpus = " ".join(tweet_df['text'].astype(str))
        st.caption("Word cloud generated from original tweets")
    
    # Create word cloud with better parameters
    wordcloud = WordCloud(
        width=800, 
        height=300, 
        background_color='white',
        max_words=100,
        relative_scaling=0.5,
        colormap='viridis',
        stopwords=set(['https', 'http', 'RT', 'amp', 'user', 'users'])  # Additional stopwords
    ).generate(text_corpus)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
else:
    st.info("No tweet texts available for word cloud.")

# --- Footer ---
st.caption("Green=Positive, Red=Negative, White=Neutral. Dashboard refreshes every 60s. Use sidebar filters to explore.")
st.caption("💡 Tip: Toggle 'Show original tweets' in the sidebar to switch between cleaned and raw tweet text.")