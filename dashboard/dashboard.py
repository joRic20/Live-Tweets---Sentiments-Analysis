import streamlit as st                          # Import Streamlit for dashboard UI
import requests                                 # For API requests to backend
import pandas as pd                             # For dataframe handling
import plotly.graph_objects as go               # For plotting charts
from wordcloud import WordCloud                 # For generating word clouds
import matplotlib.pyplot as plt                 # For plotting word cloud
from streamlit_autorefresh import st_autorefresh # For auto-refreshing dashboard

API_BASE = os.environ.get("API_BASE")              # Base URL for FastAPI backend (change if hosted remotely)
API_KEY = os.environ.get("API_KEY")                     # <-- Set your API key here, or load from st.secrets/env

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
                headers={"access_token": API_KEY}
            )
            if resp.status_code == 200:
                st.sidebar.success(f"Ingestion started for: {new_term}")
            else:
                st.sidebar.error(f"Failed to start ingestion. {resp.text}")
        except Exception as e:
            st.sidebar.error(f"Error contacting backend: {e}")

# --- Helper: Fetch campaign options for the multi-select ---
@st.cache_data
def fetch_campaigns():
    # Fetch list of available campaigns from backend
    return requests.get(f"{API_BASE}/campaigns/", headers={"access_token": API_KEY}).json()

# --- Helper: Fetch date range for filtering ---
@st.cache_data
def fetch_date_range():
    # Get min/max date in the analytics DB
    dates = requests.get(f"{API_BASE}/date_range/", headers={"access_token": API_KEY}).json()
    return pd.to_datetime(dates['min_date']).date(), pd.to_datetime(dates['max_date']).date()

# --- Sidebar campaign/date filters ---
campaigns = fetch_campaigns()
min_date, max_date = fetch_date_range()
selected_campaigns = st.sidebar.multiselect(
    "Select Campaign(s)", campaigns, default=campaigns[:2] if len(campaigns) > 1 else campaigns)
date_range = st.sidebar.date_input(
    "Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
params = {"campaigns": ",".join(selected_campaigns), "start": str(date_range[0]), "end": str(date_range[1])}

# --- Analytics Data from backend ---
analytics_response = requests.get(f"{API_BASE}/analytics/", params=params, headers={"access_token": API_KEY})
df = pd.DataFrame(analytics_response.json())

if df.empty:
    st.warning("No data found for selected filters.")
    st.stop()

# --- Top Hashtags Section ---
st.subheader("🏷️ Top Hashtags")
tags_resp = requests.get(f"{API_BASE}/top_hashtags/", params={**params, "limit": 10}, headers={"access_token": API_KEY})
tags_dict = tags_resp.json()
if tags_dict:
    st.dataframe(pd.DataFrame(list(tags_dict.items()), columns=["Hashtag", "Count"]), use_container_width=True)
else:
    st.info("No hashtags found for selected filters.")

# --- Top Users Section ---
st.subheader("👤 Top Users")
users_resp = requests.get(f"{API_BASE}/top_users/", params={**params, "limit": 10}, headers={"access_token": API_KEY})
users_dict = users_resp.json()
if users_dict:
    st.dataframe(pd.DataFrame(list(users_dict.items()), columns=["User ID", "Tweet Count"]), use_container_width=True)
else:
    st.info("No users found for selected filters.")

# --- Sentiment Distribution Pie Chart ---
st.subheader("Sentiment Distribution (Pie Chart)")
tweets_params = params.copy()
tweets_params["limit"] = 1000  # For more representative pie/wordcloud
tweets_response = requests.get(f"{API_BASE}/latest_tweets/", params=tweets_params, headers={"access_token": API_KEY})
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
st.subheader("🌟 Most Positive/Negative Tweets")
if not tweet_df.empty and 'sentiment_label' in tweet_df.columns:
    pos_tweet = tweet_df[tweet_df['sentiment_label'] == 'POSITIVE'].sort_values("created_at", ascending=False).head(1)
    neg_tweet = tweet_df[tweet_df['sentiment_label'] == 'NEGATIVE'].sort_values("created_at", ascending=False).head(1)
    if not pos_tweet.empty:
        st.success(f"Most Recent Positive Tweet: {pos_tweet.iloc[0]['text']}")
    if not neg_tweet.empty:
        st.error(f"Most Recent Negative Tweet: {neg_tweet.iloc[0]['text']}")

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
st.download_button("Download CSV", df.to_csv(index=False), "filtered_campaign_data.csv", "text/csv")

# --- Colored tweets table ---
st.subheader("🗃️ Latest Tweets (Colored by Sentiment)")
def sentiment_row_color(row):
    if row.get('sentiment_label') == 'POSITIVE':
        return ['background-color: #e6ffe6'] * len(row)
    elif row.get('sentiment_label') == 'NEGATIVE':
        return ['background-color: #ffe6e6'] * len(row)
    else:
        return [''] * len(row)

if not tweet_df.empty:
    tweet_df['created_at'] = pd.to_datetime(tweet_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
    st.dataframe(tweet_df[["created_at", "text", "sentiment_label"]].style.apply(sentiment_row_color, axis=1))
else:
    st.info("No tweets available for selected campaigns.")

# --- Word Cloud ---
st.subheader("☁️ Word Cloud")
if not tweet_df.empty:
    text_corpus = " ".join(tweet_df['text'].astype(str))
    wordcloud = WordCloud(width=800, height=300, background_color='white').generate(text_corpus)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
else:
    st.info("No tweet texts available for word cloud.")

st.caption("Green=Positive, Red=Negative, White=Neutral. Dashboard refreshes every 60s. Use sidebar filters to explore.")
