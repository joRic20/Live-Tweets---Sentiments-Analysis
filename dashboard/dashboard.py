import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

API_BASE = "http://localhost:8000"  # Change if backend is hosted elsewhere

st.set_page_config(page_title="Campaign Sentiment Analytics", layout="wide")
st.title("📊 Campaign Sentiment Analytics Dashboard")

# --- Auto-refresh every 60 seconds ---
st_autorefresh(interval=60000, key="dashboard_autorefresh")

# --- Sidebar Filters (Campaigns and Date Range) ---
@st.cache_data
def fetch_campaigns():
    return requests.get(f"{API_BASE}/campaigns/").json()

@st.cache_data
def fetch_date_range():
    dates = requests.get(f"{API_BASE}/date_range/").json()
    return pd.to_datetime(dates['min_date']).date(), pd.to_datetime(dates['max_date']).date()

campaigns = fetch_campaigns()
min_date, max_date = fetch_date_range()
selected_campaigns = st.sidebar.multiselect("Select Campaign(s)", campaigns, default=campaigns[:2])
date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
params = {"campaigns": ",".join(selected_campaigns), "start": str(date_range[0]), "end": str(date_range[1])}

# --- Analytics Data ---
analytics_response = requests.get(f"{API_BASE}/analytics/", params=params)
df = pd.DataFrame(analytics_response.json())

if df.empty:
    st.warning("No data found for selected filters.")
    st.stop()

# --- Top Hashtags ---
st.subheader("🏷️ Top Hashtags")
tags_resp = requests.get(f"{API_BASE}/top_hashtags/", params={**params, "limit": 10})
tags_dict = tags_resp.json()
if tags_dict:
    st.dataframe(pd.DataFrame(list(tags_dict.items()), columns=["Hashtag", "Count"]), use_container_width=True)
else:
    st.info("No hashtags found for selected filters.")

# --- Top Users ---
st.subheader("👤 Top Users")
users_resp = requests.get(f"{API_BASE}/top_users/", params={**params, "limit": 10})
users_dict = users_resp.json()
if users_dict:
    st.dataframe(pd.DataFrame(list(users_dict.items()), columns=["User ID", "Tweet Count"]), use_container_width=True)
else:
    st.info("No users found for selected filters.")

# --- Sentiment Distribution Pie Chart ---
st.subheader("Sentiment Distribution (Pie Chart)")
tweets_params = params.copy()
tweets_params["limit"] = 1000  # For more accuracy in pie chart and word cloud
tweets_response = requests.get(f"{API_BASE}/latest_tweets/", params=tweets_params)
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

# --- Tweet Volume Over Time (by Sentiment) ---
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

# --- Most Positive/Negative Tweets ---
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
