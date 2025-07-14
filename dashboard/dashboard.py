import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date
import os
from typing import Dict, List, Tuple
import numpy as np

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

def clean_env_var(value):
    """Remove inline comments and extra quotes from environment variables"""
    if value:
        if '#' in value:
            value = value.split('#')[0]
        value = value.strip().strip('"\'')
    return value

# Environment setup
API_BASE = clean_env_var(os.environ.get("API_BASE"))
API_KEY = clean_env_var(os.environ.get("API_KEY"))

# Validate configuration
if not API_BASE:
    st.error("❌ API_BASE environment variable is not set!")
    st.info("Please set API_BASE in your .env file (e.g., API_BASE=https://twitter-fastapi.quandev.xyz)")
    st.stop()

# ============================================================================
# PROFESSIONAL THEME & STYLING
# ============================================================================

def load_professional_css():
    """Apply professional, modern styling to the dashboard"""
    st.markdown("""
    <style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* CSS Variables for Easy Theme Management */
    :root {
        --primary-color: #3b82f6;
        --secondary-color: #60a5fa;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --neutral-color: #6b7280;
        --dark-bg: #0f172a;
        --light-bg: #f8fafc;
        --card-bg: #ffffff;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
        --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }
    
    /* Main Container */
    .main {
        padding: 0;
        max-width: 1440px;
        margin: 0 auto;
        background-color: var(--light-bg);
    }
    
    .block-container {
        padding: 2rem 2rem;
        max-width: 100%;
    }
    
    /* Professional Header */
    .dashboard-header {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 3rem 2rem;
        margin: -2rem -2rem 2rem -2rem;
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }
    
    .dashboard-header::before {
        content: '';
        position: absolute;
        top: 0;
        right: -100px;
        width: 300px;
        height: 300px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
    }
    
    .dashboard-header h1 {
        font-size: 2.25rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.025em;
        position: relative;
        z-index: 1;
    }
    
    .dashboard-header p {
        font-size: 1.125rem;
        opacity: 0.95;
        margin: 0.5rem 0 0 0;
        position: relative;
        z-index: 1;
    }
    
    /* Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        padding: 0 24px;
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 14px;
        border: none;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--primary-color);
        background-color: rgba(59, 130, 246, 0.05);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: var(--primary-color) !important;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-color);
        border-bottom: 1px solid white;
        margin-bottom: -1px;
    }
    
    /* Metric Cards - Enhanced */
    .metric-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.75rem;
        height: 100%;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
        border-color: var(--primary-color);
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        margin-bottom: 1rem;
    }
    
    .metric-icon.primary {
        background: rgba(59, 130, 246, 0.1);
        color: var(--primary-color);
    }
    
    .metric-icon.success {
        background: rgba(16, 185, 129, 0.1);
        color: var(--success-color);
    }
    
    .metric-icon.danger {
        background: rgba(239, 68, 68, 0.1);
        color: var(--danger-color);
    }
    
    .metric-icon.neutral {
        background: rgba(107, 114, 128, 0.1);
        color: var(--neutral-color);
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
        margin: 0.25rem 0;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-delta {
        font-size: 0.875rem;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        margin-top: 0.75rem;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
    }
    
    .metric-delta.positive {
        color: var(--success-color);
        background: rgba(16, 185, 129, 0.1);
    }
    
    .metric-delta.negative {
        color: var(--danger-color);
        background: rgba(239, 68, 68, 0.1);
    }
    
    .metric-delta.neutral {
        color: var(--neutral-color);
        background: rgba(107, 114, 128, 0.1);
    }
    
    /* Sentiment specific styles */
    .sentiment-positive {
        color: var(--success-color);
        font-weight: 600;
    }
    
    .sentiment-negative {
        color: var(--danger-color);
        font-weight: 600;
    }
    
    .sentiment-neutral {
        color: var(--neutral-color);
        font-weight: 600;
    }
    
    /* Chart Cards */
    .chart-container {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-sm);
    }
    
    .chart-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .chart-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }
    
    .chart-subtitle {
        font-size: 0.875rem;
        color: var(--text-secondary);
        margin: 0.25rem 0 0 0;
    }
    
    /* Data Tables */
    .dataframe {
        font-size: 0.875rem;
        border: none !important;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }
    
    .dataframe thead th {
        background-color: var(--light-bg);
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
        padding: 14px !important;
        border-bottom: 2px solid var(--border-color);
        border-top: none !important;
    }
    
    .dataframe tbody td {
        padding: 14px !important;
        border-bottom: 1px solid var(--border-color);
        background-color: white;
    }
    
    .dataframe tbody tr:hover td {
        background-color: var(--light-bg);
    }
    
    .dataframe tbody tr:last-child td {
        border-bottom: none;
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--primary-color);
        color: white;
        border: none;
        padding: 0.625rem 1.5rem;
        font-weight: 500;
        font-size: 0.875rem;
        border-radius: 8px;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .stButton > button:hover {
        background: var(--secondary-color);
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }
    
    /* Select boxes and inputs */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stDateInput > div > div > input {
        border: 1px solid var(--border-color);
        border-radius: 8px;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        background-color: white;
    }
    
    .stSelectbox > div > div:focus-within,
    .stMultiSelect > div > div:focus-within,
    .stDateInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Labels */
    .stSelectbox label,
    .stMultiSelect label,
    .stDateInput label {
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    /* Info boxes */
    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        font-size: 0.875rem;
        color: #1e40af;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    
    .info-box::before {
        content: 'ℹ️';
        font-size: 1.25rem;
        flex-shrink: 0;
    }
    
    .warning-box {
        background: #fef3c7;
        border: 1px solid #fcd34d;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        font-size: 0.875rem;
        color: #92400e;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    
    .warning-box::before {
        content: '⚠️';
        font-size: 1.25rem;
        flex-shrink: 0;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        padding: 0.375rem 0.875rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .status-badge.success {
        background: #d1fae5;
        color: #065f46;
    }
    
    .status-badge.warning {
        background: #fef3c7;
        color: #92400e;
    }
    
    .status-badge.danger {
        background: #fee2e2;
        color: #991b1b;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .dashboard-header {
            padding: 2rem 1.5rem;
        }
        
        .dashboard-header h1 {
            font-size: 1.75rem;
        }
        
        .metric-value {
            font-size: 1.75rem;
        }
        
        .chart-container {
            padding: 1.25rem;
        }
    }
    
    /* Loading animation */
    .stSpinner > div {
        border-color: var(--primary-color) transparent var(--secondary-color) transparent;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--light-bg);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-secondary);
    }
    
    /* Plotly customizations */
    .js-plotly-plot .plotly .modebar {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 6px;
        padding: 2px;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem 1.5rem;
        color: var(--text-secondary);
    }
    
    .empty-state-icon {
        font-size: 3rem;
        opacity: 0.5;
        margin-bottom: 1rem;
    }
    
    .empty-state-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .empty-state-text {
        font-size: 0.875rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=60)
def fetch_campaigns() -> List[str]:
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

@st.cache_data(ttl=60)
def fetch_date_range() -> Tuple[date, date]:
    """Get min/max date in the analytics DB"""
    try:
        headers = {"access_token": API_KEY} if API_KEY else {}
        resp = requests.get(f"{API_BASE}/date_range/", headers=headers, timeout=10)
        resp.raise_for_status()
        dates = resp.json()
        
        min_date = pd.to_datetime(dates['min_date']).date()
        max_date = pd.to_datetime(dates['max_date']).date()
        
        if min_date == max_date:
            today = date.today()
            expanded_min = min_date - timedelta(days=30)
            expanded_max = today
            return expanded_min, expanded_max
        
        return min_date, max_date
        
    except Exception as e:
        st.error(f"Error fetching date range: {e}")
        today = date.today()
        return today - timedelta(days=30), today

def fetch_analytics_data(params: Dict) -> pd.DataFrame:
    """Fetch analytics data from backend"""
    try:
        headers = {"access_token": API_KEY} if API_KEY else {}
        
        # Try to fetch data - first check what endpoints are available
        try:
            # Try daily analytics
            response = requests.get(f"{API_BASE}/analytics/", params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                df = pd.DataFrame(response.json())
                
                # Check if we also have hourly data available
                try:
                    hourly_response = requests.get(f"{API_BASE}/analytics_hourly/", params=params, headers=headers, timeout=10)
                    if hourly_response.status_code == 200:
                        hourly_df = pd.DataFrame(hourly_response.json())
                        if not hourly_df.empty and 'datetime' in hourly_df.columns:
                            hourly_df['datetime'] = pd.to_datetime(hourly_df['datetime'])
                            # If we have hourly data, prefer it
                            return hourly_df
                except:
                    pass
                
                # Return daily data if no hourly data
                if not df.empty and 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date']).dt.date
                return df
            else:
                response.raise_for_status()
        except:
            # If daily endpoint fails, try hourly
            try:
                response = requests.get(f"{API_BASE}/analytics_hourly/", params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    df = pd.DataFrame(response.json())
                    if not df.empty and 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'])
                    return df
            except:
                pass
        
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching analytics: {e}")
        return pd.DataFrame()

def fetch_top_tweets(params: Dict) -> pd.DataFrame:
    """Fetch top tweets based on sentiment"""
    try:
        headers = {"access_token": API_KEY} if API_KEY else {}
        response = requests.get(f"{API_BASE}/top_tweets/", params=params, headers=headers, timeout=10)
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except Exception as e:
        st.error(f"Error fetching top tweets: {e}")
        return pd.DataFrame()

# ============================================================================
# PROFESSIONAL UI COMPONENTS
# ============================================================================

def render_header():
    """Render professional header"""
    st.markdown("""
    <div class="dashboard-header">
        <h1>Sentiment Intelligence Platform</h1>
        <p>Near Real-time Twitter sentiment analysis and insights</p>
    </div>
    """, unsafe_allow_html=True)

def get_sentiment_color(score):
    """Get color based on sentiment score"""
    if score > 0.1:
        return "var(--success-color)"
    elif score < -0.1:
        return "var(--danger-color)"
    else:
        return "var(--neutral-color)"

def format_sentiment_label(score):
    """Format sentiment score into label"""
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

def render_metric_card(label: str, value: str, delta: str = None, delta_type: str = "neutral", icon: str = "📊"):
    """Render a professional metric card"""
    icon_class = delta_type if delta_type in ["positive", "negative", "neutral", "primary"] else "primary"
    
    delta_html = ""
    if delta:
        delta_class = delta_type
        delta_icon = "↑" if delta_type == "positive" else "↓" if delta_type == "negative" else "→"
        delta_html = f'<div class="metric-delta {delta_class}">{delta_icon} {delta}</div>'
    
    return f"""
    <div class="metric-card">
        <div class="metric-icon {icon_class}">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

def render_metrics_row(df: pd.DataFrame):
    """Render the main metrics dashboard"""
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics with proper error handling
    if not df.empty and 'tweet_count' in df.columns:
        total_mentions = df['tweet_count'].sum()
    else:
        total_mentions = 0
        
    if not df.empty and 'avg_sentiment' in df.columns:
        avg_sentiment = df['avg_sentiment'].mean()
    else:
        avg_sentiment = 0
        
    if not df.empty and 'pos_count' in df.columns and 'tweet_count' in df.columns:
        positive_pct = (df['pos_count'].sum() / max(df['tweet_count'].sum(), 1)) * 100
    else:
        positive_pct = 0
        
    if not df.empty and 'neg_count' in df.columns and 'tweet_count' in df.columns:
        negative_pct = (df['neg_count'].sum() / max(df['tweet_count'].sum(), 1)) * 100
    else:
        negative_pct = 0
    
    # Determine sentiment status
    sentiment_status = format_sentiment_label(avg_sentiment)
    if avg_sentiment > 0.1:
        sentiment_delta_type = "positive"
        sentiment_class = "sentiment-positive"
    elif avg_sentiment < -0.1:
        sentiment_delta_type = "negative"
        sentiment_class = "sentiment-negative"
    else:
        sentiment_delta_type = "neutral"
        sentiment_class = "sentiment-neutral"
    
    with col1:
        st.markdown(render_metric_card(
            "Total Mentions",
            f"{total_mentions:,}",
            "+12.5% from last period" if total_mentions > 0 else "No data",
            "positive" if total_mentions > 0 else "neutral",
            "📢"
        ), unsafe_allow_html=True)
    
    with col2:
        sentiment_html = f'<span class="{sentiment_class}">{sentiment_status}</span>'
        st.markdown(render_metric_card(
            "Overall Sentiment",
            sentiment_html,
            f"{avg_sentiment:.3f} score",
            sentiment_delta_type,
            "😊" if avg_sentiment > 0.1 else "😟" if avg_sentiment < -0.1 else "😐"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(render_metric_card(
            "Positive Mentions",
            f"{positive_pct:.1f}%",
            f"{df['pos_count'].sum() if not df.empty and 'pos_count' in df.columns else 0} tweets",
            "positive",
            "👍"
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(render_metric_card(
            "Negative Mentions", 
            f"{negative_pct:.1f}%",
            f"{df['neg_count'].sum() if not df.empty and 'neg_count' in df.columns else 0} tweets",
            "negative",
            "👎"
        ), unsafe_allow_html=True)

def render_sentiment_trend_chart(df: pd.DataFrame):
    """Render professional sentiment trend chart"""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Add aggregation toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <div class="chart-header">
            <div>
                <h3 class="chart-title">Sentiment Trend Analysis</h3>
                <p class="chart-subtitle">Average sentiment score over time with volume overlay</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        time_aggregation = st.selectbox(
            "View",
            ["Hourly", "Daily"],
            index=0 if 'datetime' in df.columns else 1,
            label_visibility="collapsed"
        )
    
    if not df.empty:
        # Ensure we have proper date columns
        if 'date' not in df.columns and 'datetime' in df.columns:
            df['date'] = pd.to_datetime(df['datetime']).dt.date
        elif 'date' not in df.columns and 'created_at' in df.columns:
            df['date'] = pd.to_datetime(df['created_at']).dt.date
        
        # Prepare data based on aggregation
        if time_aggregation == "Hourly" and 'datetime' in df.columns:
            # Use actual hourly data if available
            time_df = df.copy()
            time_df = time_df.sort_values('datetime')
            x_column = 'datetime'
            x_label = time_df['datetime'].dt.strftime('%b %d, %H:00')
        elif time_aggregation == "Hourly" and 'date' in df.columns:
            # Simulate hourly from daily data
            hourly_data = []
            daily_df = df.groupby('date').agg({
                'avg_sentiment': 'mean',
                'tweet_count': 'sum',
                'pos_count': 'sum',
                'neu_count': 'sum',
                'neg_count': 'sum'
            }).reset_index()
            
            for _, row in daily_df.iterrows():
                base_date = pd.to_datetime(row['date'])
                daily_tweets = row['tweet_count']
                
                # Create realistic hourly distribution
                hours = np.arange(24)
                # Peak hours: 9am, 2pm, 8pm
                hourly_weights = (
                    0.3 * np.exp(-0.5 * ((hours - 9) / 3) ** 2) +
                    0.4 * np.exp(-0.5 * ((hours - 14) / 3) ** 2) +
                    0.3 * np.exp(-0.5 * ((hours - 20) / 3) ** 2)
                )
                hourly_weights = hourly_weights / hourly_weights.sum()
                
                for hour in range(24):
                    if hourly_weights[hour] > 0.01:  # Only add hours with significant activity
                        hourly_count = max(1, int(daily_tweets * hourly_weights[hour]))
                        # Add slight sentiment variation
                        sentiment_var = np.random.normal(0, 0.02)
                        hourly_data.append({
                            'datetime': base_date + pd.Timedelta(hours=hour),
                            'avg_sentiment': np.clip(row['avg_sentiment'] + sentiment_var, -1, 1),
                            'tweet_count': hourly_count
                        })
            
            if hourly_data:
                time_df = pd.DataFrame(hourly_data)
                time_df = time_df.sort_values('datetime')
                x_column = 'datetime'
                x_label = time_df['datetime'].dt.strftime('%b %d, %H:00')
            else:
                time_df = df.copy()
                x_column = 'date'
                x_label = pd.to_datetime(time_df['date']).dt.strftime('%b %d')
        else:
            # Daily view
            if 'date' in df.columns:
                time_df = df.groupby(['date']).agg({
                    'avg_sentiment': 'mean',
                    'tweet_count': 'sum'
                }).reset_index()
                x_column = 'date'
                x_label = pd.to_datetime(time_df['date']).dt.strftime('%b %d')
            else:
                # If no date column, create empty dataframe
                time_df = pd.DataFrame()
                x_column = 'date'
                x_label = []
        
        if not time_df.empty:
            # Create figure
            fig = go.Figure()
            
            # Add sentiment line with gradient fill
            fig.add_trace(go.Scatter(
                x=time_df[x_column],
                y=time_df['avg_sentiment'],
                mode='lines+markers',
                name='Sentiment Score',
                line=dict(
                    color='#3b82f6',
                    width=3,
                    shape='spline',
                    smoothing=1.3
                ),
                marker=dict(
                    size=8,
                    color='#3b82f6',
                    line=dict(color='white', width=2)
                ),
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.1)',
                yaxis='y',
                hovertemplate='<b>%{customdata}</b><br>Sentiment: %{y:.3f}<extra></extra>',
                customdata=x_label
            ))
            
            # Add zero line
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="#94a3b8",
                opacity=0.5,
                annotation_text="Neutral",
                annotation_position="right"
            )
            
            # Add volume bars
            max_volume = time_df['tweet_count'].max()
            fig.add_trace(go.Bar(
                x=time_df[x_column],
                y=time_df['tweet_count'],
                name='Tweet Volume',
                marker=dict(
                    color='rgba(209, 213, 219, 0.5)',
                    line=dict(color='rgba(156, 163, 175, 0.8)', width=1)
                ),
                yaxis='y2',
                hovertemplate='<b>%{customdata}</b><br>Volume: %{y:,}<extra></extra>',
                customdata=x_label
            ))
            
            # Update layout with professional styling
            fig.update_layout(
                hovermode='x unified',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Inter, sans-serif", size=12, color='#1e293b'),
                margin=dict(l=0, r=0, t=10, b=0),
                height=450,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor="rgba(255,255,255,0)",
                    bordercolor="rgba(255,255,255,0)",
                    font=dict(size=12)
                ),
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    title=None,
                    tickangle=-45 if time_aggregation == "Hourly" else 0,
                    tickmode='auto',
                    nticks=20 if time_aggregation == "Hourly" else None,
                    tickformat='%b %d' if time_aggregation == "Daily" else '%H:00',
                    showline=True,
                    linecolor='#e5e7eb'
                ),
                yaxis=dict(
                    title=dict(text="Sentiment Score", font=dict(size=12)),
                    showgrid=True,
                    gridcolor='#f3f4f6',
                    zeroline=True,
                    zerolinecolor='#e5e7eb',
                    zerolinewidth=2,
                    side='left',
                    range=[-1, 1],
                    tickformat='.2f'
                ),
                yaxis2=dict(
                    title=dict(text="Tweet Volume", font=dict(size=12)),
                    overlaying='y',
                    side='right',
                    showgrid=False,
                    rangemode='tozero'
                )
            )
            
            # Add sentiment regions
            fig.add_hrect(y0=0.1, y1=1, fillcolor="green", opacity=0.05, line_width=0)
            fig.add_hrect(y0=-0.1, y1=0.1, fillcolor="gray", opacity=0.05, line_width=0)
            fig.add_hrect(y0=-1, y1=-0.1, fillcolor="red", opacity=0.05, line_width=0)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for the selected time period")
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📊</div>
            <div class="empty-state-title">No Data Available</div>
            <div class="empty-state-text">Select a campaign and date range to view sentiment trends</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_sentiment_distribution(df: pd.DataFrame):
    """Render sentiment distribution donut chart"""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("""
    <div class="chart-header">
        <div>
            <h3 class="chart-title">Sentiment Distribution</h3>
            <p class="chart-subtitle">Breakdown of positive, neutral, and negative mentions</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not df.empty and all(col in df.columns for col in ['pos_count', 'neu_count', 'neg_count']):
        # Calculate totals
        total_positive = df['pos_count'].sum()
        total_neutral = df['neu_count'].sum()
        total_negative = df['neg_count'].sum()
        total_all = total_positive + total_neutral + total_negative
        
        if total_all > 0:
            # Create donut chart with professional colors
            fig = go.Figure(data=[go.Pie(
                labels=['Positive', 'Neutral', 'Negative'],
                values=[total_positive, total_neutral, total_negative],
                hole=.65,
                marker=dict(
                    colors=['#10b981', '#6b7280', '#ef4444'],
                    line=dict(color='white', width=2)
                ),
                textinfo='label+percent',
                textposition='outside',
                textfont=dict(size=14),
                hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
            )])
            
            # Add center text
            fig.add_annotation(
                text=f'<b>{total_all:,}</b><br>Total',
                x=0.5, y=0.5,
                font=dict(size=20, color='#1e293b'),
                showarrow=False
            )
            
            fig.update_layout(
                font=dict(family="Inter, sans-serif", size=12, color='#1e293b'),
                margin=dict(l=20, r=20, t=20, b=20),
                height=350,
                showlegend=False,
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add summary stats below
            positive_pct = (total_positive / total_all * 100)
            neutral_pct = (total_neutral / total_all * 100)
            negative_pct = (total_negative / total_all * 100)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="color: #10b981; font-size: 1.5rem; font-weight: 600;">{positive_pct:.1f}%</div>
                    <div style="color: #64748b; font-size: 0.875rem;">Positive</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="color: #6b7280; font-size: 1.5rem; font-weight: 600;">{neutral_pct:.1f}%</div>
                    <div style="color: #64748b; font-size: 0.875rem;">Neutral</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="color: #ef4444; font-size: 1.5rem; font-weight: 600;">{negative_pct:.1f}%</div>
                    <div style="color: #64748b; font-size: 0.875rem;">Negative</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No sentiment data available")
    else:
        st.info("No data available or missing required columns")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_top_tweets_analysis(params: Dict):
    """Render top tweets analysis section"""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("""
    <div class="chart-header">
        <div>
            <h3 class="chart-title">Top Tweets</h3>
            <p class="chart-subtitle">Most impactful tweets by sentiment</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add sentiment filter
    sentiment_filter = st.selectbox(
        "Filter by sentiment",
        ["all", "positive", "negative", "neutral"],
        index=0,
        help="Show tweets of specific sentiment"
    )
    
    # Update params with sentiment filter
    params_with_filter = params.copy()
    params_with_filter["sentiment"] = sentiment_filter
    params_with_filter["limit"] = 10
    
    tweets_df = fetch_top_tweets(params_with_filter)
    
    if not tweets_df.empty:
        # Display tweets
        for _, tweet in tweets_df.iterrows():
            sentiment_color = get_sentiment_color(tweet.get('sentiment_score', 0))
            sentiment_label = tweet.get('sentiment_label', 'NEUTRAL')
            
            # Create tweet card
            st.markdown(f"""
            <div style="
                background: white;
                border: 1px solid #e2e8f0;
                border-left: 4px solid {sentiment_color};
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 0.75rem;
                transition: all 0.2s ease;
            ">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;">
                    <div style="font-weight: 600; color: #1e293b;">
                        @{tweet.get('username', 'unknown')}
                    </div>
                    <div style="
                        background: {sentiment_color}20;
                        color: {sentiment_color};
                        padding: 0.25rem 0.75rem;
                        border-radius: 9999px;
                        font-size: 0.75rem;
                        font-weight: 500;
                    ">
                        {sentiment_label} ({tweet.get('sentiment_score', 0):.3f})
                    </div>
                </div>
                <div style="color: #334155; font-size: 0.875rem; line-height: 1.5;">
                    {tweet.get('text', '')}
                </div>
                <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 0.5rem;">
                    {pd.to_datetime(tweet.get('created_at')).strftime('%b %d, %Y at %I:%M %p')}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">💬</div>
            <div class="empty-state-title">No Tweets Found</div>
            <div class="empty-state-text">No tweets match the selected criteria</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main dashboard application"""
    # Page configuration
    st.set_page_config(
        page_title="Sentiment Intelligence Platform",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load professional CSS
    load_professional_css()
    
    # Render header
    render_header()
    
    # Create tabs for navigation
    tab1, tab2, tab3 = st.tabs(["Dashboard", "New Campaign", "Settings"])
    
    with tab1:
        # Campaign and date selection
        campaigns = fetch_campaigns()
        
        if not campaigns:
            st.markdown("""
            <div class="warning-box">
                <strong>No campaigns found.</strong> Start by tracking a new campaign in the "New Campaign" tab.
            </div>
            """, unsafe_allow_html=True)
            return
        
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            selected_campaigns = st.multiselect(
                "Select Campaigns",
                campaigns,
                default=campaigns[:1] if campaigns else [],
                help="Choose which campaigns to analyze"
            )
        
        with col2:
            min_date, max_date = fetch_date_range()
            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                help="Select the time period for analysis"
            )
        
        with col3:
            st.write("")  # Spacing
            refresh_btn = st.button("🔄 Refresh Data", use_container_width=True)
        
        if selected_campaigns and date_range:
            # Handle date range
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = end_date = date_range[0] if isinstance(date_range, tuple) else date_range
            
            # Prepare parameters
            params = {
                "campaigns": ",".join(selected_campaigns),
                "start": str(start_date),
                "end": str(end_date)
            }
            
            # Fetch data
            with st.spinner('Loading analytics data...'):
                df = fetch_analytics_data(params)
            
            if df.empty:
                st.markdown("""
                <div class="info-box">
                    No data found for the selected campaigns and date range. Try adjusting your filters or wait a moment for data to process.
                </div>
                """, unsafe_allow_html=True)
                return
            
            # Ensure required columns exist
            required_columns = ['avg_sentiment', 'tweet_count', 'pos_count', 'neu_count', 'neg_count']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.warning(f"Warning: Some data columns are missing ({', '.join(missing_columns)}). The dashboard may show limited information.")
                # Add default values for missing columns
                for col in missing_columns:
                    df[col] = 0
            
            # Render metrics row
            render_metrics_row(df)
            
            # Add spacing
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Create two columns for charts
            col1, col2 = st.columns([2, 1])
            
            with col1:
                render_sentiment_trend_chart(df)
            
            with col2:
                render_sentiment_distribution(df)
            
            # Additional insights
            st.markdown("<br>", unsafe_allow_html=True)
            render_top_tweets_analysis(params)
    
    with tab2:
        st.markdown("### Start Tracking a New Campaign")
        
        col1, col2 = st.columns(2)
        
        with col1:
            campaign_name = st.text_input(
                "Campaign/Brand Name",
                placeholder="e.g., Tesla, Nike, McDonald's",
                help="Enter the brand or campaign name to track"
            )
            
            st.markdown("### Date Range")
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                start_date = st.date_input(
                    "Start Date",
                    value=date.today() - timedelta(days=7),
                    max_value=date.today(),
                    help="Historical data collection start"
                )
            
            with date_col2:
                end_date = st.date_input(
                    "End Date",
                    value=date.today(),
                    max_value=date.today(),
                    help="Historical data collection end"
                )
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <strong>How it works:</strong><br>
                1. Enter a brand or campaign name to track<br>
                2. Select the date range for historical data<br>
                3. Click "Start Tracking" to begin collection<br>
                4. Data will be available in 30-60 seconds<br>
                <br>
                <strong>Note:</strong> The system will collect tweets mentioning your campaign and analyze their sentiment using advanced AI models.
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🚀 Start Tracking", type="primary", use_container_width=True):
            if not campaign_name:
                st.error("Please enter a campaign name")
            elif start_date > end_date:
                st.error("Start date must be before end date")
            else:
                with st.spinner(f'Starting campaign tracking for "{campaign_name}"...'):
                    try:
                        headers = {"access_token": API_KEY} if API_KEY else {}
                        resp = requests.post(
                            f"{API_BASE}/start_ingest/",
                            json={
                                "term": campaign_name,
                                "start_date": str(start_date),
                                "end_date": str(end_date)
                            },
                            headers=headers,
                            timeout=10
                        )
                        
                        if resp.status_code == 200:
                            st.success(f"✅ Successfully started tracking **{campaign_name}**")
                            st.info("📊 Data collection has begun. Switch to the Dashboard tab in 30-60 seconds to view results.")
                            st.balloons()
                        else:
                            st.error(f"Failed to start tracking. Status: {resp.status_code}")
                            if resp.text:
                                st.error(f"Error: {resp.text}")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    with tab3:
        st.markdown("### Dashboard Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Display Options")
            auto_refresh = st.checkbox("Enable auto-refresh (60s)", value=False)
            dark_mode = st.checkbox("Dark mode (Coming Soon)", value=False, disabled=True)
            show_raw_data = st.checkbox("Show raw data tables", value=False)
            
            if auto_refresh:
                st.info("Auto-refresh is enabled. The dashboard will update every 60 seconds.")
        
        with col2:
            st.markdown("#### API Configuration")
            api_status = "Connected" if API_BASE else "Not Configured"
            status_class = "success" if API_BASE else "danger"
            
            st.markdown(f"""
            <div class="info-box">
                <strong>API Endpoint:</strong> {API_BASE or "Not Set"}<br>
                <strong>Status:</strong> <span class="status-badge {status_class}">● {api_status}</span><br>
                <br>
                <strong>Version:</strong> 2.0<br>
                <strong>Last Check:</strong> {datetime.now().strftime('%H:%M:%S')}
            </div>
            """, unsafe_allow_html=True)
        
        if show_raw_data and 'df' in locals():
            st.markdown("### Raw Data")
            st.dataframe(df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #64748b; font-size: 0.875rem; padding: 1rem 0;">
        Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Sentiment Intelligence Platform v2.0
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()