import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random
# Removed unused imports to keep it clean, you can add them back if needed elsewhere
# import folium
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# import base64
# from io import BytesIO
# import tempfile
# import os

print("Generating data for Dashboard Visualizations...")

# --- Configuration for Data Loading ---
try:
    df_transactions = pd.read_csv("synthetic_transaction_data.csv")
    df_sentiment = pd.read_csv("sentiment.csv")
    df_journey = pd.read_csv("journey_entry.csv")
    df_after_sales = pd.read_csv("after_sales.csv")

    # Ensure date columns are datetime objects
    df_transactions['transaction_date'] = pd.to_datetime(df_transactions['transaction_date'])
    df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
    df_journey['stage_date'] = pd.to_datetime(df_journey['stage_date'])
    df_after_sales['interaction_date'] = pd.to_datetime(df_after_sales['interaction_date'])

    # --- NEW CHECK FOR REQUIRED COLUMNS IN DF_AFTER_SALES ---
    # These columns are essential for the dashboard functions, especially NPS drill-down
    required_after_sales_columns = [
        'agent_id', 'interaction_id', 'resolution_time_minutes',
        'interaction_date', 'resolution_status', 'nps_score', # nps_score is crucial for this task
        'issue_category', 'sentiment_category', 'interaction_type',
        'queue_time_seconds', 'sla_met', 'is_first_contact_resolution',
        'feedback_score_agent', 'customer_id' # customer_id is crucial for drill-down
    ]
    missing_columns = [col for col in required_after_sales_columns if col not in df_after_sales.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns in 'after_sales.csv': {missing_columns}. Please ensure your CSV file has these columns.")

except FileNotFoundError as e:
    print(f"Error: {e}. One or more required CSV files not found. Please ensure all data CSVs are in the same directory.")
    exit()
except KeyError as e:
    print(f"Data Loading Error: {e}")
    print("Please check the 'after_sales.csv' file to ensure it contains all necessary columns as per the dashboard functions.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    exit()

# --- Dashboard Plotting Functions ---

# --- 1. Real-Time Sentiment Dashboard ---
def plot_realtime_sentiment(df_sentiment):
    recent_sentiment = df_sentiment[df_sentiment['date'] > (datetime.now() - timedelta(days=30))]
    if recent_sentiment.empty:
        return go.Figure().to_html(full_html=False, include_plotlyjs='cdn')

    sentiment_trend = recent_sentiment.groupby(pd.Grouper(key='date', freq='D'))['sentiment_score'].mean().reset_index()
    sentiment_trend.columns = ['Date', 'Average Sentiment']

    fig_line = px.line(sentiment_trend, x='Date', y='Average Sentiment',
                        title='30-Day Sentiment Trend',
                        color_discrete_sequence=['#1f77b4'])
    fig_line.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        height=280,
        title_font_size=14,
        font_size=11
    )

    return fig_line.to_html(full_html=False, include_plotlyjs='cdn')

# --- 2. Call Center Agent Performance (Now a Horizontal Bar Chart) ---
def create_agent_performance_chart(df_after_sales):
    """
    Generates a horizontal bar chart showing the top 10 agents by calls handled.
    """
    agent_performance = df_after_sales.groupby('agent_id').agg(
        calls_handled=('interaction_id', 'count'),
        total_duration_minutes=('resolution_time_minutes', 'sum')
    ).reset_index()

    agent_performance['avg_duration'] = (
        agent_performance['total_duration_minutes'] / agent_performance['calls_handled']
    ).round(2)

    # Note: 'status' is simulated here, if you have real status data, use that.
    agent_performance['status'] = [random.choice(['Online', 'Offline', 'Busy']) for _ in range(len(agent_performance))]

    agent_performance = agent_performance.nlargest(10, 'calls_handled').sort_values(
        'calls_handled', ascending=True
    )

    fig = px.bar(
        agent_performance,
        x='calls_handled',
        y='agent_id',
        orientation='h',
        title='Top 10 Agent Performance by Calls Handled',
        labels={
            'calls_handled': 'Number of Calls Handled',
            'agent_id': 'Agent ID'
        },
        hover_data={
            'total_duration_minutes': ':.2f',
            'avg_duration': ':.2f',
            'status': True
        },
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
        title_font_size=16,
        xaxis_title="Number of Calls Handled",
        yaxis_title="Agent ID",
        yaxis_categoryorder='total ascending'
    )

    fig.update_yaxes(tickangle=-45)

    return fig.to_html(full_html=False, include_plotlyjs='cdn')


# --- 3. CX Health Snapshot Radar Chart ---
def plot_cx_radar_chart(df_after_sales, df_sentiment):
    customer_id_choices = df_after_sales['customer_id'].unique()
    if len(customer_id_choices) == 0:
        return go.Figure().to_html(full_html=False, include_plotlyjs='cdn')

    # Randomly select a customer for the snapshot
    customer_id = random.choice(customer_id_choices)
    cust_interactions = df_after_sales[df_after_sales['customer_id'] == customer_id]
    cust_sentiment = df_sentiment[df_sentiment['customer_id'] == customer_id]

    # Metrics calculation
    # Using the latest interaction date from df_after_sales as a reference for 'now'
    current_time_ref = df_after_sales['interaction_date'].max() if not df_after_sales.empty else datetime.now()

    recent_sentiment_score = cust_sentiment['sentiment_score'].mean() if not cust_sentiment.empty else 0
    num_open_issues = cust_interactions[cust_interactions['resolution_status'] != 'Resolved'].shape[0]
    time_since_last_interaction = (current_time_ref - cust_interactions['interaction_date'].max()).days if not cust_interactions.empty else 365
    recent_nps_score = cust_interactions['nps_score'].mean() if not cust_interactions.empty else 5
    product_ownership_flag = 1 if not cust_interactions.empty else 0 # Simple proxy for now

    # Normalize values for radar chart (0-1 scale)
    metrics = [
        (recent_sentiment_score + 1) / 2, # Assuming sentiment_score is -1 to 1
        1 - (min(num_open_issues, 5) / 5), # Cap open issues at 5 for normalization
        1 - (min(time_since_last_interaction, 90) / 90), # Cap days at 90 for normalization (longer is worse)
        recent_nps_score / 10, # NPS score is 0-10, normalize to 0-1
        product_ownership_flag
    ]
    categories = ['Sentiment', 'Issues', 'Recency', 'NPS', 'Product Ownership']

    fig = go.Figure(data=go.Scatterpolar(
        r=metrics,
        theta=categories,
        fill='toself',
        line_color='#2ca02c',
        fillcolor='rgba(44, 160, 44, 0.3)',
        name=f'Customer {customer_id}'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title=f'CX Health: {customer_id}',
        margin=dict(l=20, r=20, t=60, b=20),
        height=280,
        title_font_size=14
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')


# --- 4. Issues Categorization Treemap ---
def plot_issue_treemap(df_after_sales):
    df_issues = df_after_sales.groupby(['issue_category', 'sentiment_category']).size().reset_index(name='count')

    fig = px.treemap(df_issues,
                        path=[px.Constant("All Issues"), 'issue_category', 'sentiment_category'],
                        values='count',
                        color='sentiment_category',
                        color_discrete_map={
                            '(?)': 'lightgray', # Handle potential missing sentiment
                            'Positive': 'lightgreen',
                            'Neutral': 'lightgray',
                            'Negative': 'lightcoral'
                        },
                        title='Issues by Category & Sentiment')

    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=20),
        height=280,
        title_font_size=14
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')

# --- 5. Call Queue & SLA Metrics ---
def plot_queue_sla_metrics(df_after_sales):
    # Ensure date column is datetime objects
    df_after_sales['interaction_date'] = pd.to_datetime(df_after_sales['interaction_date'], errors='coerce')

    # Filter for Call/Chat interactions with valid dates and relevant columns
    df = df_after_sales[
        df_after_sales['interaction_type'].isin(["Call", "Chat"])
    ].dropna(subset=['interaction_date', 'queue_time_seconds', 'sla_met']).copy() # Use .copy()

    if df.empty:
        fig = go.Figure().update_layout(title_text='No Queue/SLA Data',
                                         margin=dict(l=40, r=40, t=60, b=40), height=450, title_font_size=14)
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Get the most recent date in the data
    max_date = df['interaction_date'].max()
    # Calculate start date for last 30 days
    start_date = max_date - timedelta(days=29) # Adjusted to be 30 full days including the start date
    df = df[df['interaction_date'].between(start_date, max_date)]

    # Aggregate daily metrics
    daily_metrics = df.groupby(pd.Grouper(key='interaction_date', freq='D')).agg(
        queue_calls=('queue_time_seconds', lambda x: (x > 0).sum()),
        sla_met=('sla_met', lambda x: (x == 'Yes').sum()),
        total_calls=('interaction_id', 'count')
    ).reset_index()

    # Avoid division by zero for SLA percentage
    daily_metrics['sla_percentage'] = daily_metrics.apply(
        lambda row: (row['sla_met'] / row['total_calls']) * 100 if row['total_calls'] > 0 else 0, axis=1
    ).round(2)

    # --- Plotting ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=daily_metrics['interaction_date'],
            y=daily_metrics['queue_calls'],
            name='Queue Calls (Line)',
            mode='lines+markers',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6),
            hovertemplate='Date: %{x|%b %d}<br>Queue Calls: %{y}<extra></extra>'
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Bar(
            x=daily_metrics['interaction_date'],
            y=daily_metrics['sla_percentage'],
            name='SLA Met (%)',
            marker_color='#2ca02c',
            opacity=0.6,
            hovertemplate='Date: %{x|%b %d}<br>SLA Met: %{y:.2f}%<extra></extra>'
        ),
        secondary_y=True
    )

    sla_target = 85
    fig.add_hline(
        y=sla_target,
        line_dash="dot",
        line_color="red",
        secondary_y=True,
        annotation_text=f"SLA Target: {sla_target}%",
        annotation_position="top left",
        annotation_font_size=10
    )

    fig.update_layout(
        title='<b>Call Queue & SLA Metrics (Last 30 Days)</b>',
        height=450,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified",
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_xaxes(
        title_text="Date",
        tickformat="%b %d",
        showgrid=True,
        gridcolor='lightgray'
    )

    fig.update_yaxes(
        title_text="Queue Calls",
        secondary_y=False,
        showgrid=True,
        gridcolor='lightgray'
    )

    fig.update_yaxes(
        title_text="SLA Met (%)",
        secondary_y=True,
        range=[0, 100],
        showgrid=False
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')


# --- 6. KPI Sparklines ---
def plot_kpi_sparklines(df_after_sales, df_sentiment):
    # Daily sentiment
    daily_sentiment = df_sentiment.groupby(pd.Grouper(key='date', freq='D'))['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['Date', 'Sentiment']

    # Daily SLA
    daily_sla = df_after_sales.groupby(pd.Grouper(key='interaction_date', freq='D')).agg(
        total=('interaction_id', 'count'),
        sla_met=('sla_met', lambda x: (x == 'Yes').sum())
    ).reset_index()
    daily_sla['SLA_Pct'] = daily_sla.apply(lambda row: (row['sla_met'] / row['total']) * 100 if row['total'] > 0 else 0, axis=1)

    # Daily FCR
    daily_fcr = df_after_sales.groupby(pd.Grouper(key='interaction_date', freq='D')).agg(
        total=('interaction_id', 'count'),
        fcr=('is_first_contact_resolution', lambda x: (x == 'Yes').sum())
    ).reset_index()
    daily_fcr['FCR_Pct'] = daily_fcr.apply(lambda row: (row['fcr'] / row['total']) * 100 if row['total'] > 0 else 0, axis=1)

    # Combine data using outer merge to ensure all dates are present
    kpi_data = pd.merge(daily_sentiment, daily_sla, left_on='Date', right_on='interaction_date', how='outer')
    kpi_data = pd.merge(kpi_data, daily_fcr, left_on='Date', right_on='interaction_date', how='outer', suffixes=('_sla', '_fcr'))
    
    # Clean up redundant date columns and sort
    kpi_data = kpi_data.drop(columns=['interaction_date_sla', 'interaction_date_fcr'], errors='ignore')
    kpi_data = kpi_data.sort_values('Date').set_index('Date')
    
    # Fill missing values (e.g., if no activity on a given day)
    kpi_data = kpi_data.ffill().tail(30).reset_index() # Use ffill() instead of fillna(method='ffill')

    if kpi_data.empty:
        return go.Figure().to_html(full_html=False, include_plotlyjs='cdn')

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=('Sentiment', 'SLA %', 'FCR %'))

    fig.add_trace(go.Scatter(x=kpi_data['Date'], y=kpi_data['Sentiment'],
                             mode='lines', name='Sentiment', line=dict(color='blue', width=2)),
                    row=1, col=1)
    fig.add_hline(y=0.3, line_dash="dash", line_color="green", row=1, col=1)

    fig.add_trace(go.Scatter(x=kpi_data['Date'], y=kpi_data['SLA_Pct'],
                             mode='lines', name='SLA %', line=dict(color='orange', width=2)),
                    row=2, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1)

    fig.add_trace(go.Scatter(x=kpi_data['Date'], y=kpi_data['FCR_Pct'],
                             mode='lines', name='FCR %', line=dict(color='green', width=2)),
                    row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="purple", row=3, col=1)

    fig.update_layout(
        height=280,
        showlegend=False,
        title_text='KPI Trends (30 Days)',
        margin=dict(l=40, r=40, t=60, b=40),
        title_font_size=14
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')

# --- Helper for NPS Calculation ---
def calculate_nps(scores):
    """
    Calculates NPS from a Series of 0-10 scores.
    NPS = % Promoters - % Detractors
    Promoters: 9-10
    Passives: 7-8
    Detractors: 0-6
    """
    if scores.empty or scores.isnull().all(): # Handle cases with no valid scores
        return 0

    promoters = (scores >= 9).sum()
    passives = ((scores >= 7) & (scores <= 8)).sum()
    detractors = (scores <= 6).sum()

    total = promoters + passives + detractors
    if total == 0:
        return 0 # Avoid division by zero if no responses

    percent_promoters = (promoters / total) * 100
    percent_detractors = (detractors / total) * 100

    return percent_promoters - percent_detractors

# --- 7. NPS Tracking (Modified for Drill-down) ---
def plot_nps_tracking(df_after_sales):
    """
    Plots the daily NPS score and identifies the lowest and highest NPS days
    for drill-down, returning relevant customer IDs and details.

    Args:
        df_after_sales (pd.DataFrame): DataFrame containing 'interaction_date',
                                       'customer_id', and 'nps_score' (0-10 raw score).

    Returns:
        tuple: (
            HTML string of the Plotly chart,
            dict: {
                'lowest_score': float,
                'lowest_date': datetime,
                'lowest_nps_customer_ids': list,
                'highest_score': float,
                'highest_date': datetime,
                'highest_nps_customer_ids': list
            }
        )
    """
    # Ensure 'interaction_date' is datetime type
    df_after_sales['interaction_date'] = pd.to_datetime(df_after_sales['interaction_date'], errors='coerce')

    # Filter out rows where NPS score is missing or invalid, as they cannot contribute to NPS
    df_after_sales_clean = df_after_sales.dropna(subset=['nps_score', 'customer_id']).copy() # Also drop NaNs in customer_id for drill-down
    if df_after_sales_clean.empty:
        fig = go.Figure().update_layout(title_text='Daily NPS Score (Last 30 Days) - No Data',
                                         margin=dict(l=40, r=40, t=60, b=40), height=280, title_font_size=14)
        return fig.to_html(full_html=False, include_plotlyjs='cdn'), {
            'lowest_score': None, 'lowest_date': None, 'lowest_nps_customer_ids': [],
            'highest_score': None, 'highest_date': None, 'highest_nps_customer_ids': []
        }

    # Get the most recent date in the data for filtering (consistent with current time context)
    latest_data_date = df_after_sales_clean['interaction_date'].max()
    start_date = latest_data_date - pd.Timedelta(days=29) # Covers 30 days including start and end

    # Filter data for the last 30 days
    df_last_30_days = df_after_sales_clean[
        (df_after_sales_clean['interaction_date'] >= start_date) &
        (df_after_sales_clean['interaction_date'] <= latest_data_date)
    ].copy()

    # Group by day and calculate NPS for each day
    # This ensures that even if there are gaps in daily data, the dates are aligned for plotting
    all_dates_in_period = pd.date_range(start=start_date, end=latest_data_date, freq='D')
    
    # Calculate daily NPS
    nps_data_raw = df_last_30_days.groupby(pd.Grouper(key='interaction_date', freq='D'))['nps_score'].apply(calculate_nps).reset_index()
    nps_data_raw.columns = ['Date', 'NPS']

    # Create a full date range and merge to ensure all 30 days are in the plot
    nps_data = pd.DataFrame({'Date': all_dates_in_period})
    nps_data = pd.merge(nps_data, nps_data_raw, on='Date', how='left')
    nps_data['NPS'] = nps_data['NPS'].fillna(0) # Fill days with no interactions with NPS 0 or previous day's NPS

    drilldown_info = {
        'lowest_score': None, 'lowest_date': None, 'lowest_nps_customer_ids': [],
        'highest_score': None, 'highest_date': None, 'highest_nps_customer_ids': []
    }

    if not nps_data.empty and 'customer_id' in df_last_30_days.columns:
        # Find Lowest NPS
        if not nps_data['NPS'].empty:
            lowest_nps_row = nps_data.loc[nps_data['NPS'].idxmin()]
            drilldown_info['lowest_score'] = lowest_nps_row['NPS']
            drilldown_info['lowest_date'] = lowest_nps_row['Date']

            # Get customers who were Detractors (score <= 6) on the lowest NPS date
            lowest_nps_customers_df = df_last_30_days[
                (df_last_30_days['interaction_date'].dt.date == drilldown_info['lowest_date'].date()) &
                (df_last_30_days['nps_score'] <= 6)
            ]
            drilldown_info['lowest_nps_customer_ids'] = lowest_nps_customers_df['customer_id'].unique().tolist()
            print(f"Lowest NPS Score: {drilldown_info['lowest_score']:.1f} on {drilldown_info['lowest_date'].strftime('%Y-%m-%d')}")
            print(f"Customers contributing to lowest NPS: {drilldown_info['lowest_nps_customer_ids']}")

        # Find Highest NPS
        if not nps_data['NPS'].empty:
            highest_nps_row = nps_data.loc[nps_data['NPS'].idxmax()]
            drilldown_info['highest_score'] = highest_nps_row['NPS']
            drilldown_info['highest_date'] = highest_nps_row['Date']

            # Get customers who were Promoters (score >= 9) on the highest NPS date
            highest_nps_customers_df = df_last_30_days[
                (df_last_30_days['interaction_date'].dt.date == drilldown_info['highest_date'].date()) &
                (df_last_30_days['nps_score'] >= 9)
            ]
            drilldown_info['highest_nps_customer_ids'] = highest_nps_customers_df['customer_id'].unique().tolist()
            print(f"Highest NPS Score: {drilldown_info['highest_score']:.1f} on {drilldown_info['highest_date'].strftime('%Y-%m-%d')}")
            print(f"Customers contributing to highest NPS: {drilldown_info['highest_nps_customer_ids']}")
    else:
        print("Warning: 'customer_id' column not found or NPS data is empty. Cannot drill down into individual customers.")


    # --- Plotting the NPS Trend ---
    fig = px.line(nps_data, x='Date', y='NPS',
                    title='Daily NPS Score (Last 30 Days)',
                    color_discrete_sequence=['#2ca02c'])
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        height=280,
        title_font_size=14
    )
    fig.update_yaxes(range=[-100, 100]) # NPS ranges from -100 to 100

    return fig.to_html(full_html=False, include_plotlyjs='cdn'), drilldown_info

# --- New Function for NPS Drill-down Customer Activity ---
def get_nps_drilldown_html(df_after_sales, drilldown_info, type_of_nps="lowest"):
    """
    Generates an HTML string for NPS drill-down (either lowest or highest).

    Args:
        df_after_sales (pd.DataFrame): The main DataFrame containing all customer interactions.
        drilldown_info (dict): Dictionary containing 'lowest_score', 'lowest_date',
                               'lowest_nps_customer_ids', 'highest_score', 'highest_date',
                               'highest_nps_customer_ids'.
        type_of_nps (str): 'lowest' or 'highest' to indicate which drill-down to generate.

    Returns:
        str: HTML string of customer activities and details for the specified NPS type.
    """
    score_key = f"{type_of_nps}_score"
    date_key = f"{type_of_nps}_date"
    customers_key = f"{type_of_nps}_nps_customer_ids"

    nps_score = drilldown_info.get(score_key)
    nps_date = drilldown_info.get(date_key)
    customer_ids = drilldown_info.get(customers_key, [])

    title_text_prefix = "NPS Drill-down: "
    if type_of_nps == "lowest":
        title_text_suffix = "Lowest Score Contributors"
        score_color = "red"
        customer_label = "Detractors"
    else: # type_of_nps == "highest"
        title_text_suffix = "Highest Score Promoters"
        score_color = "green"
        customer_label = "Promoters"
    
    # Handle cases where no valid score/date or customers were found
    if nps_score is None or nps_date is None:
        return f"""
        <div class="customer-activity-list">
            <h5>{title_text_prefix}{title_text_suffix}</h5>
            <p>No valid {type_of_nps} NPS score found for drill-down.</p>
        </div>
        """

    html_output = f"""
    <div class="customer-activity-list">
        <h5>NPS Score: <span style="color: {score_color}; font-weight: bold;">{nps_score:.1f}</span> on <span style="font-weight: bold;">{nps_date.strftime('%Y-%m-%d')}</span></h5>
        <h6>Customers ({customer_label}) on this day:</h6>
        <ul>
    """

    if not customer_ids:
        html_output += f"<li>No specific {customer_label.lower()} identified for this day.</li>"
    else:
        for cust_id in customer_ids:
            html_output += f"<li><strong>{cust_id}</strong></li>"
    html_output += "</ul>"

    html_output += "<h6>Last 6 Months of Activity for these Customers:</h6>"

    # Determine the end date for the 6-month activity window using the latest data date
    latest_data_date_overall = df_after_sales['interaction_date'].max() if not df_after_sales.empty else datetime.now()
    six_months_ago = latest_data_date_overall - pd.Timedelta(days=180) # Approximately 6 months

    found_activity = False
    if customer_ids: # Only proceed if there are customers to check
        for customer_id in customer_ids:
            customer_activities = df_after_sales[
                (df_after_sales['customer_id'] == customer_id) &
                (df_after_sales['interaction_date'] >= six_months_ago)
            ].sort_values(by='interaction_date', ascending=False)

            if not customer_activities.empty:
                found_activity = True
                html_output += f"<p><strong>Customer ID: {customer_id}</strong></p><ul>"
                for _, row in customer_activities.iterrows():
                    # Display relevant activity details from df_after_sales
                    html_output += f"<li>Date: {row['interaction_date'].strftime('%Y-%m-%d')}, " \
                                   f"Type: {row.get('interaction_type', 'N/A')}, " \
                                   f"Issue: {row.get('issue_category', 'N/A')}, " \
                                   f"Sentiment: {row.get('sentiment_category', 'N/A')}, " \
                                   f"Resolution: {row.get('resolution_status', 'N/A')}</li>"
                html_output += "</ul>"
    
    if not found_activity: # If no activities were found for ANY of the identified customers
        html_output += "<p>No activity found for these customers in the last 6 months.</p>"

    html_output += "</div>"
    return html_output


# --- 8. Staff Feedback ---
def plot_staff_feedback(df_after_sales):
    # Ensure 'interaction_date' is datetime type
    df_after_sales['interaction_date'] = pd.to_datetime(df_after_sales['interaction_date'])

    # Get the most recent date in the data
    latest_date = df_after_sales['interaction_date'].max()

    # Calculate the date 30 days prior to the latest date
    start_date = latest_date - pd.Timedelta(days=29) # Adjusted to be 30 full days including the start date

    # Filter data for the last 30 days
    df_last_30_days = df_after_sales[(df_after_sales['interaction_date'] >= start_date) & \
                                     (df_after_sales['interaction_date'] <= latest_date)]

    feedback_counts = df_last_30_days.groupby(['agent_id', 'feedback_score_agent']).size().unstack(fill_value=0)

    if feedback_counts.empty:
        fig = go.Figure(layout=go.Layout(title="No Staff Feedback Data for Last 30 Days"))
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Get top 8 agents based on total feedback received in the last 30 days
    # This avoids issues if an agent only has feedback for a single score category
    if 'agent_id' in df_last_30_days.columns:
        agent_feedback_total = df_last_30_days['agent_id'].value_counts()
        if not agent_feedback_total.empty:
            top_agents = agent_feedback_total.nlargest(8).index
            feedback_counts = feedback_counts.loc[feedback_counts.index.isin(top_agents)]
            feedback_counts = feedback_counts.sort_values(by=feedback_counts.columns[0], ascending=True) # Sort for better viz

    fig = px.bar(feedback_counts,
                    x=feedback_counts.index,
                    y=feedback_counts.columns,
                    barmode='group',
                    title='Agent Feedback Distribution (Last 30 Days)',
                    labels={'agent_id': 'Agent', 'value': 'Count'},
                    color_discrete_sequence=px.colors.qualitative.Set2)

    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        height=280,
        title_font_size=14,
        font_size=10
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')


# --- 9. Campaign KPIs ---
def plot_campaign_kpis(df_journey):
    if df_journey.empty:
        fig = go.Figure().update_layout(title_text="No Campaign Data Available",
                                         margin=dict(l=20, r=20, t=60, b=20), height=280, title_font_size=14)
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    total_sent = df_journey[df_journey['stage'] == 'sent'].shape[0]
    total_opens = df_journey[df_journey['campaign_open'] == 'Yes'].shape[0]
    total_clicks = df_journey[df_journey['campaign_click'] == 'Yes'].shape[0]
    total_conversions = df_journey[df_journey['conversion_flag'] == 'Yes'].shape[0]

    open_rate = (total_opens / total_sent) * 100 if total_sent > 0 else 0
    click_rate = (total_clicks / total_opens) * 100 if total_opens > 0 else 0 # Click rate is usually clicks per open
    conversion_rate = (total_conversions / total_sent) * 100 if total_sent > 0 else 0 # Conversion per sent

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
        subplot_titles=('Open Rate', 'Click Rate', 'Conversion')
    )

    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=open_rate,
        title={'text': "Open %"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "#636efa"},
               'steps': [
                   {'range': [0, 40], 'color': "lightcoral"},
                   {'range': [40, 70], 'color': "lightgray"},
                   {'range': [70, 100], 'color': "lightgreen"}],
               }), row=1, col=1)

    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=click_rate,
        title={'text': "Click %"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "#EF553B"},
               'steps': [
                   {'range': [0, 10], 'color': "lightcoral"},
                   {'range': [10, 30], 'color': "lightgray"},
                   {'range': [30, 100], 'color': "lightgreen"}],
               }), row=1, col=2)

    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=conversion_rate,
        title={'text': "Conv %"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "#00cc96"},
               'steps': [
                   {'range': [0, 5], 'color': "lightcoral"},
                   {'range': [5, 15], 'color': "lightgray"},
                   {'range': [15, 100], 'color': "lightgreen"}],
               }), row=1, col=3)

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
        title_text="Campaign Metrics",
        title_font_size=14
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')

# --- Generate HTML Dashboard ---
def generate_dashboard_html(plots):
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CX Analytics Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f8f9fa;
                color: #333;
                margin: 0;
                padding: 0;
            }}
            .dashboard-container {{
                padding: 15px;
                max-width: 1400px;
                margin: 0 auto;
            }}
            .dashboard-title {{
                text-align: center;
                color: #007bff;
                margin-bottom: 25px;
                font-weight: 600;
                font-size: 2.2rem;
            }}
            .dashboard-row {{
                margin-bottom: 15px;
            }}
            .dashboard-card {{
                border: none;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                margin-bottom: 15px;
                background-color: #fff;
                height: 350px; /* Fixed height for most cards */
                display: flex;
                flex-direction: column;
            }}
            .dashboard-card.nps-drilldown-card {{
                height: auto; /* Allow this card to expand based on content */
                min-height: 350px;
            }}
            .card-header {{
                background: linear-gradient(135deg, #007bff, #0056b3);
                color: white;
                padding: 12px 20px;
                font-size: 1rem;
                font-weight: 600;
                border-radius: 10px 10px 0 0;
                border-bottom: none;
                flex-shrink: 0;
            }}
            .card-body {{
                padding: 15px;
                flex-grow: 1;
                display: flex;
                flex-direction: column; /* Changed to column for vertical stacking of elements */
                align-items: flex-start; /* Align content to the start */
                justify-content: flex-start; /* Align content to the start */
                overflow: hidden;
            }}
            .plot-container {{
                width: 100%;
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden; /* Ensure plots don't overflow their container */
            }}
            .plot-container > div {{
                width: 100% !important;
                height: 100% !important;
            }}
            /* Specific style for NPS drilldown content */
            .customer-activity-list {{
                width: 100%;
                overflow-y: auto; /* Make content scrollable if it exceeds card height */
                padding: 10px;
                font-size: 0.9em;
                margin-top: 5px; /* Space from header */
            }}
            .customer-activity-list h5, .customer-activity-list h6 {{
                font-size: 1.1em;
                color: #007bff;
                margin-bottom: 5px;
            }}
            .customer-activity-list ul {{
                list-style-type: none;
                padding-left: 10px; /* Indent for readability */
                margin-bottom: 5px;
            }}
            .customer-activity-list ul li {{
                margin-bottom: 3px;
                border-bottom: 1px dotted #eee;
                padding-bottom: 2px;
            }}
            .customer-activity-list ul li:last-child {{
                border-bottom: none;
            }}
            .customer-activity-list p strong {{
                 color: #0056b3; /* Darker blue for customer IDs */
            }}


            /* Responsive adjustments */
            @media (max-width: 768px) {{
                .dashboard-card {{
                    height: 300px;
                }}
                .dashboard-card.nps-drilldown-card {{
                    height: auto;
                    min-height: 300px;
                }}
                .dashboard-title {{
                    font-size: 1.8rem;
                }}
                .card-header {{
                    font-size: 0.9rem;
                    padding: 10px 15px;
                }}
            }}
            @media (max-width: 576px) {{
                .dashboard-container {{
                    padding: 10px;
                }}
                .dashboard-card {{
                    height: 280px;
                    margin-bottom: 10px;
                }}
                .dashboard-card.nps-drilldown-card {{
                    height: auto;
                    min-height: 280px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="dashboard-container">
            <h1 class="dashboard-title">Customer Experience Analytics Dashboard</h1>

            <div class="row dashboard-row">
                <div class="col-lg-4 col-md-6 col-sm-12">
                    <div class="dashboard-card">
                        <div class="card-header">📊 Real-Time Sentiment</div>
                        <div class="card-body">
                            <div class="plot-container">
                                {plot_html_sentiment}
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-lg-4 col-md-6 col-sm-12">
                    <div class="dashboard-card">
                        <div class="card-header">👥 Agent Performance</div>
                        <div class="card-body">
                            <div class="plot-container">
                                {agent_table_html}
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-lg-4 col-md-6 col-sm-12">
                    <div class="dashboard-card">
                        <div class="card-header">🎯 CX Health Snapshot</div>
                        <div class="card-body">
                            <div class="plot-container">
                                {plot_html_cx_radar}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row dashboard-row">
                <div class="col-lg-4 col-md-6 col-sm-12">
                    <div class="dashboard-card">
                        <div class="card-header">🗂️ Issues Categorization</div>
                        <div class="card-body">
                            <div class="plot-container">
                                {plot_html_issue_treemap}
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-lg-4 col-md-6 col-sm-12">
                    <div class="dashboard-card">
                        <div class="card-header">⏱️ Queue & SLA Metrics</div>
                        <div class="card-body">
                            <div class="plot-container">
                                {plot_html_queue_sla}
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-lg-4 col-md-6 col-sm-12">
                    <div class="dashboard-card">
                        <div class="card-header">📈 KPI Trends</div>
                        <div class="card-body">
                            <div class="plot-container">
                                {plot_html_kpi_sparklines}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row dashboard-row">
                <div class="col-lg-4 col-md-6 col-sm-12">
                    <div class="dashboard-card">
                        <div class="card-header">⭐ NPS Tracking</div>
                        <div class="card-body">
                            <div class="plot-container">
                                {plot_html_nps_tracking}
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-lg-4 col-md-6 col-sm-12">
                    <div class="dashboard-card nps-drilldown-card">
                        <div class="card-header">🔍 NPS Drill-down: Lowest Score Contributors</div>
                        <div class="card-body">
                            {lowest_nps_drilldown_html}
                        </div>
                    </div>
                </div>
                <div class="col-lg-4 col-md-12 col-sm-12">
                    <div class="dashboard-card nps-drilldown-card">
                        <div class="card-header">📈 NPS Drill-down: Highest Score Promoters</div>
                        <div class="card-body">
                            {highest_nps_drilldown_html}
                        </div>
                    </div>
                </div>
            </div>

            <div class="row dashboard-row">
                <div class="col-lg-6 col-md-6 col-sm-12">
                    <div class="dashboard-card">
                        <div class="card-header">🧑‍💻 Staff Feedback</div>
                        <div class="card-body">
                            <div class="plot-container">
                                {plot_html_staff_feedback}
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-lg-6 col-md-6 col-sm-12">
                    <div class="dashboard-card">
                        <div class="card-header">🎯 Campaign KPIs</div>
                        <div class="card-body">
                            <div class="plot-container">
                                {plot_html_campaign_kpis}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // Ensure proper responsive behavior for Plotly charts
            window.addEventListener('resize', function() {{
                if (window.Plotly) {{
                    // Resize all Plotly plots on window resize for responsiveness
                    var plots = document.querySelectorAll('.plot-container > div.js-plotly-plot');
                    plots.forEach(function(plotDiv) {{
                        Plotly.relayout(plotDiv.id, {{autosize: true}});
                    }});
                }}
            }});
        </script>
    </body>
    </html>
    """
    return html_template.format(**plots)

# --- Main Execution Block ---
if __name__ == "__main__":
    # Generate all plots
    plot_html_sentiment = plot_realtime_sentiment(df_sentiment)
    agent_table_html = create_agent_performance_chart(df_after_sales)
    plot_html_cx_radar = plot_cx_radar_chart(df_after_sales, df_sentiment)
    plot_html_issue_treemap = plot_issue_treemap(df_after_sales)
    plot_html_queue_sla = plot_queue_sla_metrics(df_after_sales)
    plot_html_kpi_sparklines = plot_kpi_sparklines(df_after_sales, df_sentiment)

    # Call NPS tracking, which now returns the plot HTML and the drill-down info dictionary
    plot_html_nps_tracking, nps_drilldown_details = plot_nps_tracking(df_after_sales)

    # Generate HTML for lowest NPS drill-down
    lowest_nps_drilldown_html = get_nps_drilldown_html(df_after_sales, nps_drilldown_details, type_of_nps="lowest")

    # Generate HTML for highest NPS drill-down
    highest_nps_drilldown_html = get_nps_drilldown_html(df_after_sales, nps_drilldown_details, type_of_nps="highest")

    plot_html_staff_feedback = plot_staff_feedback(df_after_sales)
    plot_html_campaign_kpis = plot_campaign_kpis(df_journey)


    plots_dict = {
        'plot_html_sentiment': plot_html_sentiment,
        'agent_table_html': agent_table_html,
        'plot_html_cx_radar': plot_html_cx_radar,
        'plot_html_issue_treemap': plot_html_issue_treemap,
        'plot_html_queue_sla': plot_html_queue_sla,
        'plot_html_kpi_sparklines': plot_html_kpi_sparklines,
        'plot_html_nps_tracking': plot_html_nps_tracking,
        'lowest_nps_drilldown_html': lowest_nps_drilldown_html,
        'highest_nps_drilldown_html': highest_nps_drilldown_html,
        'plot_html_staff_feedback': plot_html_staff_feedback,
        'plot_html_campaign_kpis': plot_html_campaign_kpis,
    }

    html_output = generate_dashboard_html(plots_dict)

    # Save to an HTML file
    dashboard_filename = 'cx_dashboard.html'
    with open(dashboard_filename, 'w', encoding='utf-8') as f:
        f.write(html_output)

    print(f"Dashboard '{dashboard_filename}' generated successfully!")
    print("\nOpen the HTML file in your web browser to view the updated dashboard with NPS drill-downs.")

# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from datetime import datetime, timedelta
# import random
# import folium
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# import base64
# from io import BytesIO
# import tempfile
# import os

# print("Generating data for Dashboard Visualizations...")

# # --- Configuration for Data Generation ---
# try:
#     df_transactions = pd.read_csv("synthetic_transaction_data.csv")
#     df_sentiment = pd.read_csv("sentiment.csv")
#     df_journey = pd.read_csv("journey_entry.csv")
#     df_after_sales = pd.read_csv("after_sales.csv")

#     # Ensure date columns are datetime objects
#     df_transactions['transaction_date'] = pd.to_datetime(df_transactions['transaction_date'])
#     df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
#     df_journey['stage_date'] = pd.to_datetime(df_journey['stage_date'])
#     df_after_sales['interaction_date'] = pd.to_datetime(df_after_sales['interaction_date'])

# except FileNotFoundError as e:
#     print(f"Error: {e}. One or more required CSV files not found. Please run previous data generation scripts first.")
#     exit()

# # --- 1. Real-Time Sentiment Dashboard ---
# def plot_realtime_sentiment(df_sentiment):
#     recent_sentiment = df_sentiment[df_sentiment['date'] > (datetime.now() - timedelta(days=30))]
#     if recent_sentiment.empty:
#         return go.Figure().to_html(full_html=False, include_plotlyjs='cdn')

#     sentiment_trend = recent_sentiment.groupby(pd.Grouper(key='date', freq='D'))['sentiment_score'].mean().reset_index()
#     sentiment_trend.columns = ['Date', 'Average Sentiment']

#     fig_line = px.line(sentiment_trend, x='Date', y='Average Sentiment',
#                        title='30-Day Sentiment Trend',
#                        color_discrete_sequence=['#1f77b4'])
#     fig_line.update_layout(
#         margin=dict(l=40, r=40, t=60, b=40), 
#         height=280,
#         title_font_size=14,
#         font_size=11
#     )

#     return fig_line.to_html(full_html=False, include_plotlyjs='cdn')

# # --- 2. Call Center Agent Performance ---
# # def create_agent_performance_table(df_after_sales):
# #     agent_performance = df_after_sales.groupby('agent_id').agg(
# #         calls_handled=('interaction_id', 'count'),
# #         total_duration_minutes=('resolution_time_minutes', 'sum')
# #     ).reset_index()
# #     agent_performance['avg_duration'] = (agent_performance['total_duration_minutes'] / agent_performance['calls_handled']).round(2)
# #     agent_performance['status'] = [random.choice(['Online', 'Offline', 'Busy']) for _ in range(len(agent_performance))]

# #     # Limit to top 8 agents for better display
# #     agent_performance = agent_performance.nlargest(8, 'calls_handled')
    
# #     fig = go.Figure(data=[go.Table(
# #         header=dict(values=['Agent', 'Status', 'Calls', 'Avg Duration'],
# #                    fill_color='#007bff',
# #                    font=dict(color='white', size=12),
# #                    align='center'),
# #         cells=dict(values=[agent_performance.agent_id, 
# #                           agent_performance.status,
# #                           agent_performance.calls_handled, 
# #                           agent_performance.avg_duration],
# #                   fill_color='lightgrey',
# #                   align='center',
# #                   font_size=10))
# #     ])
    
# #     fig.update_layout(
# #         margin=dict(l=20, r=20, t=40, b=20),
# #         height=280,
# #         title='Agent Performance',
# #         title_font_size=14
# #     )
    
# #     return fig.to_html(full_html=False, include_plotlyjs='cdn')

# def create_agent_performance_table(df_after_sales):
#     """
#     Generates a horizontal bar chart showing the top 10 agents by calls handled.

#     Args:
#         df_after_sales (pd.DataFrame): DataFrame containing 'agent_id',
#                                        'interaction_id', and 'resolution_time_minutes'.

#     Returns:
#         str: HTML string of the Plotly horizontal bar chart.
#     """
#     # Group by agent_id and aggregate performance metrics
#     agent_performance = df_after_sales.groupby('agent_id').agg(
#         calls_handled=('interaction_id', 'count'),
#         total_duration_minutes=('resolution_time_minutes', 'sum')
#     ).reset_index()

#     # Calculate average duration per call
#     agent_performance['avg_duration'] = (
#         agent_performance['total_duration_minutes'] / agent_performance['calls_handled']
#     ).round(2)

#     # Add a 'status' column (optional, as it's less relevant for a bar chart directly)
#     agent_performance['status'] = [random.choice(['Online', 'Offline', 'Busy']) for _ in range(len(agent_performance))]

#     # Limit to top 10 agents based on 'calls_handled'
#     # Sort for better visualization in the bar chart (highest at the top)
#     agent_performance = agent_performance.nlargest(10, 'calls_handled').sort_values(
#         'calls_handled', ascending=True
#     )

#     # Create the horizontal bar chart
#     fig = px.bar(
#         agent_performance,
#         x='calls_handled',      # Metric on the x-axis
#         y='agent_id',           # Agent IDs on the y-axis
#         orientation='h',        # Make it a horizontal bar chart
#         title='Top 10 Agent Performance by Calls Handled', # Updated title
#         labels={
#             'calls_handled': 'Number of Calls Handled',
#             'agent_id': 'Agent ID'
#         },
#         hover_data={
#             'total_duration_minutes': ':.2f', # Show total duration in hover
#             'avg_duration': ':.2f',           # Show average duration in hover
#             'status': True                    # Show status in hover
#         },
#         color_discrete_sequence=px.colors.qualitative.Plotly # Choose a nice color sequence
#     )

#     # Update layout for better appearance
#     fig.update_layout(
#         margin=dict(l=40, r=40, t=60, b=40),
#         height=400, # Increased height for 10 bars
#         title_font_size=16,
#         xaxis_title="Number of Calls Handled",
#         yaxis_title="Agent ID",
#         yaxis_categoryorder='total ascending' # Ensure the bars are ordered correctly (smallest at bottom, largest at top)
#     )

#     # Make y-axis labels readable if there are many agents
#     fig.update_yaxes(tickangle=-45)

#     return fig.to_html(full_html=False, include_plotlyjs='cdn')

# # --- 3. CX Health Snapshot Radar Chart ---
# def plot_cx_radar_chart(df_after_sales, df_sentiment):
#     customer_id_choices = df_after_sales['customer_id'].unique()
#     if len(customer_id_choices) == 0:
#         return go.Figure().to_html(full_html=False, include_plotlyjs='cdn')

#     customer_id = random.choice(customer_id_choices)
#     cust_interactions = df_after_sales[df_after_sales['customer_id'] == customer_id]
#     cust_sentiment = df_sentiment[df_sentiment['customer_id'] == customer_id]

#     recent_sentiment_score = cust_sentiment['sentiment_score'].mean() if not cust_sentiment.empty else 0
#     num_open_issues = cust_interactions[cust_interactions['resolution_status'] != 'Resolved'].shape[0]
#     time_since_last_interaction = (datetime.now() - cust_interactions['interaction_date'].max()).days if not cust_interactions.empty else 365
#     recent_nps_score = cust_interactions['nps_score'].mean() if not cust_interactions.empty else 5
#     product_ownership_flag = 1 if not cust_interactions.empty else 0

#     # Normalize values for radar chart (0-1 scale)
#     metrics = [
#         (recent_sentiment_score + 1) / 2,
#         1 - (min(num_open_issues, 5) / 5),
#         1 - (min(time_since_last_interaction, 90) / 90),
#         recent_nps_score / 10,
#         product_ownership_flag
#     ]
#     categories = ['Sentiment', 'Issues', 'NPS', 'Product']

#     fig = go.Figure(data=go.Scatterpolar(
#         r=metrics,
#         theta=categories,
#         fill='toself',
#         line_color='#2ca02c',
#         fillcolor='rgba(44, 160, 44, 0.3)',
#         name=f'Customer {customer_id}'
#     ))
    
#     fig.update_layout(
#         polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
#         showlegend=False,
#         title=f'CX Health: {customer_id}',
#         margin=dict(l=20, r=20, t=60, b=20), 
#         height=280,
#         title_font_size=14
#     )
    
#     return fig.to_html(full_html=False, include_plotlyjs='cdn')

# # --- 4. Issues Categorization Treemap ---
# def plot_issue_treemap(df_after_sales):
#     df_issues = df_after_sales.groupby(['issue_category', 'sentiment_category']).size().reset_index(name='count')
    
#     fig = px.treemap(df_issues, 
#                      path=[px.Constant("All Issues"), 'issue_category', 'sentiment_category'],
#                      values='count',
#                      color='sentiment_category',
#                      color_discrete_map={
#                          '(?)': 'lightgray',
#                          'Positive': 'lightgreen',
#                          'Neutral': 'lightgray',
#                          'Negative': 'lightcoral'
#                      },
#                      title='Issues by Category & Sentiment')
    
#     fig.update_layout(
#         margin=dict(l=20, r=20, t=60, b=20), 
#         height=280,
#         title_font_size=14
#     )
    
#     return fig.to_html(full_html=False, include_plotlyjs='cdn')

# # # --- 5. Call Queue & SLA Metrics ---
# import pandas as pd
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from datetime import timedelta

# def plot_queue_sla_metrics(df_after_sales):
#     # Convert date column
#     df_after_sales['interaction_date'] = pd.to_datetime(df_after_sales['interaction_date'], errors='coerce')

#     # Filter for Call/Chat interactions with valid dates
#     df = df_after_sales[
#         df_after_sales['interaction_type'].isin(["Call", "Chat"])
#     ].dropna(subset=['interaction_date']).copy()

#     # Last 30 days filter
#     max_date = df['interaction_date'].max()
#     start_date = max_date - timedelta(days=29)
#     df = df[df['interaction_date'].between(start_date, max_date)]

#     # Aggregate daily metrics
#     daily_metrics = df.groupby(pd.Grouper(key='interaction_date', freq='D')).agg(
#         queue_calls=('queue_time_seconds', lambda x: (x > 0).sum()),
#         sla_met=('sla_met', lambda x: (x == 'Yes').sum()),
#         total_calls=('interaction_id', 'count')
#     ).reset_index()

#     daily_metrics['sla_percentage'] = ((daily_metrics['sla_met'] / daily_metrics['total_calls']) * 100).fillna(0).round(2)

#     # --- Plotting ---
#     fig = make_subplots(specs=[[{"secondary_y": True}]])

#     # Left Y-axis: Queue Calls (Line)
#     fig.add_trace(
#         go.Scatter(
#             x=daily_metrics['interaction_date'],
#             y=daily_metrics['queue_calls'],
#             name='Queue Calls (Line)',
#             mode='lines+markers',
#             line=dict(color='#1f77b4', width=2),
#             marker=dict(size=6),
#             hovertemplate='Date: %{x|%b %d}<br>Queue Calls: %{y}<extra></extra>'
#         ),
#         secondary_y=False
#     )

#     # Right Y-axis: SLA % (Bar)
#     fig.add_trace(
#         go.Bar(
#             x=daily_metrics['interaction_date'],
#             y=daily_metrics['sla_percentage'],
#             name='SLA Met (%)',
#             marker_color='#2ca02c',
#             opacity=0.6,
#             hovertemplate='Date: %{x|%b %d}<br>SLA Met: %{y:.2f}%<extra></extra>'
#         ),
#         secondary_y=True
#     )

#     # SLA Target Line (on right Y-axis)
#     sla_target = 85
#     fig.add_hline(
#         y=sla_target,
#         line_dash="dot",
#         line_color="red",
#         secondary_y=True,
#         annotation_text=f"SLA Target: {sla_target}%",
#         annotation_position="top left",
#         annotation_font_size=10
#     )

#     # Layout and axes
#     fig.update_layout(
#         title='<b>Call Queue & SLA Metrics (Last 30 Days)</b>',
#         height=450,
#         margin=dict(l=40, r=40, t=60, b=40),
#         hovermode="x unified",
#         plot_bgcolor='white',
#         paper_bgcolor='white',
#         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
#     )

#     fig.update_xaxes(
#         title_text="Date",
#         tickformat="%b %d",
#         showgrid=True,
#         gridcolor='lightgray'
#     )

#     fig.update_yaxes(
#         title_text="Queue Calls",
#         secondary_y=False,
#         showgrid=True,
#         gridcolor='lightgray'
#     )

#     fig.update_yaxes(
#         title_text="SLA Met (%)",
#         secondary_y=True,
#         range=[0, 100],
#         showgrid=False
#     )

#     return fig.to_html(full_html=False, include_plotlyjs='cdn')


# # --- 6. KPI Sparklines ---
# def plot_kpi_sparklines(df_after_sales, df_sentiment):
#     # Daily sentiment
#     daily_sentiment = df_sentiment.groupby(pd.Grouper(key='date', freq='D'))['sentiment_score'].mean().reset_index()
#     daily_sentiment.columns = ['Date', 'Sentiment']
    
#     # Daily SLA
#     daily_sla = df_after_sales.groupby(pd.Grouper(key='interaction_date', freq='D')).agg(
#         total=('interaction_id', 'count'),
#         sla_met=('sla_met', lambda x: (x == 'Yes').sum())
#     ).reset_index()
#     daily_sla['SLA_Pct'] = ((daily_sla['sla_met'] / daily_sla['total']) * 100).fillna(0)

#     # Daily FCR
#     daily_fcr = df_after_sales.groupby(pd.Grouper(key='interaction_date', freq='D')).agg(
#         total=('interaction_id', 'count'),
#         fcr=('is_first_contact_resolution', lambda x: (x == 'Yes').sum())
#     ).reset_index()
#     daily_fcr['FCR_Pct'] = ((daily_fcr['fcr'] / daily_fcr['total']) * 100).fillna(0)

#     # Combine data
#     kpi_data = daily_sentiment.merge(daily_sla, left_on='Date', right_on='interaction_date', how='outer')
#     kpi_data = kpi_data.merge(daily_fcr, left_on='Date', right_on='interaction_date', how='outer')
#     kpi_data = kpi_data.sort_values('Date').fillna(method='ffill').tail(30)
    
#     if kpi_data.empty:
#         return go.Figure().to_html(full_html=False, include_plotlyjs='cdn')

#     fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
#                        subplot_titles=('Sentiment', 'SLA %', 'FCR %'))

#     fig.add_trace(go.Scatter(x=kpi_data['Date'], y=kpi_data['Sentiment'], 
#                             mode='lines', name='Sentiment', line=dict(color='blue', width=2)), 
#                  row=1, col=1)
#     fig.add_hline(y=0.3, line_dash="dash", line_color="green", row=1, col=1)

#     fig.add_trace(go.Scatter(x=kpi_data['Date'], y=kpi_data['SLA_Pct'], 
#                             mode='lines', name='SLA %', line=dict(color='orange', width=2)), 
#                  row=2, col=1)
#     fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1)

#     fig.add_trace(go.Scatter(x=kpi_data['Date'], y=kpi_data['FCR_Pct'], 
#                             mode='lines', name='FCR %', line=dict(color='green', width=2)), 
#                  row=3, col=1)
#     fig.add_hline(y=70, line_dash="dash", line_color="purple", row=3, col=1)

#     fig.update_layout(
#         height=280, 
#         showlegend=False, 
#         title_text='KPI Trends (30 Days)',
#         margin=dict(l=40, r=40, t=60, b=40),
#         title_font_size=14
#     )
    
#     return fig.to_html(full_html=False, include_plotlyjs='cdn')

# # --- 7. NPS Tracking ---
# def plot_nps_tracking(df_after_sales):
#     # Ensure 'interaction_date' is datetime type
#     df_after_sales['interaction_date'] = pd.to_datetime(df_after_sales['interaction_date'])

#     # Get the most recent date in the data
#     latest_date = df_after_sales['interaction_date'].max()

#     # Calculate the date 30 days prior to the latest date
#     start_date = latest_date - pd.Timedelta(days=30)

#     # Filter data for the last 30 days
#     df_last_30_days = df_after_sales[(df_after_sales['interaction_date'] >= start_date) & \
#                                      (df_after_sales['interaction_date'] <= latest_date)]

#     nps_data = df_last_30_days.groupby(pd.Grouper(key='interaction_date', freq='D'))['nps_score'].mean().reset_index()
#     nps_data.columns = ['Date', 'NPS']

#     fig = px.line(nps_data, x='Date', y='NPS',
#                     title='Daily NPS Score (Last 30 Days)',
#                     color_discrete_sequence=['#2ca02c'])
#     fig.update_layout(
#         margin=dict(l=40, r=40, t=60, b=40),
#         height=280,
#         title_font_size=14
#     )
#     fig.update_yaxes(range=[0, 10])

#     return fig.to_html(full_html=False, include_plotlyjs='cdn')

# # --- 8. Staff Feedback ---
# def plot_staff_feedback(df_after_sales):
#     feedback_counts = df_after_sales.groupby(['agent_id', 'feedback_score_agent']).size().unstack(fill_value=0)
    
#     if len(feedback_counts) > 8:
#         top_agents = df_after_sales['agent_id'].value_counts().nlargest(8).index
#         feedback_counts = feedback_counts.loc[top_agents]

#     fig = px.bar(feedback_counts, 
#                  x=feedback_counts.index, 
#                  y=feedback_counts.columns,
#                  barmode='group',
#                  title='Agent Feedback Distribution',
#                  labels={'agent_id': 'Agent', 'value': 'Count'},
#                  color_discrete_sequence=px.colors.qualitative.Set2)
    
#     fig.update_layout(
#         margin=dict(l=40, r=40, t=60, b=40), 
#         height=280,
#         title_font_size=14,
#         font_size=10
#     )
    
#     return fig.to_html(full_html=False, include_plotlyjs='cdn')

# # --- 9. Campaign KPIs ---
# def plot_campaign_kpis(df_journey):
#     total_sent = df_journey[df_journey['stage'] == 'sent'].shape[0]
#     total_opens = df_journey[df_journey['campaign_open'] == 'Yes'].shape[0]
#     total_clicks = df_journey[df_journey['campaign_click'] == 'Yes'].shape[0]
#     total_conversions = df_journey[df_journey['conversion_flag'] == 'Yes'].shape[0]

#     open_rate = (total_opens / total_sent) * 100 if total_sent > 0 else 0
#     click_rate = (total_clicks / total_opens) * 100 if total_opens > 0 else 0
#     conversion_rate = (total_conversions / total_sent) * 100 if total_sent > 0 else 0

#     fig = make_subplots(
#         rows=1, cols=3,
#         specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
#         subplot_titles=('Open Rate', 'Click Rate', 'Conversion')
#     )

#     fig.add_trace(go.Indicator(
#         mode="gauge+number+delta",
#         value=open_rate,
#         title={'text': "Open %"},
#         gauge={'axis': {'range': [0, 100]},
#                'bar': {'color': "#636efa"},
#                'steps': [
#                    {'range': [0, 40], 'color': "lightcoral"},
#                    {'range': [40, 70], 'color': "lightgray"},
#                    {'range': [70, 100], 'color': "lightgreen"}],
#                }), row=1, col=1)

#     fig.add_trace(go.Indicator(
#         mode="gauge+number+delta",
#         value=click_rate,
#         title={'text': "Click %"},
#         gauge={'axis': {'range': [0, 100]},
#                'bar': {'color': "#EF553B"},
#                'steps': [
#                    {'range': [0, 10], 'color': "lightcoral"},
#                    {'range': [10, 30], 'color': "lightgray"},
#                    {'range': [30, 100], 'color': "lightgreen"}],
#                }), row=1, col=2)

#     fig.add_trace(go.Indicator(
#         mode="gauge+number+delta",
#         value=conversion_rate,
#         title={'text': "Conv %"},
#         gauge={'axis': {'range': [0, 100]},
#                'bar': {'color': "#00cc96"},
#                'steps': [
#                    {'range': [0, 5], 'color': "lightcoral"},
#                    {'range': [5, 15], 'color': "lightgray"},
#                    {'range': [15, 100], 'color': "lightgreen"}],
#                }), row=1, col=3)

#     fig.update_layout(
#         height=280, 
#         margin=dict(l=20, r=20, t=60, b=20), 
#         title_text="Campaign Metrics",
#         title_font_size=14
#     )
    
#     return fig.to_html(full_html=False, include_plotlyjs='cdn')

# # --- Generate HTML Dashboard ---
# def generate_dashboard_html(plots):
#     html_template = """
#     <!DOCTYPE html>
#     <html lang="en">
#     <head>
#         <meta charset="UTF-8">
#         <meta name="viewport" content="width=device-width, initial-scale=1.0">
#         <title>CX Analytics Dashboard</title>
#         <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
#         <style>
#             body {{ 
#                 font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
#                 background-color: #f8f9fa; 
#                 color: #333; 
#                 margin: 0;
#                 padding: 0;
#             }}
#             .dashboard-container {{ 
#                 padding: 15px; 
#                 max-width: 1400px;
#                 margin: 0 auto;
#             }}
#             .dashboard-title {{
#                 text-align: center;
#                 color: #007bff;
#                 margin-bottom: 25px;
#                 font-weight: 600;
#                 font-size: 2.2rem;
#             }}
#             .dashboard-row {{
#                 margin-bottom: 15px;
#             }}
#             .dashboard-card {{
#                 border: none;
#                 border-radius: 10px;
#                 box-shadow: 0 2px 8px rgba(0,0,0,0.1);
#                 margin-bottom: 15px;
#                 background-color: #fff;
#                 height: 350px; /* Fixed height for all cards */
#                 display: flex;
#                 flex-direction: column;
#             }}
#             .card-header {{
#                 background: linear-gradient(135deg, #007bff, #0056b3);
#                 color: white;
#                 padding: 12px 20px;
#                 font-size: 1rem;
#                 font-weight: 600;
#                 border-radius: 10px 10px 0 0;
#                 border-bottom: none;
#                 flex-shrink: 0;
#             }}
#             .card-body {{
#                 padding: 15px;
#                 flex-grow: 1;
#                 display: flex;
#                 align-items: center;
#                 justify-content: center;
#                 overflow: hidden;
#             }}
#             .plot-container {{
#                 width: 100%;
#                 height: 100%;
#                 display: flex;
#                 align-items: center;
#                 justify-content: center;
#             }}
#             .plot-container > div {{
#                 width: 100% !important;
#                 height: 100% !important;
#             }}
#             /* Responsive adjustments */
#             @media (max-width: 768px) {{
#                 .dashboard-card {{
#                     height: 300px;
#                 }}
#                 .dashboard-title {{
#                     font-size: 1.8rem;
#                 }}
#                 .card-header {{
#                     font-size: 0.9rem;
#                     padding: 10px 15px;
#                 }}
#             }}
#             @media (max-width: 576px) {{
#                 .dashboard-container {{
#                     padding: 10px;
#                 }}
#                 .dashboard-card {{
#                     height: 280px;
#                     margin-bottom: 10px;
#                 }}
#             }}
#         </style>
#     </head>
#     <body>
#         <div class="dashboard-container">
#             <h1 class="dashboard-title">Customer Experience Analytics Dashboard</h1>

#             <!-- Row 1 -->
#             <div class="row dashboard-row">
#                 <div class="col-lg-4 col-md-6 col-sm-12">
#                     <div class="dashboard-card">
#                         <div class="card-header">📊 Real-Time Sentiment</div>
#                         <div class="card-body">
#                             <div class="plot-container">
#                                 {plot_html_sentiment}
#                             </div>
#                         </div>
#                     </div>
#                 </div>
#                 <div class="col-lg-4 col-md-6 col-sm-12">
#                     <div class="dashboard-card">
#                         <div class="card-header">👥 Agent Performance</div>
#                         <div class="card-body">
#                             <div class="plot-container">
#                                 {agent_table_html}
#                             </div>
#                         </div>
#                     </div>
#                 </div>
#                 <div class="col-lg-4 col-md-6 col-sm-12">
#                     <div class="dashboard-card">
#                         <div class="card-header">🎯 CX Health Snapshot</div>
#                         <div class="card-body">
#                             <div class="plot-container">
#                                 {plot_html_cx_radar}
#                             </div>
#                         </div>
#                     </div>
#                 </div>
#             </div>

#             <!-- Row 2 -->
#             <div class="row dashboard-row">
#                 <div class="col-lg-4 col-md-6 col-sm-12">
#                     <div class="dashboard-card">
#                         <div class="card-header">🗂️ Issues Categorization</div>
#                         <div class="card-body">
#                             <div class="plot-container">
#                                 {plot_html_issue_treemap}
#                             </div>
#                         </div>
#                     </div>
#                 </div>
#                 <div class="col-lg-4 col-md-6 col-sm-12">
#                     <div class="dashboard-card">
#                         <div class="card-header">⏱️ Queue & SLA Metrics</div>
#                         <div class="card-body">
#                             <div class="plot-container">
#                                 {plot_html_queue_sla}
#                             </div>
#                         </div>
#                     </div>
#                 </div>
#                 <div class="col-lg-4 col-md-6 col-sm-12">
#                     <div class="dashboard-card">
#                         <div class="card-header">📈 KPI Trends</div>
#                         <div class="card-body">
#                             <div class="plot-container">
#                                 {plot_html_kpi_sparklines}
#                             </div>
#                         </div>
#                     </div>
#                 </div>
#             </div>

#             <!-- Row 3 -->
#             <div class="row dashboard-row">
#                 <div class="col-lg-4 col-md-6 col-sm-12">
#                     <div class="dashboard-card">
#                         <div class="card-header">⭐ NPS Tracking</div>
#                         <div class="card-body">
#                             <div class="plot-container">
#                                 {plot_html_nps_tracking}
#                             </div>
#                         </div>
#                     </div>
#                 </div>
#                 <div class="col-lg-4 col-md-6 col-sm-12">
#                     <div class="dashboard-card">
#                         <div class="card-header">💬 Staff Feedback</div>
#                         <div class="card-body">
#                             <div class="plot-container">
#                                 {plot_html_staff_feedback}
#                             </div>
#                         </div>
#                     </div>
#                 </div>
#                 <div class="col-lg-4 col-md-6 col-sm-12">
#                     <div class="dashboard-card">
#                         <div class="card-header">🎯 Campaign KPIs</div>
#                         <div class="card-body">
#                             <div class="plot-container">
#                                 {plot_html_campaign_kpis}
#                             </div>
#                         </div>
#                     </div>
#                 </div>
#             </div>

#         </div>

#         <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
#         <script>
#             // Ensure proper responsive behavior for Plotly charts
#             window.addEventListener('resize', function() {{
#                 if (window.Plotly) {{
#                     window.Plotly.Plots.resize();
#                 }}
#             }});
#         </script>
#     </body>
#     </html>
#     """
#     return html_template.format(**plots)

# # --- Main Execution ---
# if __name__ == "__main__":
#     # Generate all plots
#     plot_html_sentiment = plot_realtime_sentiment(df_sentiment)
#     agent_table_html = create_agent_performance_table(df_after_sales)
#     plot_html_cx_radar = plot_cx_radar_chart(df_after_sales, df_sentiment)
#     plot_html_issue_treemap = plot_issue_treemap(df_after_sales)
#     plot_html_queue_sla = plot_queue_sla_metrics(df_after_sales)
#     plot_html_kpi_sparklines = plot_kpi_sparklines(df_after_sales, df_sentiment)
#     plot_html_nps_tracking = plot_nps_tracking(df_after_sales)
#     plot_html_staff_feedback = plot_staff_feedback(df_after_sales)
#     plot_html_campaign_kpis = plot_campaign_kpis(df_journey)

#     plots_dict = {
#         'plot_html_sentiment': plot_html_sentiment,
#         'agent_table_html': agent_table_html,
#         'plot_html_cx_radar': plot_html_cx_radar,
#         'plot_html_issue_treemap': plot_html_issue_treemap,
#         'plot_html_queue_sla': plot_html_queue_sla,
#         'plot_html_kpi_sparklines': plot_html_kpi_sparklines,
#         'plot_html_nps_tracking': plot_html_nps_tracking,
#         'plot_html_staff_feedback': plot_html_staff_feedback,
#         'plot_html_campaign_kpis': plot_html_campaign_kpis,
#     }

#     html_output = generate_dashboard_html(plots_dict)

#     # Save to an HTML file
#     dashboard_filename = 'cx_dashboard.html'
#     with open(dashboard_filename, 'w', encoding='utf-8') as f:
#         f.write(html_output)

#     print(f"Fixed Dashboard '{dashboard_filename}' generated successfully!")
#     print("Key improvements made:") 
#     print("✅ Fixed height management with consistent 350px cards")
#     print("✅ Improved responsive design for mobile/tablet")
#     print("✅ Better chart sizing and margins")
#     print("✅ Enhanced visual styling with gradients and icons")
#     print("✅ Proper flexbox layout for content alignment")
#     print("✅ Simplified and cleaned up chart functions")
#     print("\nOpen the HTML file in your web browser to view the properly formatted 3x3 dashboard.")