import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random
import folium
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
from io import BytesIO
import tempfile
import os

print("Generating data for Dashboard Visualizations...")

# --- Configuration for Data Generation ---
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

except FileNotFoundError as e:
    print(f"Error: {e}. One or more required CSV files not found. Please run previous data generation scripts first.")
    exit()

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

# --- 2. Call Center Agent Performance ---
def create_agent_performance_table(df_after_sales):
    agent_performance = df_after_sales.groupby('agent_id').agg(
        calls_handled=('interaction_id', 'count'),
        total_duration_minutes=('resolution_time_minutes', 'sum')
    ).reset_index()
    agent_performance['avg_duration'] = (agent_performance['total_duration_minutes'] / agent_performance['calls_handled']).round(2)
    agent_performance['status'] = [random.choice(['Online', 'Offline', 'Busy']) for _ in range(len(agent_performance))]

    # Limit to top 8 agents for better display
    agent_performance = agent_performance.nlargest(8, 'calls_handled')
    
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Agent', 'Status', 'Calls', 'Avg Duration'],
                   fill_color='#007bff',
                   font=dict(color='white', size=12),
                   align='center'),
        cells=dict(values=[agent_performance.agent_id, 
                          agent_performance.status,
                          agent_performance.calls_handled, 
                          agent_performance.avg_duration],
                  fill_color='lightgrey',
                  align='center',
                  font_size=10))
    ])
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=280,
        title='Agent Performance',
        title_font_size=14
    )
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

# --- 3. CX Health Snapshot Radar Chart ---
def plot_cx_radar_chart(df_after_sales, df_sentiment):
    customer_id_choices = df_after_sales['customer_id'].unique()
    if len(customer_id_choices) == 0:
        return go.Figure().to_html(full_html=False, include_plotlyjs='cdn')

    customer_id = random.choice(customer_id_choices)
    cust_interactions = df_after_sales[df_after_sales['customer_id'] == customer_id]
    cust_sentiment = df_sentiment[df_sentiment['customer_id'] == customer_id]

    recent_sentiment_score = cust_sentiment['sentiment_score'].mean() if not cust_sentiment.empty else 0
    num_open_issues = cust_interactions[cust_interactions['resolution_status'] != 'Resolved'].shape[0]
    time_since_last_interaction = (datetime.now() - cust_interactions['interaction_date'].max()).days if not cust_interactions.empty else 365
    recent_nps_score = cust_interactions['nps_score'].mean() if not cust_interactions.empty else 5
    product_ownership_flag = 1 if not cust_interactions.empty else 0

    # Normalize values for radar chart (0-1 scale)
    metrics = [
        (recent_sentiment_score + 1) / 2,
        1 - (min(num_open_issues, 5) / 5),
        1 - (min(time_since_last_interaction, 90) / 90),
        recent_nps_score / 10,
        product_ownership_flag
    ]
    categories = ['Sentiment', 'Issues', 'Recency', 'NPS', 'Product']

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
                         '(?)': 'lightgray',
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

# # --- 5. Call Queue & SLA Metrics ---
# def plot_queue_sla_metrics(df_after_sales):
#     call_interactions = df_after_sales[df_after_sales['interaction_type'].isin(["Call", "Chat"])].copy()
    
#     daily_metrics = call_interactions.groupby(pd.Grouper(key='interaction_date', freq='D')).agg(
#         queue_calls=('queue_time_seconds', lambda x: (x > 0).sum()),
#         sla_met=('sla_met', lambda x: (x == 'Yes').sum()),
#         total_calls=('interaction_id', 'count')
#     ).reset_index()

#     daily_metrics['sla_percentage'] = ((daily_metrics['sla_met'] / daily_metrics['total_calls']) * 100).fillna(0).round(2)

#     fig = make_subplots(specs=[[{"secondary_y": True}]])

#     fig.add_trace(
#         go.Bar(x=daily_metrics['interaction_date'], y=daily_metrics['queue_calls'], 
#                name='Queue Calls', marker_color='#ff7f0e'),
#         secondary_y=False,
#     )

#     fig.add_trace(
#         go.Scatter(x=daily_metrics['interaction_date'], y=daily_metrics['sla_percentage'], 
#                    name='SLA %', mode='lines+markers', line_color='#2ca02c'),
#         secondary_y=True,
#     )

#     fig.update_layout(
#         title_text='Queue & SLA Metrics',
#         margin=dict(l=40, r=40, t=60, b=40), 
#         height=280,
#         title_font_size=14
#     )
#     fig.update_xaxes(title_text="Date")
#     fig.update_yaxes(title_text="Queue Calls", secondary_y=False)
#     fig.update_yaxes(title_text="SLA %", secondary_y=True, range=[0, 100])
    
#     return fig.to_html(full_html=False, include_plotlyjs='cdn')

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta # Ensure this is imported if not already

# Assuming df_after_sales is a DataFrame with 'interaction_date', 'interaction_type',
# 'queue_time_seconds', 'sla_met', 'interaction_id' columns.
# It should be processed to have 'interaction_date' as datetime.

def plot_queue_sla_metrics(df_after_sales):
    # Ensure interaction_date is datetime and handle potential errors
    df_after_sales['interaction_date'] = pd.to_datetime(df_after_sales['interaction_date'], errors='coerce')
    
    # Filter for relevant interaction types (Calls and Chats) and drop NaT dates
    call_interactions = df_after_sales[
        df_after_sales['interaction_type'].isin(["Call", "Chat"])
    ].dropna(subset=['interaction_date']).copy()

    # Aggregate daily metrics
    daily_metrics = call_interactions.groupby(pd.Grouper(key='interaction_date', freq='D')).agg(
        queue_calls=('queue_time_seconds', lambda x: (x > 0).sum()), # Count calls with queue time
        sla_met=('sla_met', lambda x: (x == 'Yes').sum()),          # Count SLA met calls
        total_calls=('interaction_id', 'count')                     # Count all calls/chats
    ).reset_index()

    # Calculate SLA percentage, handling division by zero for days with no calls
    daily_metrics['sla_percentage'] = ((daily_metrics['sla_met'] / daily_metrics['total_calls']) * 100).fillna(0).round(2)

    # --- Plotting with Plotly Graph Objects ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. Bar Chart for Queue Calls
    fig.add_trace(
        go.Bar(
            x=daily_metrics['interaction_date'], 
            y=daily_metrics['queue_calls'], 
            name='Daily Queue Calls', 
            marker_color='#1f77b4', # A standard blue, distinct from the original orange
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Queue Calls:</b> %{y}<extra></extra>'
        ),
        secondary_y=False,
    )

    # 2. Line Chart for SLA Percentage
    fig.add_trace(
        go.Scatter(
            x=daily_metrics['interaction_date'], 
            y=daily_metrics['sla_percentage'], 
            name='Daily SLA Met (%)', 
            mode='lines+markers', 
            line=dict(color='#2ca02c', width=2), # Green for good performance, slightly thicker line
            marker=dict(size=6, symbol='circle'), # Clearer markers
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>SLA Met:</b> %{y:.2f}%<extra></extra>'
        ),
        secondary_y=True,
    )

    # 3. Add an SLA Target Line (e.g., 85% or 90%)
    # This provides a visual benchmark for performance.
    sla_target = 85 # Define your target SLA percentage here
    fig.add_hline(
        y=sla_target, 
        line_dash="dot", 
        line_color="#d62728", # Red color for target/warning
        annotation_text=f"SLA Target ({sla_target}%)", 
        annotation_position="top left",
        annotation_font_size=10,
        secondary_y=True # Make sure it's on the correct axis
    )

    # --- Update Layout and Axes ---
    fig.update_layout(
        title_text='<b>Contact Center Performance: Queue & SLA Metrics</b>', # More descriptive title
        margin=dict(l=40, r=40, t=60, b=40), 
        height=350, # Slightly increased height for better readability
        title_font_size=16, # Slightly larger title font
        legend=dict(x=0.01, y=1.1, xanchor='left', yanchor='top', orientation="h"), # Horizontal legend at top
        plot_bgcolor='white', # Clean background
        paper_bgcolor='white', # Clean paper background
        hovermode="x unified" # Unify hover across both traces for a given x-point
    )

    fig.update_xaxes(
        title_text="Date",
        showgrid=True, gridwidth=1, gridcolor='lightgray', # Add light gridlines
        tickformat="%b %d", # Format date ticks (e.g., Jul 29)
        rangeslider_visible=False # Hide range slider to save space if not needed
    )
    
    fig.update_yaxes(
        title_text="Number of Calls in Queue", 
        secondary_y=False,
        showgrid=True, gridwidth=1, gridcolor='lightgray',
        zeroline=True, zerolinewidth=1, zerolinecolor='lightgray'
    )
    
    fig.update_yaxes(
        title_text="SLA Met Percentage (%)", 
        secondary_y=True, 
        range=[0, 100], # Keep range fixed for SLA percentage
        showgrid=False # No grid for secondary axis to avoid clutter
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
    daily_sla['SLA_Pct'] = ((daily_sla['sla_met'] / daily_sla['total']) * 100).fillna(0)

    # Daily FCR
    daily_fcr = df_after_sales.groupby(pd.Grouper(key='interaction_date', freq='D')).agg(
        total=('interaction_id', 'count'),
        fcr=('is_first_contact_resolution', lambda x: (x == 'Yes').sum())
    ).reset_index()
    daily_fcr['FCR_Pct'] = ((daily_fcr['fcr'] / daily_fcr['total']) * 100).fillna(0)

    # Combine data
    kpi_data = daily_sentiment.merge(daily_sla, left_on='Date', right_on='interaction_date', how='outer')
    kpi_data = kpi_data.merge(daily_fcr, left_on='Date', right_on='interaction_date', how='outer')
    kpi_data = kpi_data.sort_values('Date').fillna(method='ffill').tail(30)
    
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

# --- 7. NPS Tracking ---
def plot_nps_tracking(df_after_sales):
    nps_data = df_after_sales.groupby(pd.Grouper(key='interaction_date', freq='D'))['nps_score'].mean().reset_index()
    nps_data.columns = ['Date', 'NPS']
    
    fig = px.line(nps_data, x='Date', y='NPS', 
                  title='Daily NPS Score',
                  color_discrete_sequence=['#2ca02c'])
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40), 
        height=280,
        title_font_size=14
    )
    fig.update_yaxes(range=[0, 10])
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

# --- 8. Staff Feedback ---
def plot_staff_feedback(df_after_sales):
    feedback_counts = df_after_sales.groupby(['agent_id', 'feedback_score_agent']).size().unstack(fill_value=0)
    
    if len(feedback_counts) > 8:
        top_agents = df_after_sales['agent_id'].value_counts().nlargest(8).index
        feedback_counts = feedback_counts.loc[top_agents]

    fig = px.bar(feedback_counts, 
                 x=feedback_counts.index, 
                 y=feedback_counts.columns,
                 barmode='group',
                 title='Agent Feedback Distribution',
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
    total_sent = df_journey[df_journey['stage'] == 'sent'].shape[0]
    total_opens = df_journey[df_journey['campaign_open'] == 'Yes'].shape[0]
    total_clicks = df_journey[df_journey['campaign_click'] == 'Yes'].shape[0]
    total_conversions = df_journey[df_journey['conversion_flag'] == 'Yes'].shape[0]

    open_rate = (total_opens / total_sent) * 100 if total_sent > 0 else 0
    click_rate = (total_clicks / total_opens) * 100 if total_opens > 0 else 0
    conversion_rate = (total_conversions / total_sent) * 100 if total_sent > 0 else 0

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
                height: 350px; /* Fixed height for all cards */
                display: flex;
                flex-direction: column;
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
                align-items: center;
                justify-content: center;
                overflow: hidden;
            }}
            .plot-container {{
                width: 100%;
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .plot-container > div {{
                width: 100% !important;
                height: 100% !important;
            }}
            /* Responsive adjustments */
            @media (max-width: 768px) {{
                .dashboard-card {{
                    height: 300px;
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
            }}
        </style>
    </head>
    <body>
        <div class="dashboard-container">
            <h1 class="dashboard-title">Customer Experience Analytics Dashboard</h1>

            <!-- Row 1 -->
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

            <!-- Row 2 -->
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

            <!-- Row 3 -->
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
                    <div class="dashboard-card">
                        <div class="card-header">💬 Staff Feedback</div>
                        <div class="card-body">
                            <div class="plot-container">
                                {plot_html_staff_feedback}
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-lg-4 col-md-6 col-sm-12">
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
                    window.Plotly.Plots.resize();
                }}
            }});
        </script>
    </body>
    </html>
    """
    return html_template.format(**plots)

# --- Main Execution ---
if __name__ == "__main__":
    # Generate all plots
    plot_html_sentiment = plot_realtime_sentiment(df_sentiment)
    agent_table_html = create_agent_performance_table(df_after_sales)
    plot_html_cx_radar = plot_cx_radar_chart(df_after_sales, df_sentiment)
    plot_html_issue_treemap = plot_issue_treemap(df_after_sales)
    plot_html_queue_sla = plot_queue_sla_metrics(df_after_sales)
    plot_html_kpi_sparklines = plot_kpi_sparklines(df_after_sales, df_sentiment)
    plot_html_nps_tracking = plot_nps_tracking(df_after_sales)
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
        'plot_html_staff_feedback': plot_html_staff_feedback,
        'plot_html_campaign_kpis': plot_html_campaign_kpis,
    }

    html_output = generate_dashboard_html(plots_dict)

    # Save to an HTML file
    dashboard_filename = 'cx_dashboard.html'
    with open(dashboard_filename, 'w', encoding='utf-8') as f:
        f.write(html_output)

    print(f"Fixed Dashboard '{dashboard_filename}' generated successfully!")
    print("Key improvements made:")
    print("✅ Fixed height management with consistent 350px cards")
    print("✅ Improved responsive design for mobile/tablet")
    print("✅ Better chart sizing and margins")
    print("✅ Enhanced visual styling with gradients and icons")
    print("✅ Proper flexbox layout for content alignment")
    print("✅ Simplified and cleaned up chart functions")
    print("\nOpen the HTML file in your web browser to view the properly formatted 3x3 dashboard.")

# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from datetime import datetime, timedelta
# import random

# print("Generating data for Dashboard Visualizations...")

# # --- Configuration for Data Generation (ensure consistency with previous scripts) ---
# # Assuming these CSVs are already generated from previous steps
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
#     # Exit or create dummy data for demonstration if files are missing
#     exit() # Exiting for now as data consistency is key

# # --- 1. Real-Time Sentiment Dashboard (Summary) ---
# # Line Chart: Average sentiment over time (e.g., last 30 days)
# # Gauge Chart: Overall current sentiment

# def plot_realtime_sentiment(df_sentiment):
#     recent_sentiment = df_sentiment[df_sentiment['date'] > (datetime.now() - timedelta(days=30))]
#     if recent_sentiment.empty:
#         return go.Figure() # Return empty figure if no recent data

#     sentiment_trend = recent_sentiment.groupby(pd.Grouper(key='date', freq='D'))['sentiment_score'].mean().reset_index()
#     sentiment_trend.columns = ['Date', 'Average Sentiment']

#     fig_line = px.line(sentiment_trend, x='Date', y='Average Sentiment',
#                        title='Last 30 Days Average Sentiment',
#                        color_discrete_sequence=['#1f77b4']) # blue
#     fig_line.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=250)

#     current_sentiment_score = recent_sentiment['sentiment_score'].mean() if not recent_sentiment.empty else 0.0
#     fig_gauge = go.Figure(go.Indicator(
#         mode = "gauge+number",
#         value = current_sentiment_score,
#         domain = {'x': [0, 1], 'y': [0, 1]},
#         title = {'text': "Current Overall Sentiment"},
#         gauge = {'axis': {'range': [-1, 1]},
#                  'bar': {'color': "#1f77b4"},
#                  'steps' : [
#                      {'range': [-1, -0.3], 'color': "lightcoral"},
#                      {'range': [-0.3, 0.3], 'color': "lightgray"},
#                      {'range': [0.3, 1], 'color': "lightgreen"}],
#                  'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.5}} # Example threshold
#     ))
#     fig_gauge.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=250)
#     return fig_line, fig_gauge

# fig_sentiment_trend, fig_sentiment_gauge = plot_realtime_sentiment(df_sentiment)


# # --- 2. Call Center Agent Performance (Table with Conditional Formatting) ---
# def create_agent_performance_table(df_after_sales):
#     agent_performance = df_after_sales.groupby('agent_id').agg(
#         calls_handled=('interaction_id', 'count'),
#         total_duration_minutes=('resolution_time_minutes', 'sum')
#     ).reset_index()
#     agent_performance['average_call_duration_minutes'] = agent_performance['total_duration_minutes'] / agent_performance['calls_handled']

#     # Simulate online/offline status for agents (random for this static report)
#     agent_performance['availability_status'] = [random.choice(['Online', 'Offline', 'Busy']) for _ in range(len(agent_performance))]

#     # Select relevant columns and round duration
#     agent_performance = agent_performance[['agent_id', 'availability_status', 'calls_handled', 'average_call_duration_minutes']]
#     agent_performance['average_call_duration_minutes'] = agent_performance['average_call_duration_minutes'].round(2)

#     # Convert to HTML table with basic styling (Plotly's Table is an option, but pandas styling for simple HTML is good)
#     # For advanced conditional formatting, external CSS or more complex JS might be needed with raw HTML
#     styled_table = agent_performance.style \
#         .background_gradient(cmap='Blues', subset=['calls_handled']) \
#         .format({'average_call_duration_minutes': "{:.2f} mins"}) \
#         .to_html(classes='table table-striped table-hover', table_id='agent_performance_table')
#     return styled_table

# agent_table_html = create_agent_performance_table(df_after_sales)


# # --- 3. CX Health Snapshot Radar Chart ---
# def plot_cx_radar_chart(df_after_sales, df_sentiment):
#     # Select a random customer for the snapshot
#     customer_id = random.choice(df_after_sales['customer_id'].unique())
#     cust_interactions = df_after_sales[df_after_sales['customer_id'] == customer_id]
#     cust_sentiment = df_sentiment[df_sentiment['customer_id'] == customer_id]

#     if cust_interactions.empty and cust_sentiment.empty:
#         return go.Figure()

#     # Key CX indicators for the individual
#     recent_sentiment_score = cust_sentiment['sentiment_score'].mean() if not cust_sentiment.empty else 0
#     num_open_issues = cust_interactions[cust_interactions['resolution_status'] != 'Resolved'].shape[0]
#     time_since_last_interaction = (datetime.now() - cust_interactions['interaction_date'].max()).days if not cust_interactions.empty else 365 # Default if no interactions
#     recent_nps_score = cust_interactions['nps_score'].mean() if not cust_interactions.empty else 5 # Default neutral if no NPS
#     # product_ownership_flag is a simple 'Yes' in after_sales.csv, could derive more complex if needed
#     product_ownership_flag = 1 if not cust_interactions.empty else 0 # 1 if they have interactions, 0 otherwise

#     # Normalize values for radar chart (0-1 scale, higher is better)
#     # Assuming: sentiment: -1 to 1, open_issues: 0 to max, time_since_last: 0 to 365, nps: 0 to 10
#     # Higher is better for all except open_issues and time_since_last_interaction (where lower is better, so invert)
#     metrics = [
#         (recent_sentiment_score + 1) / 2, # Scale sentiment to 0-1
#         1 - (min(num_open_issues, 5) / 5), # Invert & Cap open issues at 5
#         1 - (min(time_since_last_interaction, 90) / 90), # Invert & Cap days at 90 for scaling
#         recent_nps_score / 10, # Scale NPS to 0-1
#         product_ownership_flag # Already 0 or 1
#     ]
#     categories = ['Sentiment', 'No. Open Issues', 'Recency of Interaction', 'NPS', 'Product Ownership']

#     fig = go.Figure(data=go.Scatterpolar(
#         r=metrics,
#         theta=categories,
#         fill='toself',
#         name=f'Customer: {customer_id}'
#     ))
#     fig.update_layout(
#         polar=dict(
#             radialaxis=dict(
#                 visible=True,
#                 range=[0, 1]
#             )),
#         showlegend=True,
#         title=f'CX Health Snapshot for {customer_id}',
#         margin=dict(l=20, r=20, t=60, b=20), height=300
#     )
#     return fig

# fig_cx_radar = plot_cx_radar_chart(df_after_sales, df_sentiment)


# # --- 4. Categorization of Issues Treemap chart ---
# def plot_issue_treemap(df_after_sales):
#     # Aggregate sentiment by issue category and sentiment category
#     df_issues = df_after_sales.groupby(['issue_category', 'sentiment_category']).size().reset_index(name='count')
    
#     # Calculate total for percentage
#     total_count = df_issues['count'].sum()
#     df_issues['percentage'] = (df_issues['count'] / total_count) * 100

#     fig = px.treemap(df_issues, path=[px.Constant("All Issues"), 'issue_category', 'sentiment_category'],
#                      values='count',
#                      color='sentiment_category',
#                      color_discrete_map={
#                          '(?)': 'lightgray',
#                          'Positive': 'lightgreen',
#                          'Neutral': 'lightgray',
#                          'Negative': 'lightcoral'
#                      },
#                      title='Categorization of Issues by Sentiment',
#                      labels={'count': 'Number of Interactions', 'sentiment_category': 'Sentiment'}
#                     )
#     fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=300)
#     fig.update_traces(hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Sentiment: %{color}<extra></extra>')
#     return fig

# fig_issue_treemap = plot_issue_treemap(df_after_sales)


# # --- 5. Call Queue & SLA Metrics Dual-Axis Chart ---
# def plot_queue_sla_metrics(df_after_sales):
#     # Filter for interactions that are likely calls/chats and could have queue time
#     call_interactions = df_after_sales[df_after_sales['interaction_type'].isin(["Call", "Chat"])].copy()

#     # Aggregate by date (daily trends)
#     daily_metrics = call_interactions.groupby(pd.Grouper(key='interaction_date', freq='D')).agg(
#         num_calls_in_queue=('queue_time_seconds', lambda x: (x > 0).sum()), # Count calls with queue time
#         calls_meeting_sla=('sla_met', lambda x: (x == 'Yes').sum()),
#         total_calls=('interaction_id', 'count')
#     ).reset_index()

#     daily_metrics['percentage_meeting_sla'] = (daily_metrics['calls_meeting_sla'] / daily_metrics['total_calls']) * 100
#     daily_metrics['percentage_meeting_sla'] = daily_metrics['percentage_meeting_sla'].fillna(0).round(2)

#     fig = make_subplots(specs=[[{"secondary_y": True}]])

#     # Add calls in queue (Y-axis 1)
#     fig.add_trace(
#         go.Bar(x=daily_metrics['interaction_date'], y=daily_metrics['num_calls_in_queue'], name='Calls in Queue'),
#         secondary_y=False,
#     )

#     # Add percentage meeting SLA (Y-axis 2)
#     fig.add_trace(
#         go.Line(x=daily_metrics['interaction_date'], y=daily_metrics['percentage_meeting_sla'], name='Percentage Meeting SLA', mode='lines+markers'),
#         secondary_y=True,
#     )

#     fig.update_layout(
#         title_text='Daily Call Queue and SLA Metrics',
#         margin=dict(l=20, r=20, t=60, b=20), height=300
#     )
#     fig.update_xaxes(title_text="Date")
#     fig.update_yaxes(title_text="Number of Calls in Queue", secondary_y=False)
#     fig.update_yaxes(title_text="Percentage Meeting SLA (%)", secondary_y=True, range=[0, 100])
#     return fig

# fig_queue_sla = plot_queue_sla_metrics(df_after_sales)


# # --- 6. Threshold-Based KPI Alerts Sparklines chart ---
# # This is a conceptual chart type. We'll simulate a few sparklines for key KPIs.
# def plot_kpi_sparklines(df_after_sales, df_sentiment):
#     # Example KPIs: Average Sentiment, SLA Met %, FCR %
    
#     # Daily average sentiment
#     daily_sentiment = df_sentiment.groupby(pd.Grouper(key='date', freq='D'))['sentiment_score'].mean().reset_index()
#     daily_sentiment.columns = ['Date', 'Avg_Sentiment']
    
#     # Daily SLA Met %
#     daily_sla = df_after_sales.groupby(pd.Grouper(key='interaction_date', freq='D')).agg(
#         total_interactions=('interaction_id', 'count'),
#         sla_met_count=('sla_met', lambda x: (x == 'Yes').sum())
#     ).reset_index()
#     daily_sla['SLA_Met_Percentage'] = (daily_sla['sla_met_count'] / daily_sla['total_interactions']) * 100
#     daily_sla['SLA_Met_Percentage'] = daily_sla['SLA_Met_Percentage'].fillna(0)

#     # Daily FCR %
#     daily_fcr = df_after_sales.groupby(pd.Grouper(key='interaction_date', freq='D')).agg(
#         total_interactions=('interaction_id', 'count'),
#         fcr_count=('is_first_contact_resolution', lambda x: (x == 'Yes').sum())
#     ).reset_index()
#     daily_fcr['FCR_Percentage'] = (daily_fcr['fcr_count'] / daily_fcr['total_interactions']) * 100
#     daily_fcr['FCR_Percentage'] = daily_fcr['FCR_Percentage'].fillna(0)

#     # Combine for common dates
#     kpi_data = daily_sentiment.merge(daily_sla, left_on='Date', right_on='interaction_date', how='outer')
#     kpi_data = kpi_data.merge(daily_fcr, left_on='Date', right_on='interaction_date', how='outer')
#     kpi_data = kpi_data.sort_values('Date').fillna(method='ffill').tail(30) # Last 30 days
    
#     if kpi_data.empty:
#         return go.Figure()

#     fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1)

#     # Sparkline for Avg Sentiment
#     fig.add_trace(go.Scatter(x=kpi_data['Date'], y=kpi_data['Avg_Sentiment'], mode='lines', name='Avg Sentiment',
#                              line=dict(color='blue', width=1)), row=1, col=1)
#     fig.add_hline(y=0.3, line_dash="dash", line_color="green", row=1, col=1) # Threshold example
#     fig.update_yaxes(title_text="Sentiment", row=1, col=1, showticklabels=False, range=[-1,1])
#     fig.update_xaxes(showticklabels=False, row=1, col=1) # Hide x-axis labels for sparkline

#     # Sparkline for SLA Met %
#     fig.add_trace(go.Scatter(x=kpi_data['Date'], y=kpi_data['SLA_Met_Percentage'], mode='lines', name='SLA Met %',
#                              line=dict(color='orange', width=1)), row=2, col=1)
#     fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1) # Threshold example
#     fig.update_yaxes(title_text="SLA %", row=2, col=1, showticklabels=False, range=[0,100])
#     fig.update_xaxes(showticklabels=False, row=2, col=1)

#     # Sparkline for FCR %
#     fig.add_trace(go.Scatter(x=kpi_data['Date'], y=kpi_data['FCR_Percentage'], mode='lines', name='FCR %',
#                              line=dict(color='green', width=1)), row=3, col=1)
#     fig.add_hline(y=70, line_dash="dash", line_color="purple", row=3, col=1) # Threshold example
#     fig.update_yaxes(title_text="FCR %", row=3, col=1, showticklabels=False, range=[0,100])
#     fig.update_xaxes(showticklabels=False, row=3, col=1) # Hide x-axis labels for sparkline

#     fig.update_layout(height=300, showlegend=False, title_text='Key KPI Sparklines (Last 30 Days)',
#                       margin=dict(l=20, r=20, t=60, b=20))
#     return fig

# fig_kpi_sparklines = plot_kpi_sparklines(df_after_sales, df_sentiment)


# # --- 7. Real-Time NPS Tracking Line Chart ---
# def plot_nps_tracking(df_after_sales):
#     nps_data = df_after_sales.groupby(pd.Grouper(key='interaction_date', freq='D'))['nps_score'].mean().reset_index()
#     nps_data.columns = ['Date', 'Average NPS']
    
#     fig = px.line(nps_data, x='Date', y='Average NPS', title='Daily Average NPS Score',
#                   color_discrete_sequence=['#2ca02c']) # green
#     fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=250)
#     fig.update_yaxes(range=[0,10]) # NPS range 0-10
#     return fig

# fig_nps_tracking = plot_nps_tracking(df_after_sales)


# # --- 8. Staff/Departmental Feedback grouped bar chart ---
# def plot_staff_feedback(df_after_sales):
#     # For simplicity, let's group by agent_id and feedback_score_agent
#     feedback_counts = df_after_sales.groupby(['agent_id', 'feedback_score_agent']).size().unstack(fill_value=0)
    
#     # If there are many agents, pick top N or sample
#     if len(feedback_counts) > 10: # Limit to top 10 agents by interactions
#         top_agents = df_after_sales['agent_id'].value_counts().nlargest(10).index
#         feedback_counts = feedback_counts.loc[top_agents]

#     fig = px.bar(feedback_counts, x=feedback_counts.index, y=feedback_counts.columns,
#                  barmode='group',
#                  title='Agent Feedback Scores Distribution (1-5)',
#                  labels={'agent_id': 'Agent ID', 'value': 'Count', 'feedback_score_agent': 'Feedback Score'},
#                  color_discrete_sequence=px.colors.sequential.Plasma) # A nice color sequence
#     fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=300)
#     return fig

# fig_staff_feedback = plot_staff_feedback(df_after_sales)


# # --- 9. Custom Dashboard Builder (using Campaign KPIs as example) ---
# # Gauge Charts for open/click/conversion rates (from df_journey)
# def plot_campaign_kpis(df_journey):
#     # Calculate overall rates from journey data
#     total_sent = df_journey[df_journey['stage'] == 'sent'].shape[0]
#     total_opens = df_journey[df_journey['campaign_open'] == 'Yes'].shape[0]
#     total_clicks = df_journey[df_journey['campaign_click'] == 'Yes'].shape[0]
#     total_conversions = df_journey[df_journey['conversion_flag'] == 'Yes'].shape[0]

#     open_rate = (total_opens / total_sent) * 100 if total_sent > 0 else 0
#     click_rate = (total_clicks / total_opens) * 100 if total_opens > 0 else 0 # Clicks out of opens
#     conversion_rate = (total_conversions / total_sent) * 100 if total_sent > 0 else 0

#     fig = make_subplots(
#         rows=1, cols=3,
#         specs=[[{'type':'indicator'}, {'type':'indicator'}, {'type':'indicator'}]],
#         subplot_titles=('Open Rate', 'Click Rate', 'Conversion Rate')
#     )

#     fig.add_trace(go.Indicator(
#         mode = "gauge+number",
#         value = open_rate,
#         title = {'text': "Open Rate"},
#         gauge = {'axis': {'range': [0, 100]},
#                  'bar': {'color': "#636efa"},
#                  'steps' : [
#                      {'range': [0, 40], 'color': "lightcoral"},
#                      {'range': [40, 70], 'color': "lightgray"},
#                      {'range': [70, 100], 'color': "lightgreen"}],
#                 }), row=1, col=1)

#     fig.add_trace(go.Indicator(
#         mode = "gauge+number",
#         value = click_rate,
#         title = {'text': "Click Rate"},
#         gauge = {'axis': {'range': [0, 100]},
#                  'bar': {'color': "#EF553B"},
#                  'steps' : [
#                      {'range': [0, 10], 'color': "lightcoral"},
#                      {'range': [10, 30], 'color': "lightgray"},
#                      {'range': [30, 100], 'color': "lightgreen"}],
#                 }), row=1, col=2)
    
#     fig.add_trace(go.Indicator(
#         mode = "gauge+number",
#         value = conversion_rate,
#         title = {'text': "Conversion Rate"},
#         gauge = {'axis': {'range': [0, 100]},
#                  'bar': {'color': "#00cc96"},
#                  'steps' : [
#                      {'range': [0, 5], 'color': "lightcoral"},
#                      {'range': [5, 15], 'color': "lightgray"},
#                      {'range': [15, 100], 'color': "lightgreen"}],
#                 }), row=1, col=3)

#     fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20), title_text="Campaign Performance Metrics")
#     return fig

# fig_campaign_kpis = plot_campaign_kpis(df_journey)


# print("Plots generated. Now creating HTML dashboard structure...")

# # --- Generate HTML for each plot ---
# plot_html_sentiment_trend = fig_sentiment_trend.to_html(full_html=False, include_plotlyjs='cdn')
# plot_html_sentiment_gauge = fig_sentiment_gauge.to_html(full_html=False, include_plotlyjs='cdn')
# plot_html_cx_radar = fig_cx_radar.to_html(full_html=False, include_plotlyjs='cdn')
# plot_html_issue_treemap = fig_issue_treemap.to_html(full_html=False, include_plotlyjs='cdn')
# plot_html_queue_sla = fig_queue_sla.to_html(full_html=False, include_plotlyjs='cdn')
# plot_html_kpi_sparklines = fig_kpi_sparklines.to_html(full_html=False, include_plotlyjs='cdn')
# plot_html_nps_tracking = fig_nps_tracking.to_html(full_html=False, include_plotlyjs='cdn')
# plot_html_staff_feedback = fig_staff_feedback.to_html(full_html=False, include_plotlyjs='cdn')
# plot_html_campaign_kpis = fig_campaign_kpis.to_html(full_html=False, include_plotlyjs='cdn')


# # --- HTML Dashboard Structure ---
# html_content = f"""
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>CX Analytics Dashboard</title>
#     <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
#     <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
#     <style>
#         body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; color: #333; }}
#         .dashboard-container {{ padding: 20px; }}
#         .card {{
#             border-radius: 8px;
#             box-shadow: 0 4px 8px rgba(0,0,0,0.05);
#             margin-bottom: 20px;
#             overflow: hidden; /* Important for Plotly charts */
#             background-color: #fff;
#         }}
#         .card-header {{
#             background-color: #007bff;
#             color: white;
#             padding: 10px 15px;
#             font-size: 1.1em;
#             font-weight: 600;
#             border-bottom: 1px solid #ddd;
#         }}
#         .card-body {{ padding: 15px; }}
#         .plotly-graph-div {{ height: 100% !important; width: 100% !important; }} /* Ensure plots fill card */
#         .table-container {{ max-height: 400px; overflow-y: auto; }}
#         /* Specific styling for sparklines */
#         .sparkline-title {{
#             font-size: 0.9em;
#             font-weight: bold;
#             text-align: center;
#             margin-top: 10px;
#         }}
#     </style>
# </head>
# <body>
#     <div class="container-fluid dashboard-container">
#         <h1 class="text-center mb-4 text-primary">Customer Experience Analytics Dashboard</h1>

#         <div class="row">
#             <div class="col-md-4">
#                 <div class="card">
#                     <div class="card-header">Real-Time Sentiment Dashboard</div>
#                     <div class="card-body">
#                         {plot_html_sentiment_gauge}
#                         {plot_html_sentiment_trend}
#                     </div>
#                 </div>
#             </div>

#             <div class="col-md-4">
#                 <div class="card">
#                     <div class="card-header">Call Center Agent Performance</div>
#                     <div class="card-body">
#                         <div class="table-container">
#                             {agent_table_html}
#                         </div>
#                     </div>
#                 </div>
#             </div>

#             <div class="col-md-4">
#                 <div class="card">
#                     <div class="card-header">CX Health Snapshot</div>
#                     <div class="card-body">
#                         {plot_html_cx_radar}
#                     </div>
#                 </div>
#             </div>
#         </div>

#         <div class="row">
#             <div class="col-md-4">
#                 <div class="card">
#                     <div class="card-header">Categorization of Issues</div>
#                     <div class="card-body">
#                         {plot_html_issue_treemap}
#                     </div>
#                 </div>
#             </div>

#             <div class="col-md-4">
#                 <div class="card">
#                     <div class="card-header">Call Queue & SLA Metrics</div>
#                     <div class="card-body">
#                         {plot_html_queue_sla}
#                     </div>
#                 </div>
#             </div>

#             <div class="col-md-4">
#                 <div class="card">
#                     <div class="card-header">Threshold-Based KPI Alerts</div>
#                     <div class="card-body">
#                         {plot_html_kpi_sparklines}
#                     </div>
#                 </div>
#             </div>
#         </div>

#         <div class="row">
#             <div class="col-md-4">
#                 <div class="card">
#                     <div class="card-header">Real-Time NPS Tracking</div>
#                     <div class="card-body">
#                         {plot_html_nps_tracking}
#                     </div>
#                 </div>
#             </div>

#             <div class="col-md-4">
#                 <div class="card">
#                     <div class="card-header">Staff/Departmental Feedback</div>
#                     <div class="card-body">
#                         {plot_html_staff_feedback}
#                     </div>
#                 </div>
#             </div>

#             <div class="col-md-4">
#                 <div class="card">
#                     <div class="card-header">Campaign Performance KPIs</div>
#                     <div class="card-body">
#                         {plot_html_campaign_kpis}
#                     </div>
#                 </div>
#             </div>
#         </div>
#     </div>
# </body>
# </html>
# """

# # Save the HTML to a file
# dashboard_filename = "cx_dashboard.html"
# with open(dashboard_filename, "w") as f:
#     f.write(html_content)

# print(f"\nDashboard HTML saved to {dashboard_filename}. Open this file in your browser to view the dashboard.")
# print("Remember to have 'synthetic_transaction_data.csv', 'sentiment.csv', 'journey_entry.csv', and 'after_sales.csv' in the same directory as this script.")