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
# def create_agent_performance_table(df_after_sales):
#     agent_performance = df_after_sales.groupby('agent_id').agg(
#         calls_handled=('interaction_id', 'count'),
#         total_duration_minutes=('resolution_time_minutes', 'sum')
#     ).reset_index()
#     agent_performance['avg_duration'] = (agent_performance['total_duration_minutes'] / agent_performance['calls_handled']).round(2)
#     agent_performance['status'] = [random.choice(['Online', 'Offline', 'Busy']) for _ in range(len(agent_performance))]

#     # Limit to top 8 agents for better display
#     agent_performance = agent_performance.nlargest(8, 'calls_handled')
    
#     fig = go.Figure(data=[go.Table(
#         header=dict(values=['Agent', 'Status', 'Calls', 'Avg Duration'],
#                    fill_color='#007bff',
#                    font=dict(color='white', size=12),
#                    align='center'),
#         cells=dict(values=[agent_performance.agent_id, 
#                           agent_performance.status,
#                           agent_performance.calls_handled, 
#                           agent_performance.avg_duration],
#                   fill_color='lightgrey',
#                   align='center',
#                   font_size=10))
#     ])
    
#     fig.update_layout(
#         margin=dict(l=20, r=20, t=40, b=20),
#         height=280,
#         title='Agent Performance',
#         title_font_size=14
#     )
    
#     return fig.to_html(full_html=False, include_plotlyjs='cdn')

def create_agent_performance_table(df_after_sales):
    """
    Generates a horizontal bar chart showing the top 10 agents by calls handled.

    Args:
        df_after_sales (pd.DataFrame): DataFrame containing 'agent_id',
                                       'interaction_id', and 'resolution_time_minutes'.

    Returns:
        str: HTML string of the Plotly horizontal bar chart.
    """
    # Group by agent_id and aggregate performance metrics
    agent_performance = df_after_sales.groupby('agent_id').agg(
        calls_handled=('interaction_id', 'count'),
        total_duration_minutes=('resolution_time_minutes', 'sum')
    ).reset_index()

    # Calculate average duration per call
    agent_performance['avg_duration'] = (
        agent_performance['total_duration_minutes'] / agent_performance['calls_handled']
    ).round(2)

    # Add a 'status' column (optional, as it's less relevant for a bar chart directly)
    agent_performance['status'] = [random.choice(['Online', 'Offline', 'Busy']) for _ in range(len(agent_performance))]

    # Limit to top 10 agents based on 'calls_handled'
    # Sort for better visualization in the bar chart (highest at the top)
    agent_performance = agent_performance.nlargest(10, 'calls_handled').sort_values(
        'calls_handled', ascending=True
    )

    # Create the horizontal bar chart
    fig = px.bar(
        agent_performance,
        x='calls_handled',      # Metric on the x-axis
        y='agent_id',           # Agent IDs on the y-axis
        orientation='h',        # Make it a horizontal bar chart
        title='Top 10 Agent Performance by Calls Handled', # Updated title
        labels={
            'calls_handled': 'Number of Calls Handled',
            'agent_id': 'Agent ID'
        },
        hover_data={
            'total_duration_minutes': ':.2f', # Show total duration in hover
            'avg_duration': ':.2f',           # Show average duration in hover
            'status': True                    # Show status in hover
        },
        color_discrete_sequence=px.colors.qualitative.Plotly # Choose a nice color sequence
    )

    # Update layout for better appearance
    fig.update_layout(
        margin=dict(l=40, r=40, t=60, b=40),
        height=400, # Increased height for 10 bars
        title_font_size=16,
        xaxis_title="Number of Calls Handled",
        yaxis_title="Agent ID",
        yaxis_categoryorder='total ascending' # Ensure the bars are ordered correctly (smallest at bottom, largest at top)
    )

    # Make y-axis labels readable if there are many agents
    fig.update_yaxes(tickangle=-45)

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
    categories = ['Sentiment', 'Issues', 'NPS', 'Product']

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
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta

def plot_queue_sla_metrics(df_after_sales):
    # Convert date column
    df_after_sales['interaction_date'] = pd.to_datetime(df_after_sales['interaction_date'], errors='coerce')

    # Filter for Call/Chat interactions with valid dates
    df = df_after_sales[
        df_after_sales['interaction_type'].isin(["Call", "Chat"])
    ].dropna(subset=['interaction_date']).copy()

    # Last 30 days filter
    max_date = df['interaction_date'].max()
    start_date = max_date - timedelta(days=29)
    df = df[df['interaction_date'].between(start_date, max_date)]

    # Aggregate daily metrics
    daily_metrics = df.groupby(pd.Grouper(key='interaction_date', freq='D')).agg(
        queue_calls=('queue_time_seconds', lambda x: (x > 0).sum()),
        sla_met=('sla_met', lambda x: (x == 'Yes').sum()),
        total_calls=('interaction_id', 'count')
    ).reset_index()

    daily_metrics['sla_percentage'] = ((daily_metrics['sla_met'] / daily_metrics['total_calls']) * 100).fillna(0).round(2)

    # --- Plotting ---
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Left Y-axis: Queue Calls (Line)
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

    # Right Y-axis: SLA % (Bar)
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

    # SLA Target Line (on right Y-axis)
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

    # Layout and axes
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

def plot_nps_tracking(df_after_sales):
    """
    Enhanced NPS tracking with customer details and 6-month activity history
    """
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from datetime import datetime, timedelta
    
    # Ensure 'interaction_date' is datetime type
    df_after_sales['interaction_date'] = pd.to_datetime(df_after_sales['interaction_date'])

    # Get the most recent date in the data
    latest_date = df_after_sales['interaction_date'].max()

    # Calculate the date 30 days prior to the latest date for NPS trend
    start_date_30 = latest_date - pd.Timedelta(days=30)
    
    # Calculate the date 6 months prior for customer activity analysis
    start_date_6m = latest_date - pd.Timedelta(days=180)

    # Filter data for the last 30 days for NPS trend
    df_last_30_days = df_after_sales[
        (df_after_sales['interaction_date'] >= start_date_30) & 
        (df_after_sales['interaction_date'] <= latest_date)
    ]

    # Filter data for the last 6 months for customer activity
    df_last_6_months = df_after_sales[
        (df_after_sales['interaction_date'] >= start_date_6m) & 
        (df_after_sales['interaction_date'] <= latest_date)
    ]

    # Group by date and calculate NPS with customer details
    nps_detailed = []
    
    for date, group in df_last_30_days.groupby(pd.Grouper(key='interaction_date', freq='D')):
        if len(group) > 0:
            avg_nps = group['nps_score'].mean()
            
            # Get customer details for this date
            customer_details = group.groupby('customer_id').agg({
                'nps_score': 'mean',
                'interaction_type': lambda x: ', '.join(x.unique()),
                'issue_category': lambda x: ', '.join(x.unique()),
                'resolution_status': lambda x: ', '.join(x.unique())
            }).reset_index()
            
            # Create customer summary string
            customer_list = []
            for _, customer in customer_details.iterrows():
                customer_list.append(
                    f"Customer {customer['customer_id']}: NPS {customer['nps_score']:.1f} "
                    f"({customer['interaction_type']}) - {customer['issue_category']}"
                )
            
            # Get 6-month activity for customers who contributed to this day's NPS
            customer_ids = group['customer_id'].unique()
            customer_activity_6m = df_last_6_months[
                df_last_6_months['customer_id'].isin(customer_ids)
            ].groupby('customer_id').agg({
                'interaction_id': 'count',
                'nps_score': 'mean',
                'resolution_status': lambda x: (x == 'Resolved').sum(),
                'interaction_type': lambda x: ', '.join(x.unique()[:3])  # Top 3 interaction types
            }).reset_index()
            
            customer_activity_6m['resolution_rate'] = (
                customer_activity_6m['resolution_status'] / customer_activity_6m['interaction_id'] * 100
            ).round(1)
            
            # Create 6-month activity summary
            activity_summary = []
            for _, activity in customer_activity_6m.iterrows():
                activity_summary.append(
                    f"Customer {activity['customer_id']}: {activity['interaction_id']} interactions, "
                    f"Avg NPS: {activity['nps_score']:.1f}, Resolution: {activity['resolution_rate']}%"
                )
            
            nps_detailed.append({
                'Date': date,
                'NPS': avg_nps,
                'Customer_Count': len(customer_details),
                'Customer_Details': '<br>'.join(customer_list),
                'Activity_6M': '<br>'.join(activity_summary)
            })

    nps_data = pd.DataFrame(nps_detailed)
    
    if nps_data.empty:
        return go.Figure().to_html(full_html=False, include_plotlyjs='cdn')

    # Create the enhanced plot
    fig = go.Figure()

    # Add the main NPS trend line
    fig.add_trace(go.Scatter(
        x=nps_data['Date'],
        y=nps_data['NPS'],
        mode='lines+markers',
        name='Daily NPS',
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=8, color='#2ca02c'),
        customdata=list(zip(
            nps_data['Customer_Count'],
            nps_data['Customer_Details'],
            nps_data['Activity_6M']
        )),
        hovertemplate=(
            '<b>Date:</b> %{x|%b %d, %Y}<br>'
            '<b>Average NPS:</b> %{y:.2f}<br>'
            '<b>Customers:</b> %{customdata[0]}<br>'
            '<b>Customer Details:</b><br>%{customdata[1]}<br>'
            '<b>6-Month Activity:</b><br>%{customdata[2]}<br>'
            '<extra></extra>'
        )
    ))

    # Add NPS benchmark lines
    fig.add_hline(y=9, line_dash="dash", line_color="green", 
                  annotation_text="Promoter (9-10)", annotation_position="top left")
    fig.add_hline(y=7, line_dash="dash", line_color="orange", 
                  annotation_text="Passive (7-8)", annotation_position="top left")
    fig.add_hline(y=6, line_dash="dash", line_color="red", 
                  annotation_text="Detractor (0-6)", annotation_position="bottom left")

    # Enhanced layout
    fig.update_layout(
        title={
            'text': 'Daily NPS Score with Customer Details (Last 30 Days)',
            'x': 0.5,
            'xanchor': 'center'
        },
        margin=dict(l=40, r=40, t=80, b=40),
        height=350,
        title_font_size=14,
        xaxis_title="Date",
        yaxis_title="NPS Score",
        yaxis=dict(range=[0, 10]),
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

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

import pandas as pd
import plotly.graph_objects as go
import plotly.express
# --- 9. Campaign KPIs ---


# import pandas as pd
# import plotly.graph_objects as go
# import plotly.express as px

# def plot_branch_performance(df_transactions, df_after_sales):
#     """
#     Creates two gauge charts to visualize branch performance metrics:
#     1. Issues resolved by a specific branch.
#     2. Revenue performance of the best-performing branch.

#     Args:
#         df_transactions (pd.DataFrame): DataFrame with transaction data.
#         df_after_sales (pd.DataFrame): DataFrame with after-sales data.

#     Returns:
#         str: HTML string of the combined Plotly figure.
#     """
#     # Helper to return a blank plot with a message
#     def create_blank_plot(message):
#         fig = go.Figure().update_layout(title_text=message,
#                                          margin=dict(l=20, r=20, t=60, b=20), height=250, title_font_size=14)
#         return fig.to_html(full_html=False, include_plotlyjs='cdn')
        
#     # --- Part 1: Calculate Resolved Issues for a specific branch ---
#     # We will choose a branch to display. Let's pick a random one for demonstration.
#     # In a real dashboard, you might pass a branch_id as an argument or let the user select one.
#     if df_after_sales.empty or 'branch_id' not in df_after_sales.columns:
#         print("Warning: Skipping resolved issues gauge. 'branch_id' column not found in df_after_sales.")
#         resolved_issues_value = 0
#         total_issues_value = 100
#         branch_id_to_show = "N/A"
#     else:
#         # Get a list of all branches to pick one
#         all_branches = df_after_sales['branch_id'].unique()
#         if len(all_branches) > 0:
#             branch_id_to_show = all_branches[0] # Pick the first branch for display
#         else:
#             branch_id_to_show = "N/A"
        
#         branch_issues = df_after_sales[df_after_sales['branch_id'] == branch_id_to_show]
#         resolved_issues_value = (branch_issues['resolution_status'] == 'Resolved').sum()
#         total_issues_value = branch_issues.shape[0]
#         if total_issues_value == 0:
#             total_issues_value = 1 # Avoid division by zero


#     # --- Part 2: Calculate Revenue for the Best Performing Branch ---
#     if df_transactions.empty or 'branch_id' not in df_transactions.columns:
#         print("Warning: Skipping best-performing branch gauge. Required columns are missing in df_transactions.")
#         best_branch_revenue = 0
#         best_branch_id = "N/A"
#         total_revenue = 100
#     else:
#         df_transactions['total_revenue'] = df_transactions['grand_total']
#         branch_revenue = df_transactions.groupby('branch_id')['total_revenue'].sum().reset_index()
        
#         best_branch_row = branch_revenue.loc[branch_revenue['total_revenue'].idxmax()]
#         best_branch_id = best_branch_row['branch_id']
#         best_branch_revenue = best_branch_row['total_revenue']
#         total_revenue = df_transactions['total_revenue'].sum()


#     # --- Create Subplots for the two Gauges ---
#     fig = make_subplots(
#         rows=1, cols=2,
#         specs=[[{'type': 'indicator'}, {'type': 'indicator'}]],
#         subplot_titles=(f'Issues Resolved by Branch {branch_id_to_show}', f'Revenue for Best Performing Branch {best_branch_id}')
#     )

#     # Gauge 1: Issues Resolved
#     fig.add_trace(go.Indicator(
#         mode="gauge+number",
#         value=resolved_issues_value,
#         title={'text': "Issues Resolved", 'font': {'size': 14}},
#         gauge={'axis': {'range': [None, total_issues_value]},
#                'bar': {'color': "#2ca02c"}, # Green for good
#                'steps': [
#                    {'range': [0, total_issues_value * 0.5], 'color': "lightgray"},
#                    {'range': [total_issues_value * 0.5, total_issues_value], 'color': "gray"}
#                ],
#                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': total_issues_value * 0.8}}
#     ), row=1, col=1)

#     # Gauge 2: Best Performing Branch Revenue
#     fig.add_trace(go.Indicator(
#         mode="gauge+number",
#         value=best_branch_revenue,
#         title={'text': "Total Revenue", 'font': {'size': 14}},
#         gauge={'axis': {'range': [None, total_revenue]},
#                'bar': {'color': "#1f77b4"}, # Blue for good
#                'steps': [
#                    {'range': [0, total_revenue * 0.5], 'color': "lightgray"},
#                    {'range': [total_revenue * 0.5, total_revenue], 'color': "gray"}
#                ]}
#     ), row=1, col=2)

#     fig.update_layout(
#         title_text="<b>Branch Performance Overview</b>",
#         title_font_size=16,
#         margin=dict(l=20, r=20, t=60, b=20),
#         height=300,
#         paper_bgcolor='white',
#         plot_bgcolor='white',
#         font=dict(size=12, color='#333'),
#         showlegend=False
#     )
#     fig.update_traces(
#         number={'font': {'size': 20}},
#         title_font={'size': 16}
#     )
#     fig.update_annotations(font_size=14)

#     # Return the HTML representation of the figure
#     return fig.to_html(full_html=False, include_plotlyjs='cdn')

def plot_branch_performance(df_transactions, df_after_sales):
    """
    Creates a combined subplot with two gauge charts and a bar chart
    to visualize overall branch performance.

    Args:
        df_transactions (pd.DataFrame): DataFrame with transaction data.
        df_after_sales (pd.DataFrame): DataFrame with after-sales data.

    Returns:
        str: HTML string of the combined Plotly figure.
    """
    # Helper to return a blank plot with a message
    def create_blank_plot(message):
        fig = go.Figure().update_layout(title_text=message,
                                         margin=dict(l=20, r=20, t=60, b=20), height=300, title_font_size=14)
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    # --- Part 1: Calculate Resolved Issues for a specific branch ---
    resolved_issues_value = 0
    total_issues_value = 1
    branch_id_to_show = "N/A"
    
    # **Validation and data linking for after_sales data**
    if not df_after_sales.empty and 'branch_id' in df_after_sales.columns and 'resolution_status' in df_after_sales.columns:
        all_branches = df_after_sales['branch_id'].unique()
        if len(all_branches) > 0:
            branch_id_to_show = all_branches[0] # Pick the first branch for display
            branch_issues = df_after_sales[df_after_sales['branch_id'] == branch_id_to_show]
            resolved_issues_value = (branch_issues['resolution_status'] == 'Resolved').sum()
            total_issues_value = branch_issues.shape[0]
            if total_issues_value == 0:
                total_issues_value = 1 # Avoid division by zero
    else:
        print("Warning: Skipping resolved issues gauge. 'branch_id' or 'resolution_status' column not found in df_after_sales.")

    # --- Part 2: Calculate Revenue for all branches ---
    best_branch_revenue = 0
    best_branch_id = "N/A"
    total_revenue = 1
    
    # **Validation and data linking for transactions data**
    if not df_transactions.empty and 'branch_id' in df_transactions.columns and 'grand_total' in df_transactions.columns:
        df_transactions['total_revenue'] = df_transactions['grand_total']
        branch_revenue = df_transactions.groupby('branch_id')['total_revenue'].sum().reset_index()
        
        if not branch_revenue.empty:
            best_branch_row = branch_revenue.loc[branch_revenue['total_revenue'].idxmax()]
            best_branch_id = best_branch_row['branch_id']
            best_branch_revenue = best_branch_row['total_revenue']
            total_revenue = df_transactions['total_revenue'].sum()
            # Sort the branches for the bar chart
            branch_revenue_sorted = branch_revenue.sort_values(by='total_revenue', ascending=True)
        else:
            branch_revenue_sorted = pd.DataFrame(columns=['branch_id', 'total_revenue'])
            
    else:
        print("Warning: Skipping branch revenue charts. Required columns are missing in df_transactions.")
        branch_revenue_sorted = pd.DataFrame(columns=['branch_id', 'total_revenue'])

    # --- Create Subplots for the three plots ---
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'xy'}]],
        subplot_titles=(
            f'Issues Resolved by Branch {branch_id_to_show}',
            f'Revenue for Best Branch {best_branch_id}',
            'Revenue Contribution by Branch'
        )
    )

    # --- Plot Gauge 1: Issues Resolved ---
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=resolved_issues_value,
        title={'text': "Issues Resolved", 'font': {'size': 14}},
        gauge={'axis': {'range': [None, total_issues_value]},
               'bar': {'color': "#2ca02c"}, # Green for good
               'steps': [
                   {'range': [0, total_issues_value * 0.5], 'color': "lightgray"},
                   {'range': [total_issues_value * 0.5, total_issues_value], 'color': "gray"}
               ],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': total_issues_value * 0.8}}
    ), row=1, col=1)

    # --- Plot Gauge 2: Best Performing Branch Revenue ---
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=best_branch_revenue,
        title={'text': "Total Revenue", 'font': {'size': 14}},
        gauge={'axis': {'range': [None, total_revenue]},
               'bar': {'color': "#1f77b4"}, # Blue for good
               'steps': [
                   {'range': [0, total_revenue * 0.5], 'color': "lightgray"},
                   {'range': [total_revenue * 0.5, total_revenue], 'color': "gray"}
               ]}
    ), row=1, col=2)
    
    # --- Plot Bar Chart 3: Revenue Contribution by Branch ---
    fig.add_trace(go.Bar(
        x=branch_revenue_sorted['total_revenue'],
        y=branch_revenue_sorted['branch_id'],
        orientation='h',
        marker_color='#FF8C00'
    ), row=1, col=3)

    fig.update_layout(
        title_text="<b>Branch Performance Overview</b>",
        title_font_size=16,
        margin=dict(l=20, r=20, t=60, b=20),
        height=350,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(size=12, color='#333'),
        showlegend=False
    )
    
    # Update axes for the new bar chart
    fig.update_xaxes(title_text='Total Revenue', row=1, col=3)
    fig.update_yaxes(title_text='Branch ID', row=1, col=3)
    
    fig.update_traces(
        selector=dict(type='indicator'),
        number={'font': {'size': 20}},
        title_font={'size': 16}
    )
    
    fig.update_annotations(font_size=14)

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
    plot_html_campaign_kpis = plot_branch_performance(df_transactions, df_after_sales)

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