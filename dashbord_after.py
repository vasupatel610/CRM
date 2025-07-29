import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random

print("Generating data for Dashboard Visualizations...")

# --- Configuration for Data Generation (ensure consistency with previous scripts) ---
# Assuming these CSVs are already generated from previous steps
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
    # Exit or create dummy data for demonstration if files are missing
    exit() # Exiting for now as data consistency is key

# --- 1. Real-Time Sentiment Dashboard (Summary) ---
# Line Chart: Average sentiment over time (e.g., last 30 days)
# Gauge Chart: Overall current sentiment

def plot_realtime_sentiment(df_sentiment):
    recent_sentiment = df_sentiment[df_sentiment['date'] > (datetime.now() - timedelta(days=30))]
    if recent_sentiment.empty:
        return go.Figure() # Return empty figure if no recent data

    sentiment_trend = recent_sentiment.groupby(pd.Grouper(key='date', freq='D'))['sentiment_score'].mean().reset_index()
    sentiment_trend.columns = ['Date', 'Average Sentiment']

    fig_line = px.line(sentiment_trend, x='Date', y='Average Sentiment',
                       title='Last 30 Days Average Sentiment',
                       color_discrete_sequence=['#1f77b4']) # blue
    fig_line.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=250)

    current_sentiment_score = recent_sentiment['sentiment_score'].mean() if not recent_sentiment.empty else 0.0
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = current_sentiment_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Current Overall Sentiment"},
        gauge = {'axis': {'range': [-1, 1]},
                 'bar': {'color': "#1f77b4"},
                 'steps' : [
                     {'range': [-1, -0.3], 'color': "lightcoral"},
                     {'range': [-0.3, 0.3], 'color': "lightgray"},
                     {'range': [0.3, 1], 'color': "lightgreen"}],
                 'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.5}} # Example threshold
    ))
    fig_gauge.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=250)
    return fig_line, fig_gauge

fig_sentiment_trend, fig_sentiment_gauge = plot_realtime_sentiment(df_sentiment)


# --- 2. Call Center Agent Performance (Table with Conditional Formatting) ---
def create_agent_performance_table(df_after_sales):
    agent_performance = df_after_sales.groupby('agent_id').agg(
        calls_handled=('interaction_id', 'count'),
        total_duration_minutes=('resolution_time_minutes', 'sum')
    ).reset_index()
    agent_performance['average_call_duration_minutes'] = agent_performance['total_duration_minutes'] / agent_performance['calls_handled']

    # Simulate online/offline status for agents (random for this static report)
    agent_performance['availability_status'] = [random.choice(['Online', 'Offline', 'Busy']) for _ in range(len(agent_performance))]

    # Select relevant columns and round duration
    agent_performance = agent_performance[['agent_id', 'availability_status', 'calls_handled', 'average_call_duration_minutes']]
    agent_performance['average_call_duration_minutes'] = agent_performance['average_call_duration_minutes'].round(2)

    # Convert to HTML table with basic styling (Plotly's Table is an option, but pandas styling for simple HTML is good)
    # For advanced conditional formatting, external CSS or more complex JS might be needed with raw HTML
    styled_table = agent_performance.style \
        .background_gradient(cmap='Blues', subset=['calls_handled']) \
        .format({'average_call_duration_minutes': "{:.2f} mins"}) \
        .to_html(classes='table table-striped table-hover', table_id='agent_performance_table')
    return styled_table

agent_table_html = create_agent_performance_table(df_after_sales)


# --- 3. CX Health Snapshot Radar Chart ---
def plot_cx_radar_chart(df_after_sales, df_sentiment):
    # Select a random customer for the snapshot
    customer_id = random.choice(df_after_sales['customer_id'].unique())
    cust_interactions = df_after_sales[df_after_sales['customer_id'] == customer_id]
    cust_sentiment = df_sentiment[df_sentiment['customer_id'] == customer_id]

    if cust_interactions.empty and cust_sentiment.empty:
        return go.Figure()

    # Key CX indicators for the individual
    recent_sentiment_score = cust_sentiment['sentiment_score'].mean() if not cust_sentiment.empty else 0
    num_open_issues = cust_interactions[cust_interactions['resolution_status'] != 'Resolved'].shape[0]
    time_since_last_interaction = (datetime.now() - cust_interactions['interaction_date'].max()).days if not cust_interactions.empty else 365 # Default if no interactions
    recent_nps_score = cust_interactions['nps_score'].mean() if not cust_interactions.empty else 5 # Default neutral if no NPS
    # product_ownership_flag is a simple 'Yes' in after_sales.csv, could derive more complex if needed
    product_ownership_flag = 1 if not cust_interactions.empty else 0 # 1 if they have interactions, 0 otherwise

    # Normalize values for radar chart (0-1 scale, higher is better)
    # Assuming: sentiment: -1 to 1, open_issues: 0 to max, time_since_last: 0 to 365, nps: 0 to 10
    # Higher is better for all except open_issues and time_since_last_interaction (where lower is better, so invert)
    metrics = [
        (recent_sentiment_score + 1) / 2, # Scale sentiment to 0-1
        1 - (min(num_open_issues, 5) / 5), # Invert & Cap open issues at 5
        1 - (min(time_since_last_interaction, 90) / 90), # Invert & Cap days at 90 for scaling
        recent_nps_score / 10, # Scale NPS to 0-1
        product_ownership_flag # Already 0 or 1
    ]
    categories = ['Sentiment', 'No. Open Issues', 'Recency of Interaction', 'NPS', 'Product Ownership']

    fig = go.Figure(data=go.Scatterpolar(
        r=metrics,
        theta=categories,
        fill='toself',
        name=f'Customer: {customer_id}'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title=f'CX Health Snapshot for {customer_id}',
        margin=dict(l=20, r=20, t=60, b=20), height=300
    )
    return fig

fig_cx_radar = plot_cx_radar_chart(df_after_sales, df_sentiment)


# --- 4. Categorization of Issues Treemap chart ---
def plot_issue_treemap(df_after_sales):
    # Aggregate sentiment by issue category and sentiment category
    df_issues = df_after_sales.groupby(['issue_category', 'sentiment_category']).size().reset_index(name='count')
    
    # Calculate total for percentage
    total_count = df_issues['count'].sum()
    df_issues['percentage'] = (df_issues['count'] / total_count) * 100

    fig = px.treemap(df_issues, path=[px.Constant("All Issues"), 'issue_category', 'sentiment_category'],
                     values='count',
                     color='sentiment_category',
                     color_discrete_map={
                         '(?)': 'lightgray',
                         'Positive': 'lightgreen',
                         'Neutral': 'lightgray',
                         'Negative': 'lightcoral'
                     },
                     title='Categorization of Issues by Sentiment',
                     labels={'count': 'Number of Interactions', 'sentiment_category': 'Sentiment'}
                    )
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=300)
    fig.update_traces(hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Sentiment: %{color}<extra></extra>')
    return fig

fig_issue_treemap = plot_issue_treemap(df_after_sales)


# --- 5. Call Queue & SLA Metrics Dual-Axis Chart ---
def plot_queue_sla_metrics(df_after_sales):
    # Filter for interactions that are likely calls/chats and could have queue time
    call_interactions = df_after_sales[df_after_sales['interaction_type'].isin(["Call", "Chat"])].copy()

    # Aggregate by date (daily trends)
    daily_metrics = call_interactions.groupby(pd.Grouper(key='interaction_date', freq='D')).agg(
        num_calls_in_queue=('queue_time_seconds', lambda x: (x > 0).sum()), # Count calls with queue time
        calls_meeting_sla=('sla_met', lambda x: (x == 'Yes').sum()),
        total_calls=('interaction_id', 'count')
    ).reset_index()

    daily_metrics['percentage_meeting_sla'] = (daily_metrics['calls_meeting_sla'] / daily_metrics['total_calls']) * 100
    daily_metrics['percentage_meeting_sla'] = daily_metrics['percentage_meeting_sla'].fillna(0).round(2)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add calls in queue (Y-axis 1)
    fig.add_trace(
        go.Bar(x=daily_metrics['interaction_date'], y=daily_metrics['num_calls_in_queue'], name='Calls in Queue'),
        secondary_y=False,
    )

    # Add percentage meeting SLA (Y-axis 2)
    fig.add_trace(
        go.Line(x=daily_metrics['interaction_date'], y=daily_metrics['percentage_meeting_sla'], name='Percentage Meeting SLA', mode='lines+markers'),
        secondary_y=True,
    )

    fig.update_layout(
        title_text='Daily Call Queue and SLA Metrics',
        margin=dict(l=20, r=20, t=60, b=20), height=300
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Number of Calls in Queue", secondary_y=False)
    fig.update_yaxes(title_text="Percentage Meeting SLA (%)", secondary_y=True, range=[0, 100])
    return fig

fig_queue_sla = plot_queue_sla_metrics(df_after_sales)


# --- 6. Threshold-Based KPI Alerts Sparklines chart ---
# This is a conceptual chart type. We'll simulate a few sparklines for key KPIs.
def plot_kpi_sparklines(df_after_sales, df_sentiment):
    # Example KPIs: Average Sentiment, SLA Met %, FCR %
    
    # Daily average sentiment
    daily_sentiment = df_sentiment.groupby(pd.Grouper(key='date', freq='D'))['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['Date', 'Avg_Sentiment']
    
    # Daily SLA Met %
    daily_sla = df_after_sales.groupby(pd.Grouper(key='interaction_date', freq='D')).agg(
        total_interactions=('interaction_id', 'count'),
        sla_met_count=('sla_met', lambda x: (x == 'Yes').sum())
    ).reset_index()
    daily_sla['SLA_Met_Percentage'] = (daily_sla['sla_met_count'] / daily_sla['total_interactions']) * 100
    daily_sla['SLA_Met_Percentage'] = daily_sla['SLA_Met_Percentage'].fillna(0)

    # Daily FCR %
    daily_fcr = df_after_sales.groupby(pd.Grouper(key='interaction_date', freq='D')).agg(
        total_interactions=('interaction_id', 'count'),
        fcr_count=('is_first_contact_resolution', lambda x: (x == 'Yes').sum())
    ).reset_index()
    daily_fcr['FCR_Percentage'] = (daily_fcr['fcr_count'] / daily_fcr['total_interactions']) * 100
    daily_fcr['FCR_Percentage'] = daily_fcr['FCR_Percentage'].fillna(0)

    # Combine for common dates
    kpi_data = daily_sentiment.merge(daily_sla, left_on='Date', right_on='interaction_date', how='outer')
    kpi_data = kpi_data.merge(daily_fcr, left_on='Date', right_on='interaction_date', how='outer')
    kpi_data = kpi_data.sort_values('Date').fillna(method='ffill').tail(30) # Last 30 days
    
    if kpi_data.empty:
        return go.Figure()

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1)

    # Sparkline for Avg Sentiment
    fig.add_trace(go.Scatter(x=kpi_data['Date'], y=kpi_data['Avg_Sentiment'], mode='lines', name='Avg Sentiment',
                             line=dict(color='blue', width=1)), row=1, col=1)
    fig.add_hline(y=0.3, line_dash="dash", line_color="green", row=1, col=1) # Threshold example
    fig.update_yaxes(title_text="Sentiment", row=1, col=1, showticklabels=False, range=[-1,1])
    fig.update_xaxes(showticklabels=False, row=1, col=1) # Hide x-axis labels for sparkline

    # Sparkline for SLA Met %
    fig.add_trace(go.Scatter(x=kpi_data['Date'], y=kpi_data['SLA_Met_Percentage'], mode='lines', name='SLA Met %',
                             line=dict(color='orange', width=1)), row=2, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1) # Threshold example
    fig.update_yaxes(title_text="SLA %", row=2, col=1, showticklabels=False, range=[0,100])
    fig.update_xaxes(showticklabels=False, row=2, col=1)

    # Sparkline for FCR %
    fig.add_trace(go.Scatter(x=kpi_data['Date'], y=kpi_data['FCR_Percentage'], mode='lines', name='FCR %',
                             line=dict(color='green', width=1)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="purple", row=3, col=1) # Threshold example
    fig.update_yaxes(title_text="FCR %", row=3, col=1, showticklabels=False, range=[0,100])
    fig.update_xaxes(showticklabels=False, row=3, col=1) # Hide x-axis labels for sparkline

    fig.update_layout(height=300, showlegend=False, title_text='Key KPI Sparklines (Last 30 Days)',
                      margin=dict(l=20, r=20, t=60, b=20))
    return fig

fig_kpi_sparklines = plot_kpi_sparklines(df_after_sales, df_sentiment)


# --- 7. Real-Time NPS Tracking Line Chart ---
def plot_nps_tracking(df_after_sales):
    nps_data = df_after_sales.groupby(pd.Grouper(key='interaction_date', freq='D'))['nps_score'].mean().reset_index()
    nps_data.columns = ['Date', 'Average NPS']
    
    fig = px.line(nps_data, x='Date', y='Average NPS', title='Daily Average NPS Score',
                  color_discrete_sequence=['#2ca02c']) # green
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=250)
    fig.update_yaxes(range=[0,10]) # NPS range 0-10
    return fig

fig_nps_tracking = plot_nps_tracking(df_after_sales)


# --- 8. Staff/Departmental Feedback grouped bar chart ---
def plot_staff_feedback(df_after_sales):
    # For simplicity, let's group by agent_id and feedback_score_agent
    feedback_counts = df_after_sales.groupby(['agent_id', 'feedback_score_agent']).size().unstack(fill_value=0)
    
    # If there are many agents, pick top N or sample
    if len(feedback_counts) > 10: # Limit to top 10 agents by interactions
        top_agents = df_after_sales['agent_id'].value_counts().nlargest(10).index
        feedback_counts = feedback_counts.loc[top_agents]

    fig = px.bar(feedback_counts, x=feedback_counts.index, y=feedback_counts.columns,
                 barmode='group',
                 title='Agent Feedback Scores Distribution (1-5)',
                 labels={'agent_id': 'Agent ID', 'value': 'Count', 'feedback_score_agent': 'Feedback Score'},
                 color_discrete_sequence=px.colors.sequential.Plasma) # A nice color sequence
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=300)
    return fig

fig_staff_feedback = plot_staff_feedback(df_after_sales)


# --- 9. Custom Dashboard Builder (using Campaign KPIs as example) ---
# Gauge Charts for open/click/conversion rates (from df_journey)
def plot_campaign_kpis(df_journey):
    # Calculate overall rates from journey data
    total_sent = df_journey[df_journey['stage'] == 'sent'].shape[0]
    total_opens = df_journey[df_journey['campaign_open'] == 'Yes'].shape[0]
    total_clicks = df_journey[df_journey['campaign_click'] == 'Yes'].shape[0]
    total_conversions = df_journey[df_journey['conversion_flag'] == 'Yes'].shape[0]

    open_rate = (total_opens / total_sent) * 100 if total_sent > 0 else 0
    click_rate = (total_clicks / total_opens) * 100 if total_opens > 0 else 0 # Clicks out of opens
    conversion_rate = (total_conversions / total_sent) * 100 if total_sent > 0 else 0

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type':'indicator'}, {'type':'indicator'}, {'type':'indicator'}]],
        subplot_titles=('Open Rate', 'Click Rate', 'Conversion Rate')
    )

    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = open_rate,
        title = {'text': "Open Rate"},
        gauge = {'axis': {'range': [0, 100]},
                 'bar': {'color': "#636efa"},
                 'steps' : [
                     {'range': [0, 40], 'color': "lightcoral"},
                     {'range': [40, 70], 'color': "lightgray"},
                     {'range': [70, 100], 'color': "lightgreen"}],
                }), row=1, col=1)

    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = click_rate,
        title = {'text': "Click Rate"},
        gauge = {'axis': {'range': [0, 100]},
                 'bar': {'color': "#EF553B"},
                 'steps' : [
                     {'range': [0, 10], 'color': "lightcoral"},
                     {'range': [10, 30], 'color': "lightgray"},
                     {'range': [30, 100], 'color': "lightgreen"}],
                }), row=1, col=2)
    
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = conversion_rate,
        title = {'text': "Conversion Rate"},
        gauge = {'axis': {'range': [0, 100]},
                 'bar': {'color': "#00cc96"},
                 'steps' : [
                     {'range': [0, 5], 'color': "lightcoral"},
                     {'range': [5, 15], 'color': "lightgray"},
                     {'range': [15, 100], 'color': "lightgreen"}],
                }), row=1, col=3)

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20), title_text="Campaign Performance Metrics")
    return fig

fig_campaign_kpis = plot_campaign_kpis(df_journey)


print("Plots generated. Now creating HTML dashboard structure...")

# --- Generate HTML for each plot ---
plot_html_sentiment_trend = fig_sentiment_trend.to_html(full_html=False, include_plotlyjs='cdn')
plot_html_sentiment_gauge = fig_sentiment_gauge.to_html(full_html=False, include_plotlyjs='cdn')
plot_html_cx_radar = fig_cx_radar.to_html(full_html=False, include_plotlyjs='cdn')
plot_html_issue_treemap = fig_issue_treemap.to_html(full_html=False, include_plotlyjs='cdn')
plot_html_queue_sla = fig_queue_sla.to_html(full_html=False, include_plotlyjs='cdn')
plot_html_kpi_sparklines = fig_kpi_sparklines.to_html(full_html=False, include_plotlyjs='cdn')
plot_html_nps_tracking = fig_nps_tracking.to_html(full_html=False, include_plotlyjs='cdn')
plot_html_staff_feedback = fig_staff_feedback.to_html(full_html=False, include_plotlyjs='cdn')
plot_html_campaign_kpis = fig_campaign_kpis.to_html(full_html=False, include_plotlyjs='cdn')


# --- HTML Dashboard Structure ---
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CX Analytics Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; color: #333; }}
        .dashboard-container {{ padding: 20px; }}
        .card {{
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            margin-bottom: 20px;
            overflow: hidden; /* Important for Plotly charts */
            background-color: #fff;
        }}
        .card-header {{
            background-color: #007bff;
            color: white;
            padding: 10px 15px;
            font-size: 1.1em;
            font-weight: 600;
            border-bottom: 1px solid #ddd;
        }}
        .card-body {{ padding: 15px; }}
        .plotly-graph-div {{ height: 100% !important; width: 100% !important; }} /* Ensure plots fill card */
        .table-container {{ max-height: 400px; overflow-y: auto; }}
        /* Specific styling for sparklines */
        .sparkline-title {{
            font-size: 0.9em;
            font-weight: bold;
            text-align: center;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container-fluid dashboard-container">
        <h1 class="text-center mb-4 text-primary">Customer Experience Analytics Dashboard</h1>

        <div class="row">
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">Real-Time Sentiment Dashboard</div>
                    <div class="card-body">
                        {plot_html_sentiment_gauge}
                        {plot_html_sentiment_trend}
                    </div>
                </div>
            </div>

            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">Call Center Agent Performance</div>
                    <div class="card-body">
                        <div class="table-container">
                            {agent_table_html}
                        </div>
                    </div>
                </div>
            </div>

            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">CX Health Snapshot</div>
                    <div class="card-body">
                        {plot_html_cx_radar}
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">Categorization of Issues</div>
                    <div class="card-body">
                        {plot_html_issue_treemap}
                    </div>
                </div>
            </div>

            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">Call Queue & SLA Metrics</div>
                    <div class="card-body">
                        {plot_html_queue_sla}
                    </div>
                </div>
            </div>

            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">Threshold-Based KPI Alerts</div>
                    <div class="card-body">
                        {plot_html_kpi_sparklines}
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">Real-Time NPS Tracking</div>
                    <div class="card-body">
                        {plot_html_nps_tracking}
                    </div>
                </div>
            </div>

            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">Staff/Departmental Feedback</div>
                    <div class="card-body">
                        {plot_html_staff_feedback}
                    </div>
                </div>
            </div>

            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">Campaign Performance KPIs</div>
                    <div class="card-body">
                        {plot_html_campaign_kpis}
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

# Save the HTML to a file
dashboard_filename = "cx_dashboard.html"
with open(dashboard_filename, "w") as f:
    f.write(html_content)

print(f"\nDashboard HTML saved to {dashboard_filename}. Open this file in your browser to view the dashboard.")
print("Remember to have 'synthetic_transaction_data.csv', 'sentiment.csv', 'journey_entry.csv', and 'after_sales.csv' in the same directory as this script.")