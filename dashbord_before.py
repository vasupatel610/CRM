import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import base64
from io import BytesIO

# --- 1. Load Data ---
try:
    customer_churn_df = pd.read_csv('D:\Mockup\customer_churn_predictions.csv')
    journey_entry_df = pd.read_csv('journey_entry.csv')
    sentiment_df = pd.read_csv('sentiment.csv')
except FileNotFoundError as e:
    print(f"Error loading CSV file: {e}. Make sure the CSV files are in the same directory as the script.")
    exit()

# --- 2. Data Preprocessing (Common for multiple plots) ---
journey_entry_df['stage_date'] = pd.to_datetime(journey_entry_df['stage_date'])
journey_entry_df['hour'] = journey_entry_df['stage_date'].dt.hour
journey_entry_df['day_of_week'] = journey_entry_df['stage_date'].dt.day_name()

# Define time groups for heatmap
def get_time_group(hour):
    if 6 <= hour < 9:
        return 'Early Morning (6-9am)'
    elif 9 <= hour < 16:
        return 'Business Hours (9am-4pm)'
    elif 16 <= hour < 19:
        return 'Evening (4-7pm)'
    else:
        return 'Night (7pm-5am)'

journey_entry_df['time_group'] = journey_entry_df['hour'].apply(get_time_group)


# Function to convert matplotlib figure to base64 image
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Function to convert plotly figure to base64 image (PNG for embedding)
def plotly_fig_to_base64(fig):
    return base64.b64encode(fig.to_image(format="png")).decode('utf-8')


# --- 3. Plotting Functions for each dashboard cell ---

# 3.1 Campaign dashboard - Funnel charts
def plot_funnel_chart(df):
    funnel_stages = ['Sent', 'Viewed', 'Clicked', 'AddedToCart', 'Purchased-Loyalty']
    # Map journey stages to funnel stages for aggregation
    stage_mapping = {
        'sent': 'sent',  
        'viewed': 'viewed',
        'clicked': 'clicked',
        'addedtpcart': 'addedtpcart',
        'purchased': 'purchased',
    }
    df['mapped_stage'] = df['stage'].map(stage_mapping)

    # Count occurrences for each stage based on the funnel order
    funnel_counts = {stage: 0 for stage in funnel_stages}

    # This is a simplified funnel based on individual customer journey entries.
    # A more robust funnel would track a single customer's progression through stages.
    # For this example, we'll count unique customers at each 'mapped_stage'.
    for stage in funnel_stages:
        if stage == 'Sent': # Assuming all campaign_open 'Yes' implies sent/viewed
            funnel_counts[stage] = df[df['campaign_open'] == 'Yes']['customer_id'].nunique()
        elif stage == 'Viewed': # For simplicity, using 'Sent' count as 'Viewed'
             funnel_counts[stage] = df[df['campaign_open'] == 'Yes']['customer_id'].nunique()
        elif stage == 'Clicked':
            funnel_counts[stage] = df[df['campaign_click'] == 'Yes']['customer_id'].nunique()
        elif stage == 'AddedToCart':
            funnel_counts[stage] = df[df['product_in_cart'] == 'Yes']['customer_id'].nunique()
        elif stage == 'Purchased-Loyalty':
            funnel_counts[stage] = df[df['conversion_flag'] == 'Yes']['customer_id'].nunique()


    funnel_data = pd.DataFrame(list(funnel_counts.items()), columns=['Stage', 'Customers'])
    # Ensure stages are in order
    funnel_data['Stage'] = pd.Categorical(funnel_data['Stage'], categories=funnel_stages, ordered=True)
    funnel_data = funnel_data.sort_values('Stage')

    fig = px.funnel(funnel_data, x='Customers', y='Stage', title='Customer Journey Funnel')
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return plotly_fig_to_base64(fig)


# 3.2 Best time vs channel Heatmap
def plot_time_channel_heatmap(df):
    heatmap_data = df.groupby(['time_group', 'social_media_platform']).size().unstack(fill_value=0)
    # Order time groups
    time_order = ['Early Morning (6-9am)', 'Business Hours (9am-4pm)', 'Evening (4-7pm)', 'Night (7pm-5am)']
    heatmap_data = heatmap_data.reindex(time_order)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_title('Interactions by Time Group and Channel')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Time Group')
    return fig_to_base64(fig)

# 3.3 # new customers / re-engaged customers / existing customers (pie chart)
def plot_customer_segments_pie(customer_churn_df, journey_entry_df):
    # This requires a more complex definition, making some assumptions:
    # New: Customers with very few transactions (e.g., < 3) and recent activity (e.g., last 30 days)
    # Re-engaged: Customers with a gap in activity but recently active again.
    # Existing: Regular, active customers.

    # Simplified approach for this dataset:
    # Let's define based on 'days_since_last_transaction' and 'total_transactions'
    # New: days_since_last_transaction < 30 AND total_transactions < 5
    # Existing: days_since_last_transaction < 90 AND total_transactions >= 5
    # Re-engaged: days_since_last_transaction between 90 and 180 AND total_transactions >= 1

    df_merged = customer_churn_df.merge(journey_entry_df[['customer_id', 'stage_date']], on='customer_id', how='left')
    df_merged['last_activity_date'] = df_merged.groupby('customer_id')['stage_date'].transform('max')
    latest_date = df_merged['stage_date'].max()
    df_merged['days_since_last_activity'] = (latest_date - df_merged['last_activity_date']).dt.days.fillna(365) # Default for no activity

    customer_segments = {
        'New Customers': 0,
        'Re-engaged Customers': 0,
        'Existing Customers': 0
    }

    unique_customers = customer_churn_df['customer_id'].unique()

    for cust_id in unique_customers:
        cust_data = customer_churn_df[customer_churn_df['customer_id'] == cust_id].iloc[0]
        days_since_last = cust_data['days_since_last_transaction'] # Use this directly from churn data
        total_trans = cust_data['total_transactions']

        if days_since_last < 30 and total_trans < 5:
            customer_segments['New Customers'] += 1
        elif 30 <= days_since_last < 90 and total_trans >= 1: # Re-engaged might also be based on past churn/inactivity
            customer_segments['Re-engaged Customers'] += 1
        elif days_since_last < 90 and total_trans >= 5: # More active, established
            customer_segments['Existing Customers'] += 1
        else: # Catch-all for other cases not fitting the above
            customer_segments['Existing Customers'] += 1 # Default to existing if not new/re-engaged

    segments_df = pd.DataFrame(list(customer_segments.items()), columns=['Segment', 'Count'])

    fig = px.pie(segments_df, values='Count', names='Segment', title='Customer Segments Distribution')
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return plotly_fig_to_base64(fig)


def plot_localized_offers_performance(df):
    # Filter for entries where an offer was actually applied
    offers_df = df[df['offer_applied'] != 'No Offer'].copy()

    if offers_df.empty:
        # Create a dummy plot or return a message if no offer data is available
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for offers applied. Please ensure 'offer_applied' column has values other than 'No Offer'.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(title_text='Localized Offers Performance (No Data)', height=400, width=600)
        return plotly_fig_to_base64(fig)

    # Aggregate data by branch_id, latitude, and longitude
    # Calculate total offers applied and offers leading to conversion
    branch_offers_performance = offers_df.groupby(['branch_id', 'latitude', 'longitude']).agg(
        total_offers_applied=('offer_applied', 'count'),
        conversions_with_offer=('conversion_flag', lambda x: (x == 'Yes').sum())
    ).reset_index()

    # Calculate conversion rate for offers
    branch_offers_performance['conversion_rate_with_offer'] = branch_offers_performance.apply(
        lambda row: (row['conversions_with_offer'] / row['total_offers_applied']) if row['total_offers_applied'] > 0 else 0,
        axis=1
    )

    # Create an interactive map plot
    fig = px.scatter_mapbox(
        branch_offers_performance,
        lat="latitude",
        lon="longitude",
        size="total_offers_applied", # Size of marker by number of offers applied
        color="conversion_rate_with_offer", # Color by conversion rate with offer
        color_continuous_scale=px.colors.sequential.Viridis, # Choose a color scale
        size_max=30, # Max size of markers
        zoom=5, # Initial zoom level for Kenya
        center={"lat": -1.286389, "lon": 36.817223}, # Center map around Nairobi, Kenya
        mapbox_style="open-street-map", # Use OpenStreetMap style
        hover_name="branch_id", # Show branch ID on hover
        hover_data={
            "latitude": False, # Hide lat/lon in hover info
            "longitude": False,
            "total_offers_applied": True,
            "conversions_with_offer": True,
            "conversion_rate_with_offer": ":.2%" # Format as percentage
        },
        title='Localized Offers Performance by Branch Location'
    )

    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
    return plotly_fig_to_base64(fig)


# 3.5 Churn prediction vs LTV (Scatter plot)
def plot_churn_ltv_scatter(df):
    fig = px.scatter(df, x='customer_lifetime_value', y='churn_probability',
                     color='risk_level', hover_name='customer_id',
                     title='Churn Probability vs. Customer Lifetime Value',
                     labels={'customer_lifetime_value': 'Customer Lifetime Value',
                             'churn_probability': 'Churn Probability'})
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return plotly_fig_to_base64(fig)


# 3.6 At risk customers histogram (age bin)
def plot_at_risk_customers_histogram(df):
    at_risk_df = df[df['risk_level'].isin(['High Risk', 'Medium Risk'])] # Or based on churn_probability threshold
    if at_risk_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No 'High Risk' or 'Medium Risk' customers found.",
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('At Risk Customers Age Distribution')
        ax.axis('off')
        return fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(at_risk_df['customer_age'], bins=10, kde=True, ax=ax, color='salmon')
    ax.set_title('Age Distribution of At-Risk Customers')
    ax.set_xlabel('Customer Age')
    ax.set_ylabel('Number of Customers')
    return fig_to_base64(fig)


# 3.7 Social media trending (keywords / hashtags as word cloud)
def plot_social_media_wordcloud(df):
    # Combine hashtags from both journey_entry and sentiment dataframes
    all_hashtags = []
    if 'hashtags' in df.columns:
        all_hashtags.extend(df['hashtags'].dropna().tolist())
    if 'hashtags' in journey_entry_df.columns: # Assuming journey_entry_df is global or passed
        all_hashtags.extend(journey_entry_df['hashtags'].dropna().tolist())

    # Split hashtags if they are comma-separated or space-separated
    processed_hashtags = []
    for tags in all_hashtags:
        # Assuming hashtags can be space or comma separated
        processed_hashtags.extend([tag.strip().replace('#', '') for tag in tags.replace(',', ' ').split()])

    if not processed_hashtags:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No hashtags found for word cloud.",
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Social Media Trending (Word Cloud)')
        ax.axis('off')
        return fig_to_base64(fig)

    text = " ".join(processed_hashtags)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Social Media Trending (Hashtag Word Cloud)')
    return fig_to_base64(fig)


# 3.8 Trend view. No. of campaigns vs Sent (Line + bar. Combo chart over T campaigns- bar. Sent-line)
def plot_campaign_trends(df):
    # Group by campaign name to count unique campaigns and sent status
    campaign_summary = df.groupby('campaign_name').agg(
        total_sent=('campaign_open', lambda x: (x == 'Yes').sum()),
        total_campaigns=('campaign_name', 'size') # Count of entries per campaign
    ).reset_index()

    # Sort by total_sent for better visualization if many campaigns
    campaign_summary = campaign_summary.sort_values(by='total_sent', ascending=False)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart for total campaigns (count of records for each campaign)
    sns.barplot(x='campaign_name', y='total_campaigns', data=campaign_summary, ax=ax1, color='skyblue', label='Total Entries')
    ax1.set_xlabel('Campaign Name')
    ax1.set_ylabel('Total Campaign Entries (Count)', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    # Create a second y-axis for 'Total Sent' (line plot)
    ax2 = ax1.twinx()
    sns.lineplot(x='campaign_name', y='total_sent', data=campaign_summary, ax=ax2, color='red', marker='o', label='Total Sent (Opened)')
    ax2.set_ylabel('Total Sent (Opened)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    fig.suptitle('Campaign Entries vs. Sent/Opened Status')
    fig.legend(loc='upper right', bbox_to_anchor=(1,1))
    plt.tight_layout()
    return fig_to_base64(fig)


# 3.9 Emerging customer segments
def plot_emerging_customer_segments(df_churn, df_journey):
    # This is highly conceptual and depends on detailed definitions and data.
    # I'll create a simplified version based on provided descriptions:
    # 1. Segment similar to M-30to46-R1 buying pattern: using customer_age 30-46
    # 2. Segment of high-return-rate customers emerging after a new product launch: Requires product launch date.
    #    Let's use a proxy: high `return_rate` and `days_since_last_transaction` is low.
    # 3. Low-spending customers starting to show high engagement clicking on premium product ads:
    #    Low `total_spent` but `campaign_click` == 'Yes' for premium products (needs product data).

    emerging_segments_counts = {
        "Age 30-46 & Active": 0,
        "High Return Rate (Recent Activity)": 0,
        "Low Spending, High Engagement": 0
    }

    # Segment 1: Age 30-46 & Active
    segment1_customers = df_churn[(df_churn['customer_age'] >= 30) & (df_churn['customer_age'] <= 46)]
    emerging_segments_counts["Age 30-46 & Active"] = segment1_customers['customer_id'].nunique()

    # Segment 2: High Return Rate (Recent Activity) - simplified
    # Assuming 'high return rate' is > 0.1 and 'recent activity' is days_since_last_transaction < 60
    segment2_customers = df_churn[(df_churn['return_rate'] > 0.1) & (df_churn['days_since_last_transaction'] < 60)]
    emerging_segments_counts["High Return Rate (Recent Activity)"] = segment2_customers['customer_id'].nunique()

    # Segment 3: Low Spending, High Engagement - simplified
    # Low spending: total_spent < 1000000 (arbitrary threshold)
    # High engagement: At least one campaign click
    low_spending_customers = df_churn[df_churn['total_spent'] < 1000000]['customer_id'].tolist()
    engaged_customers = df_journey[df_journey['campaign_click'] == 'Yes']['customer_id'].tolist()
    segment3_customers = set(low_spending_customers).intersection(set(engaged_customers))
    emerging_segments_counts["Low Spending, High Engagement"] = len(segment3_customers)

    segments_df = pd.DataFrame(list(emerging_segments_counts.items()), columns=['Segment', 'Count'])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Segment', y='Count', data=segments_df, ax=ax, palette='coolwarm')
    ax.set_title('Emerging Customer Segments')
    ax.set_ylabel('Number of Customers')
    ax.set_xlabel('Segment')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig_to_base64(fig)


# --- 4. Generate HTML Dashboard ---
def generate_dashboard_html(plots):
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Marketing Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f7f6; }}
            .dashboard-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
                max-width: 1200px;
                margin: 0 auto;
            }}
            .card {{
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                padding: 20px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                text-align: center;
            }}
            .card h3 {{
                color: #333;
                margin-top: 0;
                margin-bottom: 15px;
            }}
            .card img {{
                max-width: 100%;
                height: auto;
                border-radius: 4px;
            }}
            .plotly-chart {{
                width: 100%;
                height: auto;
            }}
        </style>
    </head>
    <body>
        <div class="dashboard-grid">
            <div class="card">
                <h3>Campaign Dashboard - Funnel Charts</h3>
                <img src="data:image/png;base64,{plot1_base64}" alt="Funnel Chart" class="plotly-chart">
            </div>
            <div class="card">
                <h3>Best Time vs Channel Heatmap</h3>
                <img src="data:image/png;base64,{plot2_base64}" alt="Heatmap">
            </div>
            <div class="card">
                <h3>New / Re-engaged / Existing Customers</h3>
                <img src="data:image/png;base64,{plot3_base64}" alt="Customer Segments Pie Chart" class="plotly-chart">
            </div>
            <div class="card">
                <h3>Localized Offers at Branch Performance (Simulated)</h3>
                <img src="data:image/png;base64,{plot4_base64}" alt="Localized Offers Performance">
            </div>
            <div class="card">
                <h3>Churn Prediction vs LTV</h3>
                <img src="data:image/png;base64,{plot5_base64}" alt="Churn LTV Scatter Plot" class="plotly-chart">
            </div>
            <div class="card">
                <h3>At Risk Customers Histogram (Age Bin)</h3>
                <img src="data:image/png;base64,{plot6_base64}" alt="At Risk Customers Histogram">
            </div>
            <div class="card">
                <h3>Social Media Trending (Word Cloud)</h3>
                <img src="data:image/png;base64,{plot7_base64}" alt="Social Media Word Cloud">
            </div>
            <div class="card">
                <h3>Trend View: Campaigns vs Sent</h3>
                <img src="data:image/png;base64,{plot8_base64}" alt="Campaign Trends">
            </div>
            <div class="card">
                <h3>Emerging Customer Segments</h3>
                <img src="data:image/png;base64,{plot9_base64}" alt="Emerging Segments">
            </div>
        </div>
    </body>
    </html>
    """
    return html_template.format(**plots)

# --- 5. Main Execution ---
if __name__ == "__main__":
    # Generate all plots
    plot1 = plot_funnel_chart(journey_entry_df.copy()) # Copy to avoid modifying original df
    plot2 = plot_time_channel_heatmap(journey_entry_df.copy())
    plot3 = plot_customer_segments_pie(customer_churn_df.copy(), journey_entry_df.copy())
    plot4 = plot_localized_offers_performance(journey_entry_df.copy())
    plot5 = plot_churn_ltv_scatter(customer_churn_df.copy())
    plot6 = plot_at_risk_customers_histogram(customer_churn_df.copy())
    plot7 = plot_social_media_wordcloud(sentiment_df.copy())
    plot8 = plot_campaign_trends(journey_entry_df.copy())
    plot9 = plot_emerging_customer_segments(customer_churn_df.copy(), journey_entry_df.copy())

    plots_dict = {
        'plot1_base64': plot1,
        'plot2_base64': plot2,
        'plot3_base64': plot3,
        'plot4_base64': plot4,
        'plot5_base64': plot5,
        'plot6_base64': plot6,
        'plot7_base64': plot7,
        'plot8_base64': plot8,
        'plot9_base64': plot9,
    }

    html_output = generate_dashboard_html(plots_dict)

    # Save to an HTML file
    with open('marketing_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_output)

    print("Dashboard 'marketing_dashboard.html' generated successfully!")

# import pandas as pd
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots
# import numpy as np
# from datetime import datetime, timedelta
# import re
# from collections import Counter
# import plotly.offline as pyo
# import json


# class MarketingDashboard:
#     def __init__(self, journey_file, sentiment_file, churn_file):
#         """Initialize the dashboard with data files"""
#         self.journey_df = pd.read_csv(journey_file)
#         self.sentiment_df = pd.read_csv(sentiment_file)
#         self.churn_df = pd.read_csv(churn_file)
        
#         # Clean and prepare data
#         self._prepare_data()
    
#     def _prepare_data(self):
#         """Clean and prepare data for visualization"""
#         # Convert dates
#         self.journey_df['stage_date'] = pd.to_datetime(self.journey_df['stage_date'])
#         self.sentiment_df['date'] = pd.to_datetime(self.sentiment_df['date'])
        
#         # Extract time components
#         self.journey_df['hour'] = self.journey_df['stage_date'].dt.hour
#         self.journey_df['day_of_week'] = self.journey_df['stage_date'].dt.day_name()
#         self.journey_df['date'] = self.journey_df['stage_date'].dt.date
        
#         # Process hashtags - handle NaN values
#         self.journey_df['hashtags'] = self.journey_df['hashtags'].fillna('')
#         self.sentiment_df['hashtags'] = self.sentiment_df['hashtags'].fillna('')
        
#         # Clean string columns
#         for col in ['campaign_open', 'campaign_click', 'conversion_flag', 'product_in_cart']:
#             if col in self.journey_df.columns:
#                 self.journey_df[col] = self.journey_df[col].fillna('No')

#         # --- REVISED FIX: Ensure customer_age in churn_df is numeric (it already exists in this CSV) ---
#         # The 'customer_age' column is already present in 'customer_churn_predictions.csv'.
#         # No merge is needed for this column from journey_df into churn_df.
#         # Just ensure it's in a numeric format to avoid errors in plotting functions.
#         if 'customer_age' in self.churn_df.columns:
#             self.churn_df['customer_age'] = pd.to_numeric(self.churn_df['customer_age'], errors='coerce')
#         # --- END REVISED FIX ---
    
#     def create_funnel_chart(self):
#         """Create campaign funnel chart using actual journey stages"""
#         # Calculate funnel metrics from actual data
#         stage_metrics = self.journey_df.groupby('stage').agg({
#             'customer_id': 'nunique',
#             'journey_id': 'count'
#         }).reset_index()
        
#         # Order stages by typical customer journey flow
#         stage_order = ['Awareness', 'Consideration', 'Lead', 'Purchase', 'Service', 'Loyalty', 'Advocacy']
        
#         # Filter to only stages that exist in data and order them
#         existing_stages = stage_metrics['stage'].unique()
#         ordered_stages = [stage for stage in stage_order if stage in existing_stages]
        
#         # Add any remaining stages not in our predefined order
#         remaining_stages = [stage for stage in existing_stages if stage not in ordered_stages]
#         final_order = ordered_stages + remaining_stages
        
#         # Reorder data
#         stage_metrics['stage'] = pd.Categorical(stage_metrics['stage'], categories=final_order, ordered=True)
#         stage_metrics = stage_metrics.sort_values('stage').reset_index(drop=True)
        
#         fig = go.Figure(go.Funnel(
#             y=stage_metrics['stage'],
#             x=stage_metrics['customer_id'],
#             textinfo="value+percent initial",
#             marker=dict(color=["#5DADE2", "#48C9B0", "#F7DC6F", "#F8C471", "#EC7063", "#BB8FCE", "#85C1E9"][:len(stage_metrics)]),
#             textfont=dict(size=12, color="white")
#         ))
        
#         fig.update_layout(
#             title="Campaign Funnel - Customer Journey Stages",
#             title_font_size=14,
#             height=350,
#             margin=dict(l=10, r=10, t=40, b=10)
#         )
        
#         return fig
    
#     def create_heatmap(self):
#         """Create best time vs stage heatmap using journey_entry.csv data"""
#         # Define time periods based on hour with your specified groupings
#         def get_time_period(hour):
#             if 6 <= hour < 9:
#                 return "Early Morning (6-9)"
#             elif 9 <= hour < 16:
#                 return "Business Hours (9-4pm)"
#             elif 16 <= hour < 19:
#                 return "Evening (4-7pm)"
#             else:
#                 return "Night (7pm-5am)"
        
#         # Add time period column to journey data
#         self.journey_df['stage_date'] = pd.to_datetime(self.journey_df['stage_date'])
#         self.journey_df['time_period'] = self.journey_df['stage_date'].dt.hour.apply(get_time_period)
        
#         # Calculate engagement metrics by stage and time_period
#         heatmap_data = self.journey_df.groupby(['stage', 'time_period']).agg({
#             'journey_id': 'count',
#             'campaign_open': lambda x: (x == 'Yes').sum(),
#             'campaign_click': lambda x: (x == 'Yes').sum(),
#             'conversion_flag': lambda x: (x == 'Yes').sum()
#         }).reset_index()
        
#         # Calculate engagement rate (opens + clicks + conversions) / total interactions
#         heatmap_data['engagement_rate'] = (
#             heatmap_data['campaign_open'] + 
#             heatmap_data['campaign_click'] + 
#             heatmap_data['conversion_flag']
#         ) / heatmap_data['journey_id']
        
#         # Pivot for heatmap
#         heatmap_pivot = heatmap_data.pivot(
#             index='stage', 
#             columns='time_period', 
#             values='engagement_rate'
#         ).fillna(0)
        
#         # Ensure time periods are in the correct order
#         time_order = ['Early Morning (6-9)', 'Business Hours (9-4pm)', 'Evening (4-7pm)', 'Night (7pm-5am)']
#         available_times = [time for time in time_order if time in heatmap_pivot.columns]
#         heatmap_pivot = heatmap_pivot[available_times]
        
#         fig = px.imshow(
#             heatmap_pivot.values,
#             x=heatmap_pivot.columns,
#             y=heatmap_pivot.index,
#             color_continuous_scale='Blues',
#             aspect='auto',
#             labels=dict(color='Engagement Rate', x='Time Period', y='Journey Stage')
#         )
        
#         fig.update_layout(
#             title='Best Time vs Journey Stage Heatmap',
#             title_font_size=14,
#             height=350,
#             margin=dict(l=10, r=10, t=40, b=10),
#             xaxis_title='Time Period',
#             yaxis_title='Journey Stage'
#         )
        
#         return fig

    
#     def create_customer_pie_chart(self):
#         """Create customer segmentation based on actual journey data"""
#         # Analyze customer behavior patterns
#         customer_analysis = self.journey_df.groupby('customer_id').agg({
#             'stage': ['count', 'nunique'],
#             'conversion_flag': lambda x: (x == 'Yes').sum(),
#             'campaign_click': lambda x: (x == 'Yes').sum(),
#             'stage_date': ['min', 'max']
#         }).reset_index()
        
#         # Flatten column names
#         customer_analysis.columns = ['customer_id', 'total_interactions', 'unique_stages', 'conversions', 'clicks', 'first_interaction', 'last_interaction']
        
#         # Calculate engagement duration
#         customer_analysis['engagement_days'] = (customer_analysis['last_interaction'] - customer_analysis['first_interaction']).dt.days
        
#         # Categorize customers based on actual behavior
#         def categorize_customer(row):
#             if row['unique_stages'] <= 2 and row['total_interactions'] <= 3:
#                 return "New Customers"
#             elif row['conversions'] > 0 or row['clicks'] > 2:
#                 return "Re-engaged Customers"
#             else:
#                 return "Existing Customers"
        
#         customer_analysis['category'] = customer_analysis.apply(categorize_customer, axis=1)
#         category_counts = customer_analysis['category'].value_counts()
        
#         fig = go.Figure(data=[go.Pie(
#             labels=category_counts.index,
#             values=category_counts.values,
#             hole=0.4,
#             marker_colors=['#3498DB', '#E74C3C', '#2ECC71'],
#             textinfo='label+percent+value'
#         )])
        
#         fig.update_layout(
#             title="Customer Segmentation (Based on Journey Behavior)",
#             title_font_size=14,
#             height=350,
#             margin=dict(l=10, r=10, t=40, b=10)
#         )
        
#         return fig
    
#     def create_campaign_performance(self):
#         """Create campaign performance chart using actual data"""
#         # Calculate campaign performance metrics
#         campaign_performance = self.journey_df.groupby('campaign_name').agg({
#             'customer_id': 'nunique',
#             'campaign_open': lambda x: (x == 'Yes').sum(),
#             'campaign_click': lambda x: (x == 'Yes').sum(),
#             'conversion_flag': lambda x: (x == 'Yes').sum(),
#             'journey_id': 'count'
#         }).reset_index()
        
#         # Calculate rates
#         campaign_performance['open_rate'] = campaign_performance['campaign_open'] / campaign_performance['journey_id']
#         campaign_performance['click_rate'] = campaign_performance['campaign_click'] / campaign_performance['journey_id']
#         campaign_performance['conversion_rate'] = campaign_performance['conversion_flag'] / campaign_performance['journey_id']
        
#         # Sort by total interactions and take top 10
#         campaign_performance = campaign_performance.sort_values('journey_id', ascending=True).tail(10)
        
#         fig = go.Figure()
        
#         # Add bars for different metrics
#         fig.add_trace(go.Bar(
#             name='Open Rate',
#             x=campaign_performance['open_rate'],
#             y=campaign_performance['campaign_name'],
#             orientation='h',
#             marker_color='#3498DB',
#             opacity=0.8
#         ))
        
#         fig.add_trace(go.Bar(
#             name='Click Rate',
#             x=campaign_performance['click_rate'],
#             y=campaign_performance['campaign_name'],
#             orientation='h',
#             marker_color='#E74C3C',
#             opacity=0.8
#         ))
        
#         fig.add_trace(go.Bar(
#             name='Conversion Rate',
#             x=campaign_performance['conversion_rate'],
#             y=campaign_performance['campaign_name'],
#             orientation='h',
#             marker_color='#2ECC71',
#             opacity=0.8
#         ))
        
#         fig.update_layout(
#             title="Campaign Performance (Open/Click/Conversion Rates)",
#             title_font_size=14,
#             height=350,
#             margin=dict(l=10, r=10, t=40, b=10),
#             barmode='group',
#             xaxis_title="Rate",
#             yaxis_title="Campaign"
#         )
        
#         return fig
    
#     def create_churn_ltv_scatter(self):
#         """Create churn prediction vs LTV scatter plot using actual churn data"""
#         # Ensure churn_probability and customer_lifetime_value are numeric
#         self.churn_df['churn_probability'] = pd.to_numeric(self.churn_df['churn_probability'], errors='coerce')
        
#         # Calculate LTV from actual data - using customer_lifetime_value column directly
#         if 'customer_lifetime_value' in self.churn_df.columns:
#             self.churn_df['ltv'] = pd.to_numeric(self.churn_df['customer_lifetime_value'], errors='coerce')
#         else:
#             # Fallback calculation if the column doesn't exist or is not numeric
#             self.churn_df['total_spent'] = pd.to_numeric(self.churn_df['total_spent'], errors='coerce')
#             self.churn_df['total_transactions'] = pd.to_numeric(self.churn_df['total_transactions'], errors='coerce')
#             self.churn_df['ltv'] = self.churn_df['total_spent'] / self.churn_df['total_transactions']
        
#         # Handle any infinite or NaN values that might arise from division or coercion
#         self.churn_df['ltv'] = self.churn_df['ltv'].replace([np.inf, -np.inf], np.nan)
        
#         # Drop rows with NaN in critical columns for plotting
#         # customer_age is now confirmed to be in self.churn_df and numeric
#         churn_data_clean = self.churn_df.dropna(subset=['ltv', 'churn_probability', 'risk_level', 'total_transactions', 'customer_id', 'total_spent', 'days_since_last_transaction', 'customer_age']).copy()
        
#         # Create the scatter plot
#         fig = px.scatter(
#             churn_data_clean,
#             x='churn_probability',
#             y='ltv',
#             color='risk_level',
#             size='total_transactions',
#             # Pass all relevant data for hover as customdata
#             hover_data=['customer_id', 'total_spent', 'days_since_last_transaction', 'customer_age', 'risk_level', 'total_transactions'],
#             color_discrete_map={
#                 'High Risk': '#E74C3C', 
#                 'Medium Risk': '#F39C12', 
#                 'Low Risk': '#27AE60'
#             },
#             size_max=20,
#             opacity=0.7
#         )
        
#         # Update layout for better visualization
#         fig.update_layout(
#             title="Churn Probability vs Customer Lifetime Value",
#             title_font_size=14,
#             xaxis_title="Churn Probability",
#             yaxis_title="Customer Lifetime Value ($)",
#             height=350,
#             margin=dict(l=10, r=10, t=40, b=10),
#             showlegend=True
#         )
        
#         # Format the hover template for better readability
#         # Access customdata elements by their index as defined in hover_data list
#         fig.update_traces(
#             hovertemplate="<b>Customer ID: %{customdata[0]}</b><br>" +
#                           "Churn Probability: %{x:.2f}<br>" +
#                           "LTV: $%{y:,.2f}<br>" +
#                           "Risk Level: %{customdata[4]}<br>" +  # Access risk_level from customdata
#                           "Total Transactions: %{customdata[5]}<br>" + # Access total_transactions from customdata
#                           "Total Spent: $%{customdata[1]:,.2f}<br>" +
#                           "Days Since Last Transaction: %{customdata[2]}<br>" +
#                           "Customer Age: %{customdata[3]}<br>" +
#                           "<extra></extra>" # Removes the default trace name from hover
#         )
        
#         return fig

    
#     def create_risk_histogram(self):
#         """Create at-risk customers histogram using actual age data"""
#         # Ensure customer_age is numeric (already handled in _prepare_data)
#         fig = px.histogram(
#             self.churn_df,
#             x='customer_age',
#             color='risk_level',
#             nbins=15,
#             color_discrete_map={'High Risk': '#E74C3C', 'Medium Risk': '#F39C12', 'Low Risk': '#27AE60'},
#             title="Customer Age Distribution by Risk Level"
#         )
        
#         fig.update_layout(
#             title="At Risk Customers by Age Distribution",
#             title_font_size=14,
#             xaxis_title="Customer Age",
#             yaxis_title="Number of Customers",
#             height=350,
#             margin=dict(l=10, r=10, t=40, b=10)
#         )
        
#         return fig
    
#     def create_social_keywords_chart(self):
#         """Create social media trending keywords from actual hashtag data"""
#         # Extract hashtags from both datasets
#         all_hashtags = []
        
#         # Process journey hashtags
#         for hashtags in self.journey_df['hashtags']:
#             if hashtags and isinstance(hashtags, str):
#                 tags = re.findall(r'#\w+', hashtags.lower())
#                 all_hashtags.extend([tag.replace('#', '') for tag in tags])
        
#         # Process sentiment hashtags
#         for hashtags in self.sentiment_df['hashtags']:
#             if hashtags and isinstance(hashtags, str):
#                 tags = re.findall(r'#\w+', hashtags.lower())
#                 all_hashtags.extend([tag.replace('#', '') for tag in tags])
        
#         # Count frequency and get top hashtags
#         if all_hashtags:
#             hashtag_counts = Counter(all_hashtags)
#             top_hashtags = dict(hashtag_counts.most_common(12))
            
#             fig = go.Figure(go.Bar(
#                 x=list(top_hashtags.values()),
#                 y=list(top_hashtags.keys()),
#                 orientation='h',
#                 marker_color='#3498DB',
#                 text=list(top_hashtags.values()),
#                 textposition='auto'
#             ))
            
#             fig.update_layout(
#                 title="Social Media Trending Keywords (From Actual Hashtags)",
#                 title_font_size=14,
#                 xaxis_title="Frequency",
#                 yaxis_title="Hashtags",
#                 height=350,
#                 margin=dict(l=10, r=10, t=40, b=10)
#             )
#         else:
#             # Fallback if no hashtags found
#             fig = go.Figure()
#             fig.add_annotation(
#                 text="No hashtag data available",
#                 xref="paper", yref="paper",
#                 x=0.5, y=0.5, showarrow=False
#             )
#             fig.update_layout(
#                 title="Social Media Trending Keywords",
#                 title_font_size=14,
#                 height=350,
#                 margin=dict(l=10, r=10, t=40, b=10)
#             )
        
#         return fig
    
#     def create_campaign_trend(self):
#         """Create campaign trends over time using actual date data"""
#         # Group by date and count campaigns
#         daily_campaigns = self.journey_df.groupby(['date', 'campaign_name']).size().reset_index(name='interactions')
#         daily_totals = daily_campaigns.groupby('date')['interactions'].sum().reset_index()
        
#         # Get top 3 campaigns for individual tracking
#         top_campaigns = self.journey_df['campaign_name'].value_counts().head(3).index
        
#         fig = go.Figure()
        
#         # Add total campaigns line
#         fig.add_trace(go.Scatter(
#             x=daily_totals['date'], 
#             y=daily_totals['interactions'], 
#             mode='lines+markers', 
#             name='Total Daily Interactions',
#             line=dict(color='#2C3E50', width=3),
#             marker=dict(size=6)
#         ))
        
#         # Add individual campaign lines
#         colors = ['#E74C3C', '#3498DB', '#2ECC71']
#         for i, campaign in enumerate(top_campaigns):
#             campaign_data = daily_campaigns[daily_campaigns['campaign_name'] == campaign]
#             campaign_daily = campaign_data.groupby('date')['interactions'].sum().reset_index()
            
#             fig.add_trace(go.Scatter(
#                 x=campaign_daily['date'], 
#                 y=campaign_daily['interactions'], 
#                 mode='lines+markers',
#                 name=campaign,
#                 line=dict(color=colors[i], width=2),
#                 marker=dict(size=4)
#             ))
        
#         fig.update_layout(
#             title="Campaign Interaction Trends Over Time",
#             title_font_size=14,
#             height=350,
#             margin=dict(l=10, r=10, t=40, b=10),
#             xaxis_title="Date",
#             yaxis_title="Number of Interactions",
#             hovermode='x unified'
#         )
        
#         return fig
    
#     def create_sentiment_analysis(self):
#         """Create sentiment analysis from actual sentiment data"""
#         # Analyze sentiment distribution
#         sentiment_counts = self.sentiment_df['review_sentiment'].value_counts()
        
#         # Also analyze by platform if available
#         platform_sentiment = self.sentiment_df.groupby(['social_media_platform', 'review_sentiment']).size().unstack(fill_value=0)
        
#         fig = make_subplots(
#             rows=1, cols=2,
#             subplot_titles=('Overall Sentiment Distribution', 'Sentiment by Platform'),
#             specs=[[{"type": "pie"}, {"type": "bar"}]],
#             horizontal_spacing=0.1 # Adjust spacing between subplots
#         )
        
#         # Overall sentiment pie chart
#         fig.add_trace(
#             go.Pie(
#                 labels=sentiment_counts.index,
#                 values=sentiment_counts.values,
#                 hole=0.4,
#                 marker_colors=['#2ECC71', '#E74C3C', '#F39C12'], # Positive, Negative, Neutral
#                 name="Overall Sentiment",
#                 textinfo='label+percent'
#             ),
#             row=1, col=1
#         )
        
#         # Sentiment by platform bar chart
#         # Ensure consistent order of sentiments for coloring
#         sentiment_order = ['Positive', 'Negative', 'Neutral']
#         sentiment_colors = {'Positive': '#2ECC71', 'Negative': '#E74C3C', 'Neutral': '#F39C12'}

#         for sentiment in sentiment_order:
#             if sentiment in platform_sentiment.columns:
#                 fig.add_trace(
#                     go.Bar(
#                         name=sentiment,
#                         x=platform_sentiment.index,
#                         y=platform_sentiment[sentiment],
#                         marker_color=sentiment_colors.get(sentiment, '#95A5A6'),
#                         showlegend=True # Show legend for each sentiment in bar chart
#                     ),
#                     row=1, col=2
#                 )
        
#         fig.update_layout(
#             title="Customer Sentiment Analysis",
#             title_font_size=14,
#             height=350,
#             margin=dict(l=10, r=10, t=40, b=10),
#             barmode='stack', # Stack bars for sentiment by platform
#             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) # Position legend at top
#         )
        
#         return fig
        
#     def create_emerging_segments_analysis(self):
#         """Create emerging customer segments based on actual data analysis (Not included in 3x3 grid)"""
#         # This function is retained for completeness but its output is not used in the 3x3 dashboard HTML.
#         # It can be used if you decide to add an insights section outside the main grid.
#         customer_segments = self.journey_df.merge(
#             # Select relevant columns from churn_df, customer_age is already in churn_df
#             self.churn_df[['customer_id', 'risk_level', 'total_spent', 'customer_age']], 
#             on='customer_id', 
#             how='left'
#         )
        
#         customer_segments['age_group'] = pd.cut(
#             customer_segments['customer_age'], 
#             bins=[0, 25, 35, 50, 65, 100], 
#             labels=['18-25', '26-35', '36-50', '51-65', '65+']
#         )
        
#         segment_analysis = customer_segments.groupby(['age_group', 'risk_level'], observed=False).agg({
#             'customer_id': 'nunique',
#             'total_spent': 'mean',
#             'conversion_flag': lambda x: (x == 'Yes').sum()
#         }).reset_index()
        
#         segment_analysis = segment_analysis.dropna()
        
#         insights = []
#         for _, row in segment_analysis.iterrows():
#             if row['customer_id'] > 10:
#                 avg_spent = row['total_spent'] if pd.notna(row['total_spent']) else 0
#                 insights.append(
#                     f"<strong>{row['age_group']} - {row['risk_level']}:</strong><br>"
#                     f"<span style='color: #666;'>{int(row['customer_id'])} customers, "
#                     f"Avg. Spent: ${avg_spent:,.0f}, "
#                     f"Conversions: {int(row['conversion_flag'])}</span>"
#                 )
#         return insights[:6] 
    
#     def generate_dashboard(self, output_filename='marketing_dashboard.html'):
#         """Generate complete 3x3 dashboard using only actual CSV data based on the provided image."""
        
#         print("📊 Creating visualizations from actual data...")
        
#         # Create all 9 charts for the 3x3 grid
#         funnel_fig = self.create_funnel_chart()
#         heatmap_fig = self.create_heatmap()
#         pie_fig = self.create_customer_pie_chart()
#         campaign_fig = self.create_campaign_performance()
#         scatter_fig = self.create_churn_ltv_scatter()
#         histogram_fig = self.create_risk_histogram()
#         social_fig = self.create_social_keywords_chart()
#         trend_fig = self.create_campaign_trend()
#         sentiment_fig = self.create_sentiment_analysis() # The 9th chart
        
#         # Convert figures to JSON
#         charts_json = {
#             'funnel': funnel_fig.to_json(),
#             'heatmap': heatmap_fig.to_json(),
#             'pie': pie_fig.to_json(),
#             'campaign': campaign_fig.to_json(),
#             'scatter': scatter_fig.to_json(),
#             'histogram': histogram_fig.to_json(),
#             'social': social_fig.to_json(),
#             'trend': trend_fig.to_json(),
#             'sentiment': sentiment_fig.to_json() # Add sentiment chart
#         }
        
#         # Generate HTML structure
#         html_content = f"""
#         <!DOCTYPE html>
#         <html>
#         <head>
#             <title>Marketing Analytics Dashboard - Data-Driven Insights</title>
#             <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
#             <style>
#                 body {{
#                     font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#                     margin: 0;
#                     padding: 15px;
#                     background-color: #f5f6fa;
#                 }}
#                 .dashboard-container {{
#                     max-width: 1400px;
#                     margin: 0 auto;
#                     background: white;
#                     border-radius: 12px;
#                     box-shadow: 0 8px 25px rgba(0,0,0,0.1);
#                     overflow: hidden;
#                 }}
#                 .dashboard-header {{
#                     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#                     color: white;
#                     padding: 25px;
#                     text-align: center;
#                 }}
#                 .dashboard-header h1 {{
#                     margin: 0 0 10px 0;
#                     font-size: 28px;
#                     font-weight: 600;
#                 }}
#                 .dashboard-header p {{
#                     margin: 0;
#                     opacity: 0.9;
#                     font-size: 16px;
#                 }}
#                 .dashboard-grid {{
#                     display: grid;
#                     grid-template-columns: 1fr 1fr 1fr; /* 3 columns for 3x3 layout */
#                     gap: 15px;
#                     padding: 20px;
#                 }}
#                 .chart-container {{
#                     background: white;
#                     border-radius: 10px;
#                     box-shadow: 0 3px 10px rgba(0,0,0,0.08);
#                     padding: 15px;
#                     min-height: 350px; /* Ensure consistent height for grid cells */
#                     border: 1px solid #e1e8ed;
#                 }}
#                 .chart-container:hover {{
#                     transform: translateY(-2px);
#                     box-shadow: 0 6px 20px rgba(0,0,0,0.12);
#                     transition: all 0.3s ease;
#                 }}
#                 .data-badge {{
#                     background: #e8f5e8;
#                     color: #2d5a2d;
#                     padding: 4px 8px;
#                     border-radius: 4px;
#                     font-size: 12px;
#                     font-weight: bold;
#                 }}
#                 @media (max-width: 1200px) {{
#                     .dashboard-grid {{
#                         grid-template-columns: 1fr 1fr;
#                     }}
#                 }}
#                 @media (max-width: 768px) {{
#                     .dashboard-grid {{
#                         grid-template-columns: 1fr;
#                     }}
#                 }}
#             </style>
#         </head>
#         <body>
#             <div class="dashboard-container">
#                 <div class="dashboard-header">
#                     <h1>📊 Marketing Analytics Dashboard</h1>
#                     <p>Real insights from your actual customer journey, sentiment, and churn data</p>
#                     <div style="margin-top: 10px;">
#                         <span class="data-badge">✓ Real Data Only</span>
#                         <span class="data-badge">✓ {len(self.journey_df):,} Journey Records</span>
#                         <span class="data-badge">✓ {len(self.sentiment_df):,} Sentiment Records</span>
#                         <span class="data-badge">✓ {len(self.churn_df):,} Customer Profiles</span>
#                     </div>
#                 </div>
#                 <div class="dashboard-grid">
#                     <div class="chart-container">
#                         <div id="funnel-chart" style="height: 100%;"></div>
#                     </div>
#                     <div class="chart-container">
#                         <div id="heatmap-chart" style="height: 100%;"></div>
#                     </div>
#                     <div class="chart-container">
#                         <div id="pie-chart" style="height: 100%;"></div>
#                     </div>
#                     <div class="chart-container">
#                         <div id="campaign-chart" style="height: 100%;"></div>
#                     </div>
#                     <div class="chart-container">
#                         <div id="scatter-chart" style="height: 100%;"></div>
#                     </div>
#                     <div class="chart-container">
#                         <div id="histogram-chart" style="height: 100%;"></div>
#                     </div>
#                     <div class="chart-container">
#                         <div id="social-chart" style="height: 100%;"></div>
#                     </div>
#                     <div class="chart-container">
#                         <div id="trend-chart" style="height: 100%;"></div>
#                     </div>
#                     <div class="chart-container">
#                         <div id="sentiment-chart" style="height: 100%;"></div> {{/* The 9th chart container */}}
#                     </div>
#                 </div>
#             </div>
            
#             <script>
#                 // Chart configurations
#                 const config = {{
#                     responsive: true,
#                     displayModeBar: false,
#                     displaylogo: false
#                 }};
                
#                 // Render all charts
#                 Plotly.newPlot('funnel-chart', {charts_json['funnel']}.data, {charts_json['funnel']}.layout, config);
#                 Plotly.newPlot('heatmap-chart', {charts_json['heatmap']}.data, {charts_json['heatmap']}.layout, config);
#                 Plotly.newPlot('pie-chart', {charts_json['pie']}.data, {charts_json['pie']}.layout, config);
#                 Plotly.newPlot('campaign-chart', {charts_json['campaign']}.data, {charts_json['campaign']}.layout, config);
#                 Plotly.newPlot('scatter-chart', {charts_json['scatter']}.data, {charts_json['scatter']}.layout, config);
#                 Plotly.newPlot('histogram-chart', {charts_json['histogram']}.data, {charts_json['histogram']}.layout, config);
#                 Plotly.newPlot('social-chart', {charts_json['social']}.data, {charts_json['social']}.layout, config);
#                 Plotly.newPlot('trend-chart', {charts_json['trend']}.data, {charts_json['trend']}.layout, config);
#                 Plotly.newPlot('sentiment-chart', {charts_json['sentiment']}.data, {charts_json['sentiment']}.layout, config); // Render the 9th chart
                
#                 // Make charts responsive
#                 window.addEventListener('resize', function() {{
#                     Plotly.Plots.resize('funnel-chart');
#                     Plotly.Plots.resize('heatmap-chart');
#                     Plotly.Plots.resize('pie-chart');
#                     Plotly.Plots.resize('campaign-chart');
#                     Plotly.Plots.resize('scatter-chart');
#                     Plotly.Plots.resize('histogram-chart');
#                     Plotly.Plots.resize('social-chart');
#                     Plotly.Plots.resize('trend-chart');
#                     Plotly.Plots.resize('sentiment-chart'); // Make sentiment chart responsive
#                 }});
#             </script>
#         </body>
#         </html>
#         """
        
#         # Save to file
#         with open(output_filename, 'w', encoding='utf-8') as f:
#             f.write(html_content)
        
#         print(f"✅ Dashboard generated successfully: {output_filename}")
#         print(f"📂 File size: {len(html_content)/1024:.1f} KB")
#         print(f"🔗 Open the file in your web browser to view the dashboard")
        
#         return output_filename


# # Usage example
# if __name__ == "__main__":
#     try:
#         # Initialize dashboard
#         print("🚀 Initializing Marketing Dashboard...")
#         dashboard = MarketingDashboard(
#             journey_file='journey_entry.csv',
#             sentiment_file='sentiment.csv', 
#             churn_file='customer_churn_predictions.csv'
#         )
        
#         # Generate dashboard
#         print("📊 Creating visualizations...")
#         dashboard.generate_dashboard('marketing_dashboard.html')
        
#     except FileNotFoundError as e:
#         print(f"❌ Error: Could not find CSV file - {e}")
#         print("📁 Please ensure all CSV files are in the same directory as this script")
#     except Exception as e:
#         print(f"❌ Error generating dashboard: {e}")
#         print("🔧 Please check your data format and try again")