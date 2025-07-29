import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Keep seaborn for potential static plots if desired, or for its styling capabilities
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import base64
from io import BytesIO
from datetime import datetime, timedelta
import random

# --- 1. Load Data ---
try:
    # Ensure these paths are correct, or place CSVs in the same directory as the script
    customer_churn_df = pd.read_csv('customer_churn_predictions.csv')
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


# Function to convert matplotlib figure to base64 image (only for non-Plotly charts like WordCloud)
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# --- 3. Plotting Functions for each dashboard cell ---

# 3.1 Campaign dashboard - Funnel charts (Plotly Express)
def plot_funnel_chart(df):
    funnel_stages_order = ['Sent', 'Viewed', 'Clicked', 'AddedToCart', 'Purchased-Loyalty']

    # Count unique customers at each funnel stage
    funnel_counts = {
        'Sent': df[df['campaign_open'].notna()]['customer_id'].nunique(), # Customers where campaign status is logged
        'Viewed': df[df['campaign_open'] == 'Yes']['customer_id'].nunique(),
        'Clicked': df[df['campaign_click'] == 'Yes']['customer_id'].nunique(),
        'AddedToCart': df[df['product_in_cart'] == 'Yes']['customer_id'].nunique(),
        'Purchased-Loyalty': df[df['conversion_flag'] == 'Yes']['customer_id'].nunique()
    }
    
    funnel_data = pd.DataFrame(list(funnel_counts.items()), columns=['Stage', 'Customers'])
    funnel_data['Stage'] = pd.Categorical(funnel_data['Stage'], categories=funnel_stages_order, ordered=True)
    funnel_data = funnel_data.sort_values('Stage')

    fig = px.funnel(funnel_data, x='Customers', y='Stage', title='Customer Journey Funnel')
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig.to_html(full_html=False, include_plotlyjs='cdn') # Return HTML string

# 3.2 Best time vs channel Heatmap (Plotly Graph Objects)
def plot_time_channel_heatmap(df):
    heatmap_data = df.groupby(['time_group', 'social_media_platform']).size().unstack(fill_value=0)
    time_order = ['Early Morning (6-9am)', 'Business Hours (9am-4pm)', 'Evening (4-7pm)', 'Night (7pm-5am)']
    heatmap_data = heatmap_data.reindex(time_order)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='YlGnBu',
        text=heatmap_data.values, # Show actual values on heatmap cells
        texttemplate="%{text}",
        hoverinfo='x+y+z'
    ))

    fig.update_layout(
        title='Interactions by Time Group and Channel',
        xaxis_title='Channel',
        yaxis_title='Time Group',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

# 3.3 New / re-engaged / existing customers (Plotly Express Pie Chart)
def plot_customer_segments_pie(customer_churn_df, journey_entry_df):
    # Ensure 'days_since_last_transaction' and 'total_transactions' are numeric
    customer_churn_df['days_since_last_transaction'] = pd.to_numeric(customer_churn_df['days_since_last_transaction'], errors='coerce').fillna(365)
    customer_churn_df['total_transactions'] = pd.to_numeric(customer_churn_df['total_transactions'], errors='coerce').fillna(0)


    customer_segments = {
        'New Customers': 0,
        'Re-engaged Customers': 0,
        'Existing Customers': 0
    }

    unique_customers = customer_churn_df['customer_id'].unique()

    for cust_id in unique_customers:
        cust_data = customer_churn_df[customer_churn_df['customer_id'] == cust_id].iloc[0]
        days_since_last = cust_data['days_since_last_transaction']
        total_trans = cust_data['total_transactions']

        if days_since_last < 30 and total_trans < 5:
            customer_segments['New Customers'] += 1
        elif 90 <= days_since_last < 180 and total_trans >= 1: # Customers with a recent gap, now active
            customer_segments['Re-engaged Customers'] += 1
        elif days_since_last < 90 and total_trans >= 5: # Active, established customers
            customer_segments['Existing Customers'] += 1
        else:
            customer_segments['Existing Customers'] += 1 # Default for others

    segments_df = pd.DataFrame(list(customer_segments.items()), columns=['Segment', 'Count'])

    fig = px.pie(segments_df, values='Count', names='Segment', title='Customer Segments Distribution')
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


# 3.4 Localized offers at branch performance (Plotly Scatter Mapbox)
def plot_localized_offers_performance(df):
    offers_df = df[df['offer_applied'] != 'No Offer'].copy()

    # Create dummy lat/lon if not present for visualization (replace with actual data if available)
    if 'latitude' not in offers_df.columns or 'longitude' not in offers_df.columns:
        print("Warning: 'latitude' or 'longitude' not found in journey_entry_df for map. Generating dummy coordinates.")
        # Generate dummy lat/lon within a plausible range for Kenya (approximate Nairobi/surrounding)
        offers_df['latitude'] = offers_df['customer_id'].apply(lambda x: -1.2 + random.uniform(-0.5, 0.5))
        offers_df['longitude'] = offers_df['customer_id'].apply(lambda x: 36.8 + random.uniform(-0.5, 0.5))
        # Add a dummy branch_id if not present
        if 'branch_id' not in offers_df.columns:
            offers_df['branch_id'] = offers_df['customer_id'].apply(lambda x: f"Branch_{random.randint(1, 10)}")


    if offers_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for offers applied or missing location data. Please ensure 'offer_applied' column has values other than 'No Offer' and 'latitude'/'longitude' columns exist.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color="red")
        )
        fig.update_layout(title_text='Localized Offers Performance (No Data)', height=300, margin=dict(l=20, r=20, t=40, b=20))
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    branch_offers_performance = offers_df.groupby(['branch_id', 'latitude', 'longitude']).agg(
        total_offers_applied=('offer_applied', 'count'),
        conversions_with_offer=('conversion_flag', lambda x: (x == 'Yes').sum())
    ).reset_index()

    branch_offers_performance['conversion_rate_with_offer'] = branch_offers_performance.apply(
        lambda row: (row['conversions_with_offer'] / row['total_offers_applied']) if row['total_offers_applied'] > 0 else 0,
        axis=1
    )

    fig = px.scatter_mapbox(
        branch_offers_performance,
        lat="latitude",
        lon="longitude",
        size="total_offers_applied",
        color="conversion_rate_with_offer",
        color_continuous_scale=px.colors.sequential.Viridis,
        size_max=30,
        zoom=5,
        center={"lat": -1.286389, "lon": 36.817223}, # Centered around Nairobi
        mapbox_style="open-street-map",
        hover_name="branch_id",
        hover_data={
            "latitude": False,
            "longitude": False,
            "total_offers_applied": True,
            "conversions_with_offer": True,
            "conversion_rate_with_offer": ":.2%"
        },
        title='Localized Offers Performance by Branch Location'
    )
    fig.update_layout(height=300, margin={"r":0,"t":50,"l":0,"b":0})
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

# 3.5 Churn prediction vs LTV (Plotly Express Scatter plot)
def plot_churn_ltv_scatter(df):
    fig = px.scatter(df, x='customer_lifetime_value', y='churn_probability',
                     color='risk_level', hover_name='customer_id',
                     title='Churn Probability vs. Customer Lifetime Value',
                     labels={'customer_lifetime_value': 'Customer Lifetime Value',
                             'churn_probability': 'Churn Probability'},
                     color_discrete_map={'High Risk': 'red', 'Medium Risk': 'orange', 'Low Risk': 'green'}
                    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


# 3.6 At risk customers histogram (age bin) (Plotly Express Histogram)
def plot_at_risk_customers_histogram(df):
    # Ensure customer_age is numeric
    df['customer_age'] = pd.to_numeric(df['customer_age'], errors='coerce')
    at_risk_df = df[df['risk_level'].isin(['High Risk', 'Medium Risk'])].dropna(subset=['customer_age'])
    
    if at_risk_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No 'High Risk' or 'Medium Risk' customers with valid age data found.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color="red")
        )
        fig.update_layout(title_text='At Risk Customers Age Distribution (No Data)', height=300, margin=dict(l=20, r=20, t=40, b=20))
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    fig = px.histogram(at_risk_df, x='customer_age', nbins=10,
                       title='Age Distribution of At-Risk Customers',
                       labels={'customer_age': 'Customer Age', 'count': 'Number of Customers'},
                       color_discrete_sequence=['salmon'])
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


# 3.7 Social media trending (keywords / hashtags as word cloud) (Matplotlib + base64 for static image)
# Note: Plotly does not have a native WordCloud. This remains a static image for simplicity.
def plot_social_media_wordcloud(df_sentiment, df_journey_entry): # Pass both DFs
    all_hashtags = []
    if 'hashtags' in df_sentiment.columns:
        all_hashtags.extend(df_sentiment['hashtags'].dropna().tolist())
    if 'hashtags' in df_journey_entry.columns:
        all_hashtags.extend(df_journey_entry['hashtags'].dropna().tolist())

    processed_hashtags = []
    for tags in all_hashtags:
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


# 3.8 Trend view. No. of campaigns vs Sent (Plotly Dual Axis Line + Bar)
def plot_campaign_trends(df):
    campaign_summary = df.groupby('campaign_name').agg(
        total_sent=('campaign_open', lambda x: (x == 'Yes').sum()),
        total_campaign_entries=('campaign_name', 'size') # Count of entries per campaign
    ).reset_index()

    campaign_summary = campaign_summary.sort_values(by='total_sent', ascending=False)
    
    if campaign_summary.empty:
        fig = go.Figure()
        fig.add_annotation(text="No campaign data available.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=12, color="red"))
        fig.update_layout(title_text='Campaign Entries vs. Sent/Opened Status (No Data)', height=300, margin=dict(l=20, r=20, t=40, b=20))
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar chart for total campaign entries
    fig.add_trace(
        go.Bar(x=campaign_summary['campaign_name'], y=campaign_summary['total_campaign_entries'], name='Total Campaign Entries', marker_color='skyblue'),
        secondary_y=False,
    )

    # Line plot for Total Sent (Opened)
    fig.add_trace(
        go.Scatter(x=campaign_summary['campaign_name'], y=campaign_summary['total_sent'], name='Total Sent (Opened)', mode='lines+markers', line=dict(color='red')),
        secondary_y=True,
    )

    fig.update_layout(
        title_text='Campaign Entries vs. Sent/Opened Status',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(x=1.1, y=1, xanchor='right', yanchor='top')
    )
    fig.update_xaxes(title_text="Campaign Name", tickangle=45)
    fig.update_yaxes(title_text="Total Campaign Entries (Count)", secondary_y=False)
    fig.update_yaxes(title_text="Total Sent (Opened)", secondary_y=True)
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


# 3.9 Emerging customer segments (Plotly Express Bar Chart)
def plot_emerging_customer_segments(df_churn, df_journey):
    # Ensure numerical columns are correctly handled
    df_churn['customer_age'] = pd.to_numeric(df_churn['customer_age'], errors='coerce').fillna(0)
    df_churn['return_rate'] = pd.to_numeric(df_churn['return_rate'], errors='coerce').fillna(0)
    df_churn['days_since_last_transaction'] = pd.to_numeric(df_churn['days_since_last_transaction'], errors='coerce').fillna(365)
    df_churn['total_spent'] = pd.to_numeric(df_churn['total_spent'], errors='coerce').fillna(0)

    emerging_segments_counts = {
        "Age 30-46 & Active": 0,
        "High Return Rate (Recent Activity)": 0,
        "Low Spending, High Engagement": 0
    }

    # Segment 1: Age 30-46 & Active
    segment1_customers = df_churn[(df_churn['customer_age'] >= 30) & (df_churn['customer_age'] <= 46)]
    emerging_segments_counts["Age 30-46 & Active"] = segment1_customers['customer_id'].nunique()

    # Segment 2: High Return Rate (Recent Activity) - simplified
    segment2_customers = df_churn[(df_churn['return_rate'] > 0.1) & (df_churn['days_since_last_transaction'] < 60)]
    emerging_segments_counts["High Return Rate (Recent Activity)"] = segment2_customers['customer_id'].nunique()

    # Segment 3: Low Spending, High Engagement - simplified
    low_spending_customers = df_churn[df_churn['total_spent'] < 1000000]['customer_id'].tolist()
    engaged_customers = df_journey[df_journey['campaign_click'] == 'Yes']['customer_id'].tolist()
    segment3_customers = set(low_spending_customers).intersection(set(engaged_customers))
    emerging_segments_counts["Low Spending, High Engagement"] = len(segment3_customers)

    segments_df = pd.DataFrame(list(emerging_segments_counts.items()), columns=['Segment', 'Count'])

    fig = px.bar(segments_df, x='Segment', y='Count',
                 title='Emerging Customer Segments',
                 labels={'Count': 'Number of Customers', 'Segment': 'Segment'},
                 color='Segment', # Color by segment for distinct bars
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), xaxis={'tickangle': 45})
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


# --- 4. Generate HTML Dashboard ---
def generate_dashboard_html(plots):
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Marketing Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; color: #333; }}
            .dashboard-container {{ padding: 20px; }}
            .card {{
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.05);
                margin-bottom: 20px;
                background-color: #fff;
                overflow: hidden; /* Ensures Plotly's overflow is handled */
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
            .plotly-graph-container {{
                width: 100%;
                height: 300px; /* Standard height for plots */
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            .plotly-graph-container img {{
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
            }}
        </style>
    </head>
    <body>
        <div class="container-fluid dashboard-container">
            <h1 class="text-center mb-4 text-primary">Marketing Insights Dashboard</h1>

            <div class="row">
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">Campaign Funnel</div>
                        <div class="card-body">
                            <div class="plotly-graph-container">
                                {plot1_html}
                            </div>
                        </div>
                    </div>
                </div>

                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">Best Time vs Channel Interactions</div>
                        <div class="card-body">
                            <div class="plotly-graph-container">
                                {plot2_html}
                            </div>
                        </div>
                    </div>
                </div>

                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">Customer Segments Distribution</div>
                        <div class="card-body">
                            <div class="plotly-graph-container">
                                {plot3_html}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row">
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">Localized Offers Performance</div>
                        <div class="card-body">
                            <div class="plotly-graph-container">
                                {plot4_html}
                            </div>
                        </div>
                    </div>
                </div>

                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">Churn Probability vs LTV</div>
                        <div class="card-body">
                            <div class="plotly-graph-container">
                                {plot5_html}
                            </div>
                        </div>
                    </div>
                </div>

                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">At-Risk Customers Age Distribution</div>
                        <div class="card-body">
                            <div class="plotly-graph-container">
                                {plot6_html}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row">
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">Social Media Trending</div>
                        <div class="card-body">
                            <div class="plotly-graph-container">
                                <img src="data:image/png;base64,{plot7_base64}" alt="Social Media Word Cloud">
                            </div>
                        </div>
                    </div>
                </div>

                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">Campaign vs Sent Trend</div>
                        <div class="card-body">
                            <div class="plotly-graph-container">
                                {plot8_html}
                            </div>
                        </div>
                    </div>
                </div>

                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">Emerging Customer Segments</div>
                        <div class="card-body">
                            <div class="plotly-graph-container">
                                {plot9_html}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html_template.format(**plots)

# --- 5. Main Execution ---
if __name__ == "__main__":
    # Generate all plots
    # Note: WordCloud is still a static image, others are interactive Plotly HTML
    plot1_html = plot_funnel_chart(journey_entry_df.copy())
    plot2_html = plot_time_channel_heatmap(journey_entry_df.copy())
    plot3_html = plot_customer_segments_pie(customer_churn_df.copy(), journey_entry_df.copy())
    plot4_html = plot_localized_offers_performance(journey_entry_df.copy())
    plot5_html = plot_churn_ltv_scatter(customer_churn_df.copy())
    plot6_html = plot_at_risk_customers_histogram(customer_churn_df.copy())
    plot7_base64 = plot_social_media_wordcloud(sentiment_df.copy(), journey_entry_df.copy()) # Pass both DFs
    plot8_html = plot_campaign_trends(journey_entry_df.copy())
    plot9_html = plot_emerging_customer_segments(customer_churn_df.copy(), journey_entry_df.copy())

    plots_dict = {
        'plot1_html': plot1_html,
        'plot2_html': plot2_html,
        'plot3_html': plot3_html,
        'plot4_html': plot4_html,
        'plot5_html': plot5_html,
        'plot6_html': plot6_html,
        'plot7_base64': plot7_base64, # This remains base64
        'plot8_html': plot8_html,
        'plot9_html': plot9_html,
    }

    html_output = generate_dashboard_html(plots_dict)

    # Save to an HTML file
    dashboard_filename = 'marketing_dashboard.html'
    with open(dashboard_filename, 'w', encoding='utf-8') as f:
        f.write(html_output)

    print(f"Dashboard '{dashboard_filename}' generated successfully!")
    print("Open this HTML file in your web browser to view the interactive dashboard.")
    print("\nIf you encounter issues with Plotly static image export (e.g., for wordcloud if you change it), ensure you have 'kaleido' installed: pip install kaleido")
