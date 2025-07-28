import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import re
from collections import Counter
import plotly.offline as pyo
import json


class MarketingDashboard:
    def __init__(self, journey_file, sentiment_file, churn_file):
        """Initialize the dashboard with data files"""
        self.journey_df = pd.read_csv(journey_file)
        self.sentiment_df = pd.read_csv(sentiment_file)
        self.churn_df = pd.read_csv(churn_file)
        
        # Clean and prepare data
        self._prepare_data()
    
    def _prepare_data(self):
        """Clean and prepare data for visualization"""
        # Convert dates
        self.journey_df['stage_date'] = pd.to_datetime(self.journey_df['stage_date'])
        self.sentiment_df['date'] = pd.to_datetime(self.sentiment_df['date'])
        
        # Extract time components
        self.journey_df['hour'] = self.journey_df['stage_date'].dt.hour
        self.journey_df['day_of_week'] = self.journey_df['stage_date'].dt.day_name()
        self.journey_df['date'] = self.journey_df['stage_date'].dt.date
        
        # Process hashtags - handle NaN values
        self.journey_df['hashtags'] = self.journey_df['hashtags'].fillna('')
        self.sentiment_df['hashtags'] = self.sentiment_df['hashtags'].fillna('')
        
        # Clean string columns
        for col in ['campaign_open', 'campaign_click', 'conversion_flag', 'product_in_cart']:
            if col in self.journey_df.columns:
                self.journey_df[col] = self.journey_df[col].fillna('No')
    
    def create_funnel_chart(self):
        """Create campaign funnel chart using actual journey stages"""
        # Calculate funnel metrics from actual data
        stage_metrics = self.journey_df.groupby('stage').agg({
            'customer_id': 'nunique',
            'journey_id': 'count'
        }).reset_index()
        
        # Order stages by typical customer journey flow
        stage_order = ['Awareness', 'Consideration', 'Lead', 'Purchase', 'Service', 'Loyalty', 'Advocacy']
        
        # Filter to only stages that exist in data and order them
        existing_stages = stage_metrics['stage'].unique()
        ordered_stages = [stage for stage in stage_order if stage in existing_stages]
        
        # Add any remaining stages not in our predefined order
        remaining_stages = [stage for stage in existing_stages if stage not in ordered_stages]
        final_order = ordered_stages + remaining_stages
        
        # Reorder data
        stage_metrics['stage'] = pd.Categorical(stage_metrics['stage'], categories=final_order, ordered=True)
        stage_metrics = stage_metrics.sort_values('stage').reset_index(drop=True)
        
        fig = go.Figure(go.Funnel(
            y=stage_metrics['stage'],
            x=stage_metrics['customer_id'],
            textinfo="value+percent initial",
            marker=dict(color=["#5DADE2", "#48C9B0", "#F7DC6F", "#F8C471", "#EC7063", "#BB8FCE", "#85C1E9"][:len(stage_metrics)]),
            textfont=dict(size=12, color="white")
        ))
        
        fig.update_layout(
            title="Campaign Funnel - Customer Journey Stages",
            title_font_size=14,
            height=350,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        return fig
    
    def create_heatmap(self):
        """Create best time vs stage heatmap using journey_entry.csv data"""
        # Define time periods based on hour with your specified groupings
        def get_time_period(hour):
            if 6 <= hour < 9:
                return "Early Morning (6-9)"
            elif 9 <= hour < 16:
                return "Business Hours (9-4pm)"
            elif 16 <= hour < 19:
                return "Evening (4-7pm)"
            else:
                return "Night (7pm-5am)"
        
        # Add time period column to journey data
        self.journey_df['stage_date'] = pd.to_datetime(self.journey_df['stage_date'])
        self.journey_df['time_period'] = self.journey_df['stage_date'].dt.hour.apply(get_time_period)
        
        # Calculate engagement metrics by stage and time_period
        heatmap_data = self.journey_df.groupby(['stage', 'time_period']).agg({
            'journey_id': 'count',
            'campaign_open': lambda x: (x == 'Yes').sum(),
            'campaign_click': lambda x: (x == 'Yes').sum(),
            'conversion_flag': lambda x: (x == 'Yes').sum()
        }).reset_index()
        
        # Calculate engagement rate (opens + clicks + conversions) / total interactions
        heatmap_data['engagement_rate'] = (
            heatmap_data['campaign_open'] + 
            heatmap_data['campaign_click'] + 
            heatmap_data['conversion_flag']
        ) / heatmap_data['journey_id']
        
        # Pivot for heatmap
        heatmap_pivot = heatmap_data.pivot(
            index='stage', 
            columns='time_period', 
            values='engagement_rate'
        ).fillna(0)
        
        # Ensure time periods are in the correct order
        time_order = ['Early Morning (6-9)', 'Business Hours (9-4pm)', 'Evening (4-7pm)', 'Night (7pm-5am)']
        available_times = [time for time in time_order if time in heatmap_pivot.columns]
        heatmap_pivot = heatmap_pivot[available_times]
        
        fig = px.imshow(
            heatmap_pivot.values,
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            color_continuous_scale='Blues',
            aspect='auto',
            labels=dict(color='Engagement Rate', x='Time Period', y='Journey Stage')
        )
        
        fig.update_layout(
            title='Best Time vs Journey Stage Heatmap',
            title_font_size=14,
            height=350,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title='Time Period',
            yaxis_title='Journey Stage'
        )
        
        return fig


    
    def create_customer_pie_chart(self):
        """Create customer segmentation based on actual journey data"""
        # Analyze customer behavior patterns
        customer_analysis = self.journey_df.groupby('customer_id').agg({
            'stage': ['count', 'nunique'],
            'conversion_flag': lambda x: (x == 'Yes').sum(),
            'campaign_click': lambda x: (x == 'Yes').sum(),
            'stage_date': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        customer_analysis.columns = ['customer_id', 'total_interactions', 'unique_stages', 'conversions', 'clicks', 'first_interaction', 'last_interaction']
        
        # Calculate engagement duration
        customer_analysis['engagement_days'] = (customer_analysis['last_interaction'] - customer_analysis['first_interaction']).dt.days
        
        # Categorize customers based on actual behavior
        def categorize_customer(row):
            if row['unique_stages'] <= 2 and row['total_interactions'] <= 3:
                return "New Customers"
            elif row['conversions'] > 0 or row['clicks'] > 2:
                return "Re-engaged Customers"
            else:
                return "Existing Customers"
        
        customer_analysis['category'] = customer_analysis.apply(categorize_customer, axis=1)
        category_counts = customer_analysis['category'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=category_counts.index,
            values=category_counts.values,
            hole=0.4,
            marker_colors=['#3498DB', '#E74C3C', '#2ECC71'],
            textinfo='label+percent+value'
        )])
        
        fig.update_layout(
            title="Customer Segmentation (Based on Journey Behavior)",
            title_font_size=14,
            height=350,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        return fig
    
    def create_campaign_performance(self):
        """Create campaign performance chart using actual data"""
        # Calculate campaign performance metrics
        campaign_performance = self.journey_df.groupby('campaign_name').agg({
            'customer_id': 'nunique',
            'campaign_open': lambda x: (x == 'Yes').sum(),
            'campaign_click': lambda x: (x == 'Yes').sum(),
            'conversion_flag': lambda x: (x == 'Yes').sum(),
            'journey_id': 'count'
        }).reset_index()
        
        # Calculate rates
        campaign_performance['open_rate'] = campaign_performance['campaign_open'] / campaign_performance['journey_id']
        campaign_performance['click_rate'] = campaign_performance['campaign_click'] / campaign_performance['journey_id']
        campaign_performance['conversion_rate'] = campaign_performance['conversion_flag'] / campaign_performance['journey_id']
        
        # Sort by total interactions and take top 10
        campaign_performance = campaign_performance.sort_values('journey_id', ascending=True).tail(10)
        
        fig = go.Figure()
        
        # Add bars for different metrics
        fig.add_trace(go.Bar(
            name='Open Rate',
            x=campaign_performance['open_rate'],
            y=campaign_performance['campaign_name'],
            orientation='h',
            marker_color='#3498DB',
            opacity=0.8
        ))
        
        fig.add_trace(go.Bar(
            name='Click Rate',
            x=campaign_performance['click_rate'],
            y=campaign_performance['campaign_name'],
            orientation='h',
            marker_color='#E74C3C',
            opacity=0.8
        ))
        
        fig.add_trace(go.Bar(
            name='Conversion Rate',
            x=campaign_performance['conversion_rate'],
            y=campaign_performance['campaign_name'],
            orientation='h',
            marker_color='#2ECC71',
            opacity=0.8
        ))
        
        fig.update_layout(
            title="Campaign Performance (Open/Click/Conversion Rates)",
            title_font_size=14,
            height=350,
            margin=dict(l=10, r=10, t=40, b=10),
            barmode='group',
            xaxis_title="Rate",
            yaxis_title="Campaign"
        )
        
        return fig
    
    def create_churn_ltv_scatter(self):
        """Create churn prediction vs LTV scatter plot using actual churn data"""
        # Calculate LTV from actual data
        self.churn_df['ltv'] = self.churn_df['total_spent'] / self.churn_df['total_transactions']
        
        # Handle any infinite or NaN values
        self.churn_df['ltv'] = self.churn_df['ltv'].replace([np.inf, -np.inf], np.nan)
        self.churn_df = self.churn_df.dropna(subset=['ltv'])
        
        fig = px.scatter(
            self.churn_df,
            x='churn_probability',
            y='ltv',
            color='risk_level',
            size='total_transactions',
            hover_data=['customer_id', 'total_spent', 'days_since_last_transaction'],
            color_discrete_map={'High Risk': '#E74C3C', 'Medium Risk': '#F39C12', 'Low Risk': '#27AE60'},
            size_max=20
        )
        
        fig.update_layout(
            title="Churn Probability vs Customer Lifetime Value",
            title_font_size=14,
            xaxis_title="Churn Probability",
            yaxis_title="Lifetime Value ($)",
            height=350,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        return fig
    
    def create_risk_histogram(self):
        """Create at-risk customers histogram using actual age data"""
        fig = px.histogram(
            self.churn_df,
            x='customer_age',
            color='risk_level',
            nbins=15,
            color_discrete_map={'High Risk': '#E74C3C', 'Medium Risk': '#F39C12', 'Low Risk': '#27AE60'},
            title="Customer Age Distribution by Risk Level"
        )
        
        fig.update_layout(
            title="At Risk Customers by Age Distribution",
            title_font_size=14,
            xaxis_title="Customer Age",
            yaxis_title="Number of Customers",
            height=350,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        return fig
    
    def create_social_keywords_chart(self):
        """Create social media trending keywords from actual hashtag data"""
        # Extract hashtags from both datasets
        all_hashtags = []
        
        # Process journey hashtags
        for hashtags in self.journey_df['hashtags']:
            if hashtags and isinstance(hashtags, str):
                tags = re.findall(r'#\w+', hashtags.lower())
                all_hashtags.extend([tag.replace('#', '') for tag in tags])
        
        # Process sentiment hashtags
        for hashtags in self.sentiment_df['hashtags']:
            if hashtags and isinstance(hashtags, str):
                tags = re.findall(r'#\w+', hashtags.lower())
                all_hashtags.extend([tag.replace('#', '') for tag in tags])
        
        # Count frequency and get top hashtags
        if all_hashtags:
            hashtag_counts = Counter(all_hashtags)
            top_hashtags = dict(hashtag_counts.most_common(12))
            
            fig = go.Figure(go.Bar(
                x=list(top_hashtags.values()),
                y=list(top_hashtags.keys()),
                orientation='h',
                marker_color='#3498DB',
                text=list(top_hashtags.values()),
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Social Media Trending Keywords (From Actual Hashtags)",
                title_font_size=14,
                xaxis_title="Frequency",
                yaxis_title="Hashtags",
                height=350,
                margin=dict(l=10, r=10, t=40, b=10)
            )
        else:
            # Fallback if no hashtags found
            fig = go.Figure()
            fig.add_annotation(
                text="No hashtag data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Social Media Trending Keywords",
                title_font_size=14,
                height=350,
                margin=dict(l=10, r=10, t=40, b=10)
            )
        
        return fig
    
    def create_campaign_trend(self):
        """Create campaign trends over time using actual date data"""
        # Group by date and count campaigns
        daily_campaigns = self.journey_df.groupby(['date', 'campaign_name']).size().reset_index(name='interactions')
        daily_totals = daily_campaigns.groupby('date')['interactions'].sum().reset_index()
        
        # Get top 3 campaigns for individual tracking
        top_campaigns = self.journey_df['campaign_name'].value_counts().head(3).index
        
        fig = go.Figure()
        
        # Add total campaigns line
        fig.add_trace(go.Scatter(
            x=daily_totals['date'], 
            y=daily_totals['interactions'], 
            mode='lines+markers', 
            name='Total Daily Interactions',
            line=dict(color='#2C3E50', width=3),
            marker=dict(size=6)
        ))
        
        # Add individual campaign lines
        colors = ['#E74C3C', '#3498DB', '#2ECC71']
        for i, campaign in enumerate(top_campaigns):
            campaign_data = daily_campaigns[daily_campaigns['campaign_name'] == campaign]
            campaign_daily = campaign_data.groupby('date')['interactions'].sum().reset_index()
            
            fig.add_trace(go.Scatter(
                x=campaign_daily['date'], 
                y=campaign_daily['interactions'], 
                mode='lines+markers',
                name=campaign,
                line=dict(color=colors[i], width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title="Campaign Interaction Trends Over Time",
            title_font_size=14,
            height=350,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Date",
            yaxis_title="Number of Interactions",
            hovermode='x unified'
        )
        
        return fig
    
    def create_sentiment_analysis(self):
        """Create sentiment analysis from actual sentiment data"""
        # Analyze sentiment distribution
        sentiment_counts = self.sentiment_df['review_sentiment'].value_counts()
        
        # Also analyze by platform if available
        platform_sentiment = self.sentiment_df.groupby(['social_media_platform', 'review_sentiment']).size().unstack(fill_value=0)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Overall Sentiment Distribution', 'Sentiment by Platform'),
            specs=[[{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Overall sentiment pie chart
        fig.add_trace(
            go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                hole=0.4,
                marker_colors=['#2ECC71', '#E74C3C', '#F39C12']
            ),
            row=1, col=1
        )
        
        # Sentiment by platform bar chart
        for sentiment in platform_sentiment.columns:
            fig.add_trace(
                go.Bar(
                    name=sentiment,
                    x=platform_sentiment.index,
                    y=platform_sentiment[sentiment],
                    marker_color={'Positive': '#2ECC71', 'Negative': '#E74C3C', 'Neutral': '#F39C12'}.get(sentiment, '#95A5A6')
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title="Customer Sentiment Analysis",
            title_font_size=14,
            height=350,
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=True
        )
        
        return fig
        
    def create_emerging_segments_analysis(self):
        """Create emerging customer segments based on actual data analysis"""
        # Analyze customer segments based on multiple factors
        customer_segments = self.journey_df.merge(
            self.churn_df[['customer_id', 'risk_level', 'total_spent']], 
            on='customer_id', 
            how='left'
        )
        
        # Group by age ranges and behavior
        customer_segments['age_group'] = pd.cut(
            customer_segments['customer_age'], 
            bins=[0, 25, 35, 50, 65, 100], 
            labels=['18-25', '26-35', '36-50', '51-65', '65+']
        )
        
        segment_analysis = customer_segments.groupby(['age_group', 'risk_level'], observed=False).agg({
            'customer_id': 'nunique',
            'total_spent': 'mean',
            'conversion_flag': lambda x: (x == 'Yes').sum()
        }).reset_index()
        
        segment_analysis = segment_analysis.dropna()
        
        # Create insights text based on actual data
        insights = []
        
        for _, row in segment_analysis.iterrows():
            if row['customer_id'] > 10:  # Only include segments with sufficient data
                avg_spent = row['total_spent'] if pd.notna(row['total_spent']) else 0
                insights.append(
                    f"<strong>{row['age_group']} - {row['risk_level']}:</strong><br>"
                    f"<span style='color: #666;'>{int(row['customer_id'])} customers, "
                    f"Avg. Spent: ${avg_spent:,.0f}, "
                    f"Conversions: {int(row['conversion_flag'])}</span>"
                )
        
        return insights[:6]  # Return top 6 segments
    
    def generate_dashboard(self, output_filename='marketing_dashboard.html'):
        """Generate complete 3x3 dashboard using only actual CSV data"""
        
        print("📊 Creating visualizations from actual data...")
        
        # Create all charts using real data
        funnel_fig = self.create_funnel_chart()
        heatmap_fig = self.create_heatmap()
        pie_fig = self.create_customer_pie_chart()
        campaign_fig = self.create_campaign_performance()
        scatter_fig = self.create_churn_ltv_scatter()
        histogram_fig = self.create_risk_histogram()
        social_fig = self.create_social_keywords_chart()
        trend_fig = self.create_campaign_trend()
        
        # Create emerging segments from actual data
        segments_insights = self.create_emerging_segments_analysis()
        
        # Convert figures to JSON
        charts_json = {
            'funnel': funnel_fig.to_json(),
            'heatmap': heatmap_fig.to_json(),
            'pie': pie_fig.to_json(),
            'campaign': campaign_fig.to_json(),
            'scatter': scatter_fig.to_json(),
            'histogram': histogram_fig.to_json(),
            'social': social_fig.to_json(),
            'trend': trend_fig.to_json()
        }
        
        # Create segments HTML from actual data
        segments_html = f"""
        <h3 style='color: #2c3e50; margin-top: 0; font-size: 16px;'>Emerging Customer Segments (Data-Driven)</h3>
        <div style='font-size: 13px; line-height: 1.4;'>
        """
        
        for i, insight in enumerate(segments_insights, 1):
            segments_html += f"<div style='margin-bottom: 12px;'>{i}. {insight}</div>"
        
        segments_html += "</div>"
        
        # Generate HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Marketing Analytics Dashboard - Data-Driven Insights</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 15px;
                    background-color: #f5f6fa;
                }}
                .dashboard-container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 12px;
                    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                .dashboard-header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 25px;
                    text-align: center;
                }}
                .dashboard-header h1 {{
                    margin: 0 0 10px 0;
                    font-size: 28px;
                    font-weight: 600;
                }}
                .dashboard-header p {{
                    margin: 0;
                    opacity: 0.9;
                    font-size: 16px;
                }}
                .dashboard-grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr 1fr;
                    gap: 15px;
                    padding: 20px;
                }}
                .chart-container {{
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0 3px 10px rgba(0,0,0,0.08);
                    padding: 15px;
                    min-height: 350px;
                    border: 1px solid #e1e8ed;
                }}
                .segments-container {{
                    background: #f8f9fa;
                    border-radius: 10px;
                    padding: 20px;
                    border: 1px solid #dee2e6;
                    min-height: 310px;
                    overflow-y: auto;
                }}
                .chart-container:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 6px 20px rgba(0,0,0,0.12);
                    transition: all 0.3s ease;
                }}
                .data-badge {{
                    background: #e8f5e8;
                    color: #2d5a2d;
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: bold;
                }}
                @media (max-width: 1200px) {{
                    .dashboard-grid {{
                        grid-template-columns: 1fr 1fr;
                    }}
                }}
                @media (max-width: 768px) {{
                    .dashboard-grid {{
                        grid-template-columns: 1fr;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="dashboard-container">
                <div class="dashboard-header">
                    <h1>📊 Marketing Analytics Dashboard</h1>
                    <p>Real insights from your actual customer journey, sentiment, and churn data</p>
                    <div style="margin-top: 10px;">
                        <span class="data-badge">✓ Real Data Only</span>
                        <span class="data-badge">✓ {len(self.journey_df):,} Journey Records</span>
                        <span class="data-badge">✓ {len(self.sentiment_df):,} Sentiment Records</span>
                        <span class="data-badge">✓ {len(self.churn_df):,} Customer Profiles</span>
                    </div>
                </div>
                <div class="dashboard-grid">
                    <div class="chart-container">
                        <div id="funnel-chart" style="height: 100%;"></div>
                    </div>
                    <div class="chart-container">
                        <div id="heatmap-chart" style="height: 100%;"></div>
                    </div>
                    <div class="chart-container">
                        <div id="pie-chart" style="height: 100%;"></div>
                    </div>
                    <div class="chart-container">
                        <div id="campaign-chart" style="height: 100%;"></div>
                    </div>
                    <div class="chart-container">
                        <div id="scatter-chart" style="height: 100%;"></div>
                    </div>
                    <div class="chart-container">
                        <div id="histogram-chart" style="height: 100%;"></div>
                    </div>
                    <div class="chart-container">
                        <div id="social-chart" style="height: 100%;"></div>
                    </div>
                    <div class="chart-container">
                        <div id="trend-chart" style="height: 100%;"></div>
                    </div>
                    <div class="segments-container">
                        {segments_html}
                    </div>
                </div>
            </div>
            
            <script>
                // Chart configurations
                const config = {{
                    responsive: true,
                    displayModeBar: false,
                    displaylogo: false
                }};
                
                // Render all charts
                Plotly.newPlot('funnel-chart', {charts_json['funnel']}.data, {charts_json['funnel']}.layout, config);
                Plotly.newPlot('heatmap-chart', {charts_json['heatmap']}.data, {charts_json['heatmap']}.layout, config);
                Plotly.newPlot('pie-chart', {charts_json['pie']}.data, {charts_json['pie']}.layout, config);
                Plotly.newPlot('campaign-chart', {charts_json['campaign']}.data, {charts_json['campaign']}.layout, config);
                Plotly.newPlot('scatter-chart', {charts_json['scatter']}.data, {charts_json['scatter']}.layout, config);
                Plotly.newPlot('histogram-chart', {charts_json['histogram']}.data, {charts_json['histogram']}.layout, config);
                Plotly.newPlot('social-chart', {charts_json['social']}.data, {charts_json['social']}.layout, config);
                Plotly.newPlot('trend-chart', {charts_json['trend']}.data, {charts_json['trend']}.layout, config);
                
                // Make charts responsive
                window.addEventListener('resize', function() {{
                    Plotly.Plots.resize('funnel-chart');
                    Plotly.Plots.resize('heatmap-chart');
                    Plotly.Plots.resize('pie-chart');
                    Plotly.Plots.resize('campaign-chart');
                    Plotly.Plots.resize('scatter-chart');
                    Plotly.Plots.resize('histogram-chart');
                    Plotly.Plots.resize('social-chart');
                    Plotly.Plots.resize('trend-chart');
                }});
            </script>
        </body>
        </html>
        """
        
        # Save to file
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Dashboard generated successfully: {output_filename}")
        print(f"📂 File size: {len(html_content)/1024:.1f} KB")
        print(f"🔗 Open the file in your web browser to view the dashboard")
        
        return output_filename


# Usage example
if __name__ == "__main__":
    try:
        # Initialize dashboard
        print("🚀 Initializing Marketing Dashboard...")
        dashboard = MarketingDashboard(
            journey_file='journey_entry.csv',
            sentiment_file='sentiment.csv', 
            churn_file='customer_churn_predictions.csv'
        )
        
        # Generate dashboard
        print("📊 Creating visualizations...")
        dashboard.generate_dashboard('marketing_dashboard.html')
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find CSV file - {e}")
        print("📁 Please ensure all CSV files are in the same directory as this script")
    except Exception as e:
        print(f"❌ Error generating dashboard: {e}")
        print("🔧 Please check your data format and try again")
