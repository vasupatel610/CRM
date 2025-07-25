import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import re
from collections import Counter

class MarketingDashboard:
    def __init__(self, data_path="D:\\Mockup\\"):
        """Initialize the dashboard with data path"""
        self.data_path = data_path
        self.load_data()
        self.process_data()
    
    def load_data(self):
        """Load all CSV files"""
        try:
            self.churn_df = pd.read_csv(f"{self.data_path}customer_churn_predictions.csv")
            self.journey_df = pd.read_csv(f"{self.data_path}journey_entry.csv")
            self.sentiment_df = pd.read_csv(f"{self.data_path}sentiment.csv")
            print("Data loaded successfully!")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def process_data(self):
        """Process data for better visualizations"""
        # Convert dates
        # Ensure 'stage_date' is treated as datetime, handling potential errors
        self.journey_df['stage_date'] = pd.to_datetime(self.journey_df['stage_date'], errors='coerce')
        # Drop rows where 'stage_date' could not be parsed
        self.journey_df.dropna(subset=['stage_date'], inplace=True)
        
        self.journey_df['hour'] = self.journey_df['stage_date'].dt.hour
        self.journey_df['day_of_week'] = self.journey_df['stage_date'].dt.day_name()
        
        # Create age groups based on spending behavior (proxy)
        # Handle cases where 'total_spent' might be NaN or non-numeric before qcut
        if 'total_spent' in self.churn_df.columns:
            self.churn_df['total_spent'] = pd.to_numeric(self.churn_df['total_spent'], errors='coerce')
            self.churn_df.dropna(subset=['total_spent'], inplace=True)
            if not self.churn_df['total_spent'].empty and len(self.churn_df['total_spent'].unique()) >= 4:
                self.churn_df['spending_quartile'] = pd.qcut(self.churn_df['total_spent'], 
                                                              q=4, 
                                                              labels=['18-25', '26-35', '36-45', '46-55'])
            else:
                self.churn_df['spending_quartile'] = 'N/A' # Handle cases with insufficient unique values
        else:
            self.churn_df['spending_quartile'] = 'N/A' # Column not found
        
        # Calculate LTV estimate
        # Ensure 'total_spent' and 'days_since_last_transaction' are numeric
        self.churn_df['total_spent'] = pd.to_numeric(self.churn_df['total_spent'], errors='coerce').fillna(0)
        self.churn_df['days_since_last_transaction'] = pd.to_numeric(self.churn_df['days_since_last_transaction'], errors='coerce').fillna(1)
        self.churn_df['estimated_ltv'] = self.churn_df['total_spent'] / np.maximum(self.churn_df['days_since_last_transaction'], 1) * 365
        
        # Ensure 'risk_level' column exists for scatter plot, define based on churn_probability if not present
        if 'risk_level' not in self.churn_df.columns:
            self.churn_df['risk_level'] = self.churn_df['churn_probability'].apply(
                lambda x: 'High Risk' if x > 0.8 else ('Medium Risk' if x > 0.4 else 'Low Risk')
            )


    def create_campaign_gauge_charts(self, target_open_rate=70.0, target_click_rate=25.0, target_conversion_rate=3.0):
        """Create gauge charts for campaign open/click/conversion rates with dynamic thresholds and color coding"""
        # Calculate overall campaign metrics
        # Ensure there are records to avoid division by zero
        total_records = len(self.journey_df)
        if total_records == 0:
            open_rate = click_rate = conversion_rate = 0.0
        else:
            open_rate = (self.journey_df['campaign_open'] == 'Yes').sum() / total_records * 100
            click_rate = (self.journey_df['campaign_click'] == 'Yes').sum() / total_records * 100
            conversion_rate = (self.journey_df['conversion_flag'] == 'Yes').sum() / total_records * 100
        
        # Create subplots for 3 gauges
        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
            subplot_titles=('Open Rate', 'Click Rate', 'Conversion Rate'),
            horizontal_spacing=0.1
        )
        
        # Define step colors based on performance relative to target
        # Red: significantly below, Yellow: near target, Green: at or above target
        
        # Open Rate Gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=round(open_rate, 1),
            title={'text': "Open Rate"}, # Simplified title
            domain={'x': [0, 0.32], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, target_open_rate * 0.75], 'color': "#f08080"}, # Below 75% of target (Reddish)
                    {'range': [target_open_rate * 0.75, target_open_rate], 'color': "#ffd700"}, # 75% to target (Yellowish)
                    {'range': [target_open_rate, 100], 'color': "#90ee90"} # At or above target (Greenish)
                ],
                'threshold': {
                    'line': {'color': "darkblue", 'width': 4},
                    'thickness': 0.75,
                    'value': target_open_rate # Target as the threshold line
                }}
        ), row=1, col=1)
        
        # Click Rate Gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=round(click_rate, 1),
            title={'text': "Click Rate"},
            domain={'x': [0.34, 0.66], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#ff7f0e"},
                'steps': [
                    {'range': [0, target_click_rate * 0.75], 'color': "#f08080"},
                    {'range': [target_click_rate * 0.75, target_click_rate], 'color': "#ffd700"},
                    {'range': [target_click_rate, 100], 'color': "#90ee90"}
                ],
                'threshold': {
                    'line': {'color': "darkorange", 'width': 4},
                    'thickness': 0.75,
                    'value': target_click_rate
                }}
        ), row=1, col=2)
        
        # Conversion Rate Gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=round(conversion_rate, 1),
            title={'text': "Conversion Rate"},
            domain={'x': [0.68, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 20]}, # Max range for conversion rate typically lower, adjusted from 50 to 20
                'bar': {'color': "#2ca02c"},
                'steps': [
                    {'range': [0, target_conversion_rate * 0.75], 'color': "#f08080"},
                    {'range': [target_conversion_rate * 0.75, target_conversion_rate], 'color': "#ffd700"},
                    {'range': [target_conversion_rate, 20], 'color': "#90ee90"}
                ],
                'threshold': {
                    'line': {'color': "darkgreen", 'width': 4},
                    'thickness': 0.75,
                    'value': target_conversion_rate
                }}
        ), row=1, col=3)
        
        fig.update_layout(
            title={
                'text': "Campaign Performance Metrics (vs. Targets)",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            height=400,
            font={'family': "Arial"},
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig
    
    def create_best_time_channel_chart(self):
        """Create bar chart for best time/channel detection"""
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Best Hours for Engagement', 'Platform Performance (Positive Sentiment)'), # Clarified title
            specs=[[{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Best hours analysis - Ensure 'hour' column exists and is numeric
        if 'hour' in self.journey_df.columns:
            hourly_stats = self.journey_df.groupby('hour').agg(
                opens=('campaign_open', lambda x: (x == 'Yes').sum()),
                clicks=('campaign_click', lambda x: (x == 'Yes').sum())
            ).reset_index()
            hourly_stats['hour'] = hourly_stats['hour'].astype(int) # Ensure hour is int for plotting
            hourly_stats.sort_values('hour', inplace=True) # Sort by hour
        else:
            hourly_stats = pd.DataFrame({'hour': [], 'opens': [], 'clicks': []}) # Empty DataFrame if column missing
        
        # Add hourly engagement chart
        if not hourly_stats.empty:
            fig.add_trace(
                go.Bar(
                    x=hourly_stats['hour'],
                    y=hourly_stats['opens'],
                    name='Opens',
                    marker_color='lightblue',
                    yaxis='y',
                    offsetgroup=1
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=hourly_stats['hour'],
                    y=hourly_stats['clicks'],
                    name='Clicks',
                    marker_color='orange',
                    yaxis='y',
                    offsetgroup=2
                ),
                row=1, col=1
            )
        
        # Platform performance from sentiment data - Ensure 'social_media_platform' and 'review_sentiment' exist
        if 'social_media_platform' in self.sentiment_df.columns and 'review_sentiment' in self.sentiment_df.columns:
            platform_sentiment = self.sentiment_df.groupby('social_media_platform').agg(
                positive_sentiment=('review_sentiment', lambda x: (x == 'positive').sum() / len(x) * 100 if len(x) > 0 else 0)
            ).round(1).reset_index()
            platform_sentiment = platform_sentiment.sort_values('positive_sentiment', ascending=False) # Sort for better readability
        else:
            platform_sentiment = pd.DataFrame({'social_media_platform': [], 'positive_sentiment': []}) # Empty DataFrame
        
        if not platform_sentiment.empty:
            fig.add_trace(
                go.Bar(
                    x=platform_sentiment['social_media_platform'],
                    y=platform_sentiment['positive_sentiment'],
                    name='Positive Sentiment %',
                    marker_color='lightgreen',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_xaxes(title_text="Hour of Day", row=1, col=1, type='category') # Use type 'category' for discrete hours
        fig.update_xaxes(title_text="Social Platform", row=1, col=2, type='category')
        fig.update_yaxes(title_text="Engagement Count", row=1, col=1)
        fig.update_yaxes(title_text="Positive Sentiment (%)", row=1, col=2)
        
        fig.update_layout(
            title="Best Time/Channel Detection for Ads",
            height=400,
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=40, r=40, t=60, b=40),
            barmode='group' # Ensure bars for opens and clicks are grouped
        )
        
        return fig
    
    def create_churn_segment_pie_chart(self):
        """Create pie chart for segment recommendation based on behavior & predicted churn"""
        # Define segmentation logic
        def assign_segment(row):
            prob = row.get('churn_probability', 0)
            spent = row.get('total_spent', 0)
            days = row.get('days_since_last_transaction', np.inf)

            if prob >= 0.8 and spent >= 1000 and days <= 30:
                return 'Engaged High Risk'    # High spend, recent, but likely to churn
            elif prob >= 0.8:
                return 'At-Risk Low Engagement'  # Likely churners who haven’t spent or engaged recently
            elif prob >= 0.4 and spent >= 1000:
                return 'Loyal Medium Risk'     # Good spenders with moderate churn risk
            elif prob >= 0.4:
                return 'Churn Watch'           # Moderate risk, low engagement
            elif spent >= 1000:
                return 'Valued Low Risk'       # High spenders with low churn probability
            else:
                return 'Stable Low Risk'       # Low churn and low spend

        # Ensure numeric fields are present & clean
        seg_df = self.churn_df.copy()
        seg_df['total_spent'] = pd.to_numeric(seg_df.get('total_spent', 0), errors='coerce').fillna(0)
        seg_df['days_since_last_transaction'] = pd.to_numeric(
            seg_df.get('days_since_last_transaction', np.inf), errors='coerce'
        ).fillna(np.inf)
        seg_df['churn_probability'] = pd.to_numeric(seg_df.get('churn_probability', 0), errors='coerce').fillna(0)

        # Assign segments
        seg_df['segment'] = seg_df.apply(assign_segment, axis=1)

        # Count segments
        segment_counts = seg_df['segment'].value_counts().reindex([
            'Engaged High Risk',
            'At-Risk Low Engagement',
            'Loyal Medium Risk',
            'Churn Watch',
            'Valued Low Risk',
            'Stable Low Risk'
        ], fill_value=0)

        # Build pie chart
        fig = go.Figure(data=[go.Pie(
            labels=segment_counts.index,
            values=segment_counts.values,
            hole=0.4,
            marker_colors=[
                '#d62728',  # Engaged High Risk (red)
                '#ff7f0e',  # At-Risk Low Engagement (orange)
                '#bcbd22',  # Loyal Medium Risk (yellow-green)
                '#e377c2',  # Churn Watch (pink)
                '#2ca02c',  # Valued Low Risk (green)
                '#1f77b4'   # Stable Low Risk (blue)
            ],
            textinfo='label+percent',
            textposition='inside'
        )])

        # Add center annotation
        fig.update_layout(
            title="Customer Segmentation by Behavior & Churn Risk",
            annotations=[dict(
                text='Segments',
                x=0.5, y=0.5,
                font_size=14, showarrow=False
            )],
            height=400,
            margin=dict(l=20, r=20, t=60, b=20),
            showlegend=False
        )

        return fig
        
    def create_churn_density_plot(self):
        """Create density plot for customer churn prediction"""
        fig = go.Figure()
        
        # Filter out NaN values for 'churn_probability' before plotting
        churn_probabilities = self.churn_df['churn_probability'].dropna()

        if churn_probabilities.empty:
            fig.add_annotation(text="No data for Churn Probability",
                               xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(size=16, color="red"))
        else:
            # Create histogram that looks like density plot
            fig.add_trace(go.Histogram(
                x=churn_probabilities,
                nbinsx=25, # Number of bins can be adjusted
                histnorm='probability density',
                name='Churn Probability Distribution',
                marker_color='rgba(74, 134, 232, 0.7)',
                opacity=0.8
            ))
            
            # Add average line
            avg_churn = churn_probabilities.mean()
            fig.add_vline(
                x=avg_churn, 
                line_dash="dash", 
                line_color="red",
                line_width=2,
                annotation_text=f"Average: {avg_churn:.2f}",
                annotation_position="top"
            )
            
            # Add risk threshold lines
            fig.add_vline(
                x=0.8, 
                line_dash="dot", 
                line_color="orange",
                line_width=2,
                annotation_text="High Risk Threshold",
                annotation_position="bottom left"
            )
            
        fig.update_layout(
            title="Customer Churn Prediction - Density Distribution",
            xaxis_title="Churn Probability",
            yaxis_title="Probability Density",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=False
        )
        
        return fig
    
    def create_churn_ltv_scatter(self):
        """Create scatter plot for churn likelihood vs estimated LTV from customer_churn_predictions data"""
        # Use churn_df which is loaded from customer_churn_predictions.csv
        # Filter out NaN values for churn_probability and estimated_ltv only
        plot_df = self.churn_df.dropna(subset=['churn_probability', 'estimated_ltv']).copy()

        # Sample data for better visualization if dataset is large
        sample_df = plot_df.sample(n=min(500, len(plot_df)), random_state=42) if len(plot_df) > 500 else plot_df.copy()

        if sample_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data for Churn Likelihood vs LTV",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5, showarrow=False,
                            font=dict(size=16, color="red"))
        else:
            # Create simple scatter plot with just churn probability vs estimated LTV
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=sample_df['churn_probability'],
                y=sample_df['estimated_ltv'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=sample_df['churn_probability'],
                    colorscale='RdYlGn_r',  # Red-Yellow-Green reversed (red for high churn)
                    showscale=True,
                    colorbar=dict(title="Churn Probability"),
                    opacity=0.7
                ),
                text=[f"Customer ID: {cid}<br>Churn Prob: {prob:.2f}<br>Est. LTV: ${ltv:,.0f}" 
                    for cid, prob, ltv in zip(sample_df.get('customer_id', sample_df.index), 
                                            sample_df['churn_probability'], 
                                            sample_df['estimated_ltv'])],
                hovertemplate='<b>%{text}</b><extra></extra>',
                name='Customers'
            ))
            
            # Add trend line if desired
            from scipy import stats
            if len(sample_df) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    sample_df['churn_probability'], sample_df['estimated_ltv']
                )
                line_x = [sample_df['churn_probability'].min(), sample_df['churn_probability'].max()]
                line_y = [slope * x + intercept for x in line_x]
                
                fig.add_trace(go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode='lines',
                    line=dict(color='gray', dash='dash', width=2),
                    name=f'Trend (R²={r_value**2:.3f})',
                    showlegend=True
                ))

        fig.update_layout(
            title="Churn Likelihood vs Estimated LTV",
            xaxis_title="Churn Probability",
            yaxis_title="Estimated LTV ($)",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=True
        )

        return fig

    
    def create_top_customers_histogram(self):
        """Create histogram for at-risk customer demographics (Spending-Based Age Groups)"""
        # Focus on high-risk customers, ensuring 'spending_quartile' and 'risk_level' exist
        high_risk_customers = self.churn_df[self.churn_df['risk_level'] == 'High Risk'].dropna(subset=['spending_quartile'])
        
        # Create spending-based age groups
        age_group_counts = high_risk_customers['spending_quartile'].value_counts().sort_index() # Sort by index for ordered age groups
        
        if age_group_counts.empty:
            fig = go.Figure()
            fig.add_annotation(text="No High-Risk Customers Data",
                               xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(size=16, color="red"))
        else:
            fig = go.Figure(data=[
                go.Bar(
                    x=age_group_counts.index,
                    y=age_group_counts.values,
                    marker_color=['#ff9999', '#ffcc99', '#99ccff', '#cc99ff'], # Colors for quartiles
                    text=age_group_counts.values,
                    textposition='auto'
                )
            ])
            
        fig.update_layout(
            title="At-Risk Customers by Demographics (Spending-Based Age Groups)",
            xaxis_title="Age Groups (Based on Spending Patterns)",
            yaxis_title="Number of High-Risk Customers",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            showlegend=False
        )
        
        return fig
    
    def create_journey_funnel(self):
        """Create funnel chart for customer journey"""
        # Filter out NaN stages and 'No' conversion flags if they skew counts
        valid_journey_df = self.journey_df.dropna(subset=['stage']).copy()

        # Calculate stage progression
        stage_counts = valid_journey_df['stage'].value_counts()
        
        # Order stages logically and ensure all expected stages are present, even if count is 0
        stage_order = ['Lead', 'Awareness', 'Consideration', 'Purchase', 'Service', 'Loyalty']
        ordered_counts = [stage_counts.get(stage, 0) for stage in stage_order]
        
        # Filter out stages with 0 counts from the display if desired, or keep for full funnel visual
        # For a complete funnel, it's better to keep all stages even if count is 0
        
        if sum(ordered_counts) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No Customer Journey Data",
                               xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(size=16, color="red"))
        else:
            fig = go.Figure(go.Funnel(
                y=stage_order,
                x=ordered_counts,
                textposition="inside",
                textinfo="value+percent initial",
                opacity=0.8,
                marker={
                    "color": ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#ff99cc", "#c2c2f0"],
                    "line": {"width": 2, "color": "white"}
                }
            ))
            
        fig.update_layout(
            title="Customer Journey Funnel (Lead → Sale → Service → Loyalty)",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
    
    def create_early_dropoffs_chart(self):
        """Create stacked bar chart for early drop-offs"""
        # Filter out NaN stages and ensure conversion_flag is meaningful
        valid_journey_df = self.journey_df.dropna(subset=['stage', 'conversion_flag']).copy()

        # Calculate conversion and drop-off by stage
        # Use pivot_table to handle counts for 'Yes' and 'No' in 'conversion_flag'
        if not valid_journey_df.empty:
            # Ensure 'conversion_flag' has 'Yes'/'No' or equivalent values
            valid_journey_df['converted_flag'] = valid_journey_df['conversion_flag'].apply(lambda x: 1 if x == 'Yes' else 0)
            
            # Group by stage to get total customers and converted customers
            stage_summary = valid_journey_df.groupby('stage').agg(
                total_customers=('customer_id', 'count'),
                converted_customers=('converted_flag', 'sum') # Use the new numeric column
            ).reset_index()
            
            stage_summary['dropped_off'] = stage_summary['total_customers'] - stage_summary['converted_customers']
            
            # Define the complete set of categories for the 'stage' column
            stage_order = ['Lead', 'Awareness', 'Consideration', 'Purchase', 'Service', 'Loyalty']
            
            # Convert 'stage' to a Categorical type with ALL possible categories
            stage_summary['stage'] = pd.Categorical(stage_summary['stage'], categories=stage_order, ordered=True)
            
            # Reindex to ensure all stages are present, filling missing stages with 0 for counts
            # The fill_value here applies to the *numeric* columns, not the categorical 'stage' column directly.
            # We will fillna on the numeric columns *after* reindexing.
            stage_summary = stage_summary.set_index('stage').reindex(stage_order).reset_index()
            
            # Now, fill NaNs in numeric columns (converted_customers, dropped_off) with 0
            stage_summary[['converted_customers', 'dropped_off', 'total_customers']] = stage_summary[['converted_customers', 'dropped_off', 'total_customers']].fillna(0)
            
            # Ensure these columns are integer type after filling NaNs
            stage_summary['converted_customers'] = stage_summary['converted_customers'].astype(int)
            stage_summary['dropped_off'] = stage_summary['dropped_off'].astype(int)
            stage_summary['total_customers'] = stage_summary['total_customers'].astype(int)


        else:
            stage_summary = pd.DataFrame(columns=['stage', 'total_customers', 'converted_customers', 'dropped_off'])
        
        if stage_summary.empty or (stage_summary['converted_customers'].sum() == 0 and stage_summary['dropped_off'].sum() == 0):
            fig = go.Figure()
            fig.add_annotation(text="No Early Drop-offs Data",
                               xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(size=16, color="red"))
        else:
            fig = go.Figure()
            
            # Add converted customers
            fig.add_trace(go.Bar(
                name='Converted',
                x=stage_summary['stage'],
                y=stage_summary['converted_customers'],
                marker_color='#2ecc71',
                text=stage_summary['converted_customers'],
                textposition='inside'
            ))
            
            # Add dropped off customers
            fig.add_trace(go.Bar(
                name='Dropped Off',
                x=stage_summary['stage'],
                y=stage_summary['dropped_off'],
                marker_color='#e74c3c',
                text=stage_summary['dropped_off'],
                textposition='inside'
            ))
            
        fig.update_layout(
            title="Early Drop-offs by Customer Journey Stage",
            barmode='stack',
            height=400,
            xaxis_title="Journey Stage",
            yaxis_title="Number of Customers",
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig
    
    def create_social_media_trends_chart(self):
        """Create horizontal bar chart for social media most frequent hashtags"""
        # Extract hashtags from sentiment data, ensuring 'hashtags' column exists and is not empty
        all_hashtags = []
        if 'hashtags' in self.sentiment_df.columns:
            for hashtags_str in self.sentiment_df['hashtags'].dropna():
                # Ensure hashtags_str is a string to apply regex
                hashtags = re.findall(r'#\w+', str(hashtags_str))
                all_hashtags.extend(hashtags)
        
        # Count hashtag frequency and get top 8
        hashtag_counts = Counter(all_hashtags)
        top_hashtags = dict(hashtag_counts.most_common(8))
        
        if not top_hashtags: # Check if top_hashtags is empty
            fig = go.Figure()
            fig.add_annotation(text="No Hashtag Data Available",
                               xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(size=16, color="red"))
        else:
            # Sort hashtags by frequency for consistent display (highest at top)
            sorted_hashtags = sorted(top_hashtags.items(), key=lambda item: item[1], reverse=True)
            sorted_keys = [item[0] for item in sorted_hashtags]
            sorted_values = [item[1] for item in sorted_hashtags]

            fig = go.Figure(go.Bar(
                y=sorted_keys,
                x=sorted_values,
                orientation='h',
                marker_color='#9c27b0',
                text=sorted_values,
                textposition='auto'
            ))
            
            fig.update_layout(
                yaxis={'autorange': "reversed"} # This ensures the top value in the data is at the top of the bar chart
            )
            
        fig.update_layout(
            title="Social Media - Most Frequent Hashtags", # Updated title for accuracy
            xaxis_title="Frequency",
            yaxis_title="Hashtags",
            height=400,
            margin=dict(l=100, r=40, t=60, b=40)
        )
        
        return fig
    
    def generate_dashboard(self):
        """Generate the complete 3x3 HTML dashboard"""
        print("Creating visualizations...")
        
        # Create all visualizations
        gauge_fig = self.create_campaign_gauge_charts()
        time_channel_fig = self.create_best_time_channel_chart()
        segment_pie_fig = self.create_churn_segment_pie_chart()
        density_fig = self.create_churn_density_plot()
        scatter_fig = self.create_churn_ltv_scatter()
        histogram_fig = self.create_top_customers_histogram()
        funnel_fig = self.create_journey_funnel()
        dropoff_fig = self.create_early_dropoffs_chart()
        social_fig = self.create_social_media_trends_chart()
        
        # Convert to HTML
        charts_html = [
            gauge_fig.to_html(include_plotlyjs=False, div_id="chart1"),
            time_channel_fig.to_html(include_plotlyjs=False, div_id="chart2"),
            segment_pie_fig.to_html(include_plotlyjs=False, div_id="chart3"),
            density_fig.to_html(include_plotlyjs=False, div_id="chart4"),
            scatter_fig.to_html(include_plotlyjs=False, div_id="chart5"),
            histogram_fig.to_html(include_plotlyjs=False, div_id="chart6"),
            funnel_fig.to_html(include_plotlyjs=False, div_id="chart7"),
            dropoff_fig.to_html(include_plotlyjs=False, div_id="chart8"),
            social_fig.to_html(include_plotlyjs=False, div_id="chart9")
        ]
        
        # Create HTML dashboard
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Marketing Analytics Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .dashboard-header {{
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }}
        
        .dashboard-header h1 {{
            font-size: 2.5em;
            font-weight: 300;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            grid-template-rows: repeat(3, 1fr);
            gap: 20px;
            max-width: 1600px;
            margin: 0 auto;
            height: calc(100vh - 150px);
            min-height: 900px;
        }}
        
        .chart-container {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            overflow: hidden; /* Ensures content doesn't spill out */
        }}
        
        .chart-container:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
        }}
        
        .plotly-graph-div {{
            height: 100% !important;
            width: 100% !important;
        }}

        /* Responsive adjustments */
        @media (max-width: 1400px) {{
            .dashboard-grid {{
                grid-template-columns: 1fr;
                grid-template-rows: auto;
                height: auto; /* Allow height to adjust based on content */
            }}
            
            .chart-container {{
                height: 500px; /* Fixed height for smaller screens */
            }}
        }}

        /* Even smaller screens */
        @media (max-width: 768px) {{
            body {{
                padding: 10px;
            }}
            .dashboard-grid {{
                gap: 15px;
            }}
            .chart-container {{
                padding: 10px;
                height: 400px; /* Adjust height further if needed */
            }}
            .dashboard-header h1 {{
                font-size: 2em;
            }}
        }}

    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>🚀 Marketing Analytics Dashboard</h1>
        <p>Real-time insights into campaign performance, customer behavior, and churn prediction</p>
    </div>
    
    <div class="dashboard-grid">
        <div class="chart-container">{charts_html[0]}</div>
        <div class="chart-container">{charts_html[1]}</div>
        <div class="chart-container">{charts_html[2]}</div>
        <div class="chart-container">{charts_html[3]}</div>
        <div class="chart-container">{charts_html[4]}</div>
        <div class="chart-container">{charts_html[5]}</div>
        <div class="chart-container">{charts_html[6]}</div>
        <div class="chart-container">{charts_html[7]}</div>
        <div class="chart-container">{charts_html[8]}</div>
    </div>
    
    <script>
        // Ensure all charts resize properly
        window.addEventListener('resize', function() {{
            var charts = document.querySelectorAll('.plotly-graph-div');
            charts.forEach(function(chart) {{
                Plotly.Plots.resize(chart);
            }});
        }});
        
        // Auto-refresh data every 5 minutes (optional)
        // setInterval(function() {{
        //     location.reload();
        // }}, 300000);
    </script>
</body>
</html>
        """
        
        # Save dashboard
        output_path = f"{self.data_path}marketing_dashboard.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Dashboard generated successfully!")
        print(f"📁 File saved to: {output_path}")
        print(f"🌐 Open the HTML file in your browser to view the dashboard")
        
        return output_path

def main():
    """Main function to run the dashboard generator"""
    try:
        dashboard = MarketingDashboard()
        dashboard.generate_dashboard()
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()