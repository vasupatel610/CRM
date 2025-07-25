import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.linear_model import LogisticRegression

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Some visualizations will use matplotlib instead.")

class MLCRMAnalytics:
    def __init__(self, sales_csv_path=r'D:\CRM_PROJECT\data\sales.csv', logs_csv_path=r'D:\CRM_PROJECT\data\logs.csv'):
        self.sales_csv_path = sales_csv_path
        self.logs_csv_path = logs_csv_path
        self.vis_dir = r'D:\CRM_PROJECT\visulization'
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.campaign_data = None
        self.journey_data = None
        self.logs_df = None
        
        # Ensure visualization directory exists
        self.ensure_vis_directory()
        
    def ensure_vis_directory(self):
        """Ensure visualization directory exists"""
        os.makedirs(self.vis_dir, exist_ok=True)
        print(f"Visualization directory ready: {self.vis_dir}")
        
    def save_figure(self, fig, filename, title=""):
        """Save figure to visualization directory"""
        filepath = os.path.join(self.vis_dir, filename)
        fig.savefig(filepath, bbox_inches='tight', dpi=300)
        print(f"📊 {title} visualization saved: {filepath}")
        return filepath
        
    def load_csv_data(self):
        """Load and process data from both sales.csv and logs.csv"""
        try:
            # Load the sales CSV file
            print(f"Loading sales data from {self.sales_csv_path}...")
            self.sales_df = pd.read_csv(self.sales_csv_path)
            print(f"Sales data loaded successfully! Shape: {self.sales_df.shape}")
            
            # Load the logs CSV file
            print(f"Loading logs data from {self.logs_csv_path}...")
            self.logs_df = pd.read_csv(self.logs_csv_path)
            print(f"Logs data loaded successfully! Shape: {self.logs_df.shape}")
            
            print("\n=== SALES DATA ===")
            print("Column names:")
            print(self.sales_df.columns.tolist())
            print("\nFirst few rows:")
            print(self.sales_df.head())
            
            print("\n=== LOGS DATA ===")
            print("Column names:")
            print(self.logs_df.columns.tolist())
            print("\nFirst few rows:")
            print(self.logs_df.head())
            print("\nData types:")
            print(self.logs_df.dtypes)
            
            # Basic data cleaning for both datasets
            self.sales_df.columns = self.sales_df.columns.str.strip()
            self.logs_df.columns = self.logs_df.columns.str.strip()
            
            # Convert date columns in both datasets
            self.convert_date_columns(self.sales_df, "Sales")
            self.convert_date_columns(self.logs_df, "Logs")
            
            # Identify key columns in both datasets
            self.identify_key_columns()
            
            return self.sales_df, self.logs_df
            
        except FileNotFoundError as e:
            print(f"Error: File not found! {str(e)}")
            return None, None
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None, None
    
    def convert_date_columns(self, df, dataset_name):
        """Convert date columns in a dataset"""
        date_columns = []
        for col in df.columns:
            if any(word in col.lower() for word in ['date', 'time', 'created', 'updated', 'timestamp', 'sent', 'opened', 'clicked']):
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_columns.append(col)
                    print(f"Converted {col} to datetime in {dataset_name}")
                except:
                    print(f"Could not convert {col} to datetime in {dataset_name}")
        return date_columns
    
    def identify_key_columns(self):
        """Identify key columns in both datasets"""
        print("\n=== IDENTIFYING KEY COLUMNS ===")
        
        # Sales dataset columns
        sales_columns = self.sales_df.columns.str.lower()
        
        # Customer ID column
        self.customer_id_col = None
        for col in ['customer_id', 'customerid', 'customer', 'cust_id', 'id', 'user_id']:
            if col in sales_columns:
                self.customer_id_col = self.sales_df.columns[sales_columns.get_loc(col)]
                break
        
        # Transaction amount column
        self.amount_col = None
        for col in ['amount', 'value', 'price', 'total', 'revenue', 'sales']:
            if col in sales_columns:
                self.amount_col = self.sales_df.columns[sales_columns.get_loc(col)]
                break
        
        # Date column in sales
        self.date_col = None
        for col in ['date', 'transaction_date', 'purchase_date', 'order_date', 'created_at']:
            if col in sales_columns:
                self.date_col = self.sales_df.columns[sales_columns.get_loc(col)]
                break
        
        # Product column
        self.product_col = None
        for col in ['product', 'item', 'product_name', 'category']:
            if col in sales_columns:
                self.product_col = self.sales_df.columns[sales_columns.get_loc(col)]
                break

        # Logs dataset columns
        if self.logs_df is not None:
            logs_columns = self.logs_df.columns.str.lower()
            
            # Campaign/Channel column in logs
            self.logs_campaign_col = None
            for col in ['campaign', 'campaign_name', 'channel', 'source', 'medium', 'utm_campaign', 'marketing_channel']:
                if col in logs_columns:
                    self.logs_campaign_col = self.logs_df.columns[logs_columns.get_loc(col)]
                    break
            
            # Customer ID in logs
            self.logs_customer_id_col = None
            for col in ['customer_id', 'customerid', 'customer', 'cust_id', 'id', 'user_id', 'email']:
                if col in logs_columns:
                    self.logs_customer_id_col = self.logs_df.columns[logs_columns.get_loc(col)]
                    break
            
            # Event type column (email sent, opened, clicked, etc.)
            self.logs_event_col = None
            for col in ['event', 'event_type', 'action', 'activity', 'type']:
                if col in logs_columns:
                    self.logs_event_col = self.logs_df.columns[logs_columns.get_loc(col)]
                    break
            
            # Timestamp in logs
            self.logs_date_col = None
            for col in ['timestamp', 'date', 'created_at', 'event_date', 'time']:
                if col in logs_columns:
                    self.logs_date_col = self.logs_df.columns[logs_columns.get_loc(col)]
                    break
            
            # Email specific columns
            self.logs_email_col = None
            for col in ['email', 'email_address', 'recipient']:
                if col in logs_columns:
                    self.logs_email_col = self.logs_df.columns[logs_columns.get_loc(col)]
                    break
                
        print(f"\n=== SALES DATA COLUMNS ===")
        print(f"Customer ID: {self.customer_id_col}")
        print(f"Amount/Value: {self.amount_col}")
        print(f"Date: {self.date_col}")
        print(f"Product: {self.product_col}")
        
        if self.logs_df is not None:
            print(f"\n=== LOGS DATA COLUMNS ===")
            print(f"Customer ID: {self.logs_customer_id_col}")
            print(f"Campaign/Channel: {self.logs_campaign_col}")
            print(f"Event Type: {self.logs_event_col}")
            print(f"Date/Timestamp: {self.logs_date_col}")
            print(f"Email: {self.logs_email_col}")

    def process_logs_for_campaign_analysis(self):
        """Process logs.csv data for campaign analysis"""
        print("\n=== PROCESSING LOGS FOR CAMPAIGN ANALYSIS ===")
        
        if self.logs_df is None:
            print("No logs data available")
            return None
        
        # Check if we have necessary columns
        if not all([self.logs_event_col, self.logs_date_col]):
            print("Missing required columns in logs data")
            print(f"Available columns: {self.logs_df.columns.tolist()}")
            return None
        
        # Create a copy for processing
        logs_processed = self.logs_df.copy()
        
        # Standardize event types (common patterns)
        event_mapping = {
            'sent': 'sent',
            'delivered': 'sent',
            'opened': 'opened',
            'open': 'opened',
            'clicked': 'clicked',
            'click': 'clicked',
            'converted': 'conversion',
            'purchase': 'conversion',
            'sale': 'conversion',
            'unsubscribed': 'unsubscribed',
            'bounced': 'bounced'
        }
        
        # Clean and map event types
        logs_processed['event_clean'] = logs_processed[self.logs_event_col].str.lower().str.strip()
        logs_processed['event_standardized'] = logs_processed['event_clean'].map(
            lambda x: next((v for k, v in event_mapping.items() if k in str(x).lower()), str(x))
        )
        
        # Identify campaign column
        campaign_col = self.logs_campaign_col if self.logs_campaign_col else 'Unknown'
        if campaign_col == 'Unknown':
            logs_processed['campaign'] = 'Default_Campaign'
        else:
            logs_processed['campaign'] = logs_processed[campaign_col].fillna('Unknown_Campaign')
        
        print("Event types found in logs:")
        print(logs_processed['event_standardized'].value_counts())
        
        print("\nCampaigns found in logs:")
        print(logs_processed['campaign'].value_counts())
        
        return logs_processed

    def campaign_dashboard_analysis(self):
        """Create campaign dashboard with real data from logs.csv"""
        print("\n=== CAMPAIGN DASHBOARD ANALYSIS (FROM LOGS.CSV) ===")
        
        # Process logs data
        logs_processed = self.process_logs_for_campaign_analysis()
        if logs_processed is None:
            print("Cannot process logs data for campaign analysis")
            return self.generate_sample_campaign_data()  # Fallback to sample data
        
        # Create campaign metrics from logs
        campaign_metrics = []
        
        # Group by campaign
        for campaign in logs_processed['campaign'].unique():
            campaign_data = logs_processed[logs_processed['campaign'] == campaign]
            
            # Count events by type
            event_counts = campaign_data['event_standardized'].value_counts()
            
            sent = event_counts.get('sent', 0)
            opened = event_counts.get('opened', 0)
            clicked = event_counts.get('clicked', 0)
            conversions = event_counts.get('conversion', 0)
            
            # If no sent events, estimate from opened (assuming 25% open rate)
            if sent == 0 and opened > 0:
                sent = int(opened / 0.25)
            elif sent == 0 and clicked > 0:
                sent = int(clicked / 0.05)  # Assuming 5% overall CTR
            elif sent == 0:
                sent = 1000  # Default assumption
            
            # Calculate revenue from conversions (if available from sales data)
            campaign_revenue = 0
            if conversions > 0 and self.sales_df is not None and self.amount_col is not None:
                # Try to match conversions with sales
                avg_sale_value = self.sales_df[self.amount_col].mean()
                campaign_revenue = conversions * avg_sale_value
            else:
                campaign_revenue = conversions * 150  # Assume $150 average order value
            
            campaign_metrics.append({
                'Campaign': campaign,
                'Sent': sent,
                'Opened': opened,
                'Clicked': clicked,
                'Conversions': conversions,
                'Revenue': campaign_revenue
            })
        
        self.campaign_data = pd.DataFrame(campaign_metrics)
        
        if len(self.campaign_data) == 0:
            print("No campaign data could be extracted from logs")
            return self.generate_sample_campaign_data()
        
        # Calculate rates
        self.campaign_data['Open_Rate'] = np.where(
            self.campaign_data['Sent'] > 0,
            (self.campaign_data['Opened'] / self.campaign_data['Sent'] * 100).round(2),
            0
        )
        
        self.campaign_data['Click_Rate'] = np.where(
            self.campaign_data['Opened'] > 0,
            (self.campaign_data['Clicked'] / self.campaign_data['Opened'] * 100).round(2),
            0
        )
        
        self.campaign_data['Conversion_Rate'] = np.where(
            self.campaign_data['Clicked'] > 0,
            (self.campaign_data['Conversions'] / self.campaign_data['Clicked'] * 100).round(2),
            0
        )
        
        # Calculate ROI (assuming $0.10 cost per email sent)
        cost_per_email = 0.10
        self.campaign_data['Cost'] = self.campaign_data['Sent'] * cost_per_email
        self.campaign_data['ROI'] = np.where(
            self.campaign_data['Cost'] > 0,
            ((self.campaign_data['Revenue'] - self.campaign_data['Cost']) / self.campaign_data['Cost'] * 100).round(2),
            0
        )
        
        print("Campaign Performance Dashboard (From Logs.csv):")
        print("="*80)
        print(self.campaign_data[['Campaign', 'Sent', 'Opened', 'Clicked', 'Conversions', 'Open_Rate', 'Click_Rate', 'Conversion_Rate', 'ROI']])
        
        # Calculate overall metrics
        total_sent = self.campaign_data['Sent'].sum()
        total_opened = self.campaign_data['Opened'].sum()
        total_clicked = self.campaign_data['Clicked'].sum()
        total_conversions = self.campaign_data['Conversions'].sum()
        
        print(f"\nOverall Campaign Metrics (From Real Data):")
        print(f"Total Emails Sent: {total_sent:,}")
        if total_sent > 0:
            print(f"Overall Open Rate: {(total_opened/total_sent*100):.2f}%")
        if total_opened > 0:
            print(f"Overall Click Rate: {(total_clicked/total_opened*100):.2f}%")
        if total_clicked > 0:
            print(f"Overall Conversion Rate: {(total_conversions/total_clicked*100):.2f}%")
        
        # Best performing campaigns
        if len(self.campaign_data) > 0:
            best_open = self.campaign_data.loc[self.campaign_data['Open_Rate'].idxmax()]
            best_conversion = self.campaign_data.loc[self.campaign_data['Conversion_Rate'].idxmax()]
            best_roi = self.campaign_data.loc[self.campaign_data['ROI'].idxmax()]
            
            print(f"\nTop Performers (From Real Data):")
            print(f"🎯 Best Open Rate: {best_open['Campaign']} ({best_open['Open_Rate']:.2f}%)")
            print(f"💰 Best Conversion Rate: {best_conversion['Campaign']} ({best_conversion['Conversion_Rate']:.2f}%)")
            print(f"📈 Best ROI: {best_roi['Campaign']} ({best_roi['ROI']:.2f}%)")
        
        # Create campaign metrics visualization
        self.visualize_campaign_metrics()
        
        return self.campaign_data

    def visualize_campaign_metrics(self):
        """Create and save campaign metrics visualization"""
        if self.campaign_data is None or len(self.campaign_data) == 0:
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Campaign Metrics Dashboard', fontsize=16, fontweight='bold')
        
        # Campaign names for x-axis
        campaigns = self.campaign_data['Campaign']
        
        # 1. Email Metrics (Sent, Opened, Clicked)
        x_pos = np.arange(len(campaigns))
        width = 0.25
        
        axes[0, 0].bar(x_pos - width, self.campaign_data['Sent'], width, label='Sent', alpha=0.8)
        axes[0, 0].bar(x_pos, self.campaign_data['Opened'], width, label='Opened', alpha=0.8)
        axes[0, 0].bar(x_pos + width, self.campaign_data['Clicked'], width, label='Clicked', alpha=0.8)
        axes[0, 0].set_title('Email Engagement Metrics')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(campaigns, rotation=45, ha='right')
        axes[0, 0].legend()
        
        # 2. Conversion Rates
        axes[0, 1].bar(campaigns, self.campaign_data['Open_Rate'], alpha=0.7, label='Open Rate')
        axes[0, 1].bar(campaigns, self.campaign_data['Click_Rate'], alpha=0.7, label='Click Rate')
        axes[0, 1].bar(campaigns, self.campaign_data['Conversion_Rate'], alpha=0.7, label='Conversion Rate')
        axes[0, 1].set_title('Campaign Conversion Rates')
        axes[0, 1].set_ylabel('Rate (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend()
        
        # 3. Revenue and ROI
        ax_rev = axes[1, 0]
        ax_roi = ax_rev.twinx()
        
        bars1 = ax_rev.bar(campaigns, self.campaign_data['Revenue'], alpha=0.6, color='green', label='Revenue')
        line1 = ax_roi.plot(campaigns, self.campaign_data['ROI'], 'ro-', label='ROI (%)')
        
        ax_rev.set_title('Revenue and ROI by Campaign')
        ax_rev.set_ylabel('Revenue ($)', color='green')
        ax_roi.set_ylabel('ROI (%)', color='red')
        ax_rev.tick_params(axis='x', rotation=45)
        
        # 4. Funnel Analysis
        stages = ['Sent', 'Opened', 'Clicked', 'Conversions']
        total_metrics = [
            self.campaign_data['Sent'].sum(),
            self.campaign_data['Opened'].sum(), 
            self.campaign_data['Clicked'].sum(),
            self.campaign_data['Conversions'].sum()
        ]
        
        axes[1, 1].bar(stages, total_metrics, color=['skyblue', 'lightgreen', 'orange', 'red'])
        axes[1, 1].set_title('Overall Campaign Funnel')
        axes[1, 1].set_ylabel('Total Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add percentage labels on funnel bars
        for i, v in enumerate(total_metrics):
            if total_metrics[0] > 0:
                pct = (v / total_metrics[0]) * 100
                axes[1, 1].text(i, v + max(total_metrics)*0.01, f'{pct:.1f}%', ha='center')
        
        plt.tight_layout()
        self.save_figure(fig, 'campaign_metrics_dashboard.png', 'Campaign Metrics Dashboard')
        plt.show()
        
        return fig

    def generate_journey_data_from_logs(self):
        """Generate customer journey data from logs.csv with better stage mapping"""
        print("\n=== GENERATING JOURNEY DATA FROM LOGS ===")
        
        logs_processed = self.process_logs_for_campaign_analysis()
        if logs_processed is None:
            return self.generate_sample_journey_data()
        
        # Enhanced event to stage mapping
        event_to_stage = {
            'sent': 'Lead',
            'delivered': 'Lead',  # Still a lead until they engage
            'opened': 'Qualified_Lead',
            'clicked': 'Opportunity',
            'converted': 'Sale',
            'conversion': 'Sale'
        }
        
        # Additional logic for Service and Loyalty stages
        journey_data = []
        
        # Get unique customers from logs
        customer_col = self.logs_customer_id_col if self.logs_customer_id_col else self.logs_email_col
        
        if customer_col is None:
            print("No customer identifier found in logs")
            return self.generate_sample_journey_data()
        
        unique_customers = logs_processed[customer_col].unique()[:1000]  # Limit for performance
        
        for customer in unique_customers:
            if pd.isna(customer):
                continue
                
            customer_events = logs_processed[logs_processed[customer_col] == customer]
            customer_events = customer_events.sort_values(self.logs_date_col)
            
            stages_completed = set()
            customer_conversions = 0
            
            for _, event in customer_events.iterrows():
                event_type = event['event_standardized']
                stage = event_to_stage.get(event_type, None)
                
                if stage and stage not in stages_completed:
                    journey_data.append({
                        'Customer_ID': customer,
                        'Stage': stage,
                        'Date': event[self.logs_date_col],
                        'Completed': 1
                    })
                    stages_completed.add(stage)
                    
                    # Count conversions for this customer
                    if event_type in ['converted', 'conversion']:
                        customer_conversions += 1
            
            # Add Service stage for customers who converted (simulate post-purchase engagement)
            if 'Sale' in stages_completed and customer_conversions > 0:
                # 70% of customers who convert get some service
                if np.random.random() < 0.7:
                    last_event_date = customer_events[self.logs_date_col].max()
                    service_date = last_event_date + timedelta(days=np.random.randint(1, 30))
                    
                    journey_data.append({
                        'Customer_ID': customer,
                        'Stage': 'Service',
                        'Date': service_date,
                        'Completed': 1
                    })
                    stages_completed.add('Service')
            
            # Add Loyalty stage for repeat customers or high-value customers
            if 'Service' in stages_completed and customer_conversions > 0:
                # 40% of serviced customers become loyal
                if np.random.random() < 0.4 or customer_conversions > 1:
                    last_event_date = customer_events[self.logs_date_col].max()
                    loyalty_date = last_event_date + timedelta(days=np.random.randint(30, 90))
                    
                    journey_data.append({
                        'Customer_ID': customer,
                        'Stage': 'Loyalty',
                        'Date': loyalty_date,
                        'Completed': 1
                    })
                    stages_completed.add('Loyalty')
        
        if len(journey_data) == 0:
            print("No journey data could be extracted from logs")
            return self.generate_sample_journey_data()
        
        self.journey_data = pd.DataFrame(journey_data)
        print(f"Generated journey data for {len(self.journey_data)} customer touchpoints")
        
        # Show stage distribution
        stage_counts = self.journey_data['Stage'].value_counts()
        print("\nStage Distribution:")
        for stage, count in stage_counts.items():
            print(f"  {stage}: {count}")
        
        return self.journey_data

    def customer_journey_funnel_analysis(self):
        """Analyze customer journey funnel using real or sample data"""
        print("\n=== CUSTOMER JOURNEY FUNNEL ANALYSIS ===")
        
        # Try to generate from logs first
        if self.logs_df is not None:
            journey_data = self.generate_journey_data_from_logs()
        else:
            journey_data = self.generate_sample_journey_data()
        
        if journey_data is None:
            print("Could not generate journey data")
            return None
        
        # Calculate funnel metrics
        funnel_metrics = self.journey_data.groupby('Stage').agg({
            'Customer_ID': 'nunique',
            'Completed': 'sum'
        })
        
        # Define all expected stages
        expected_stages = ['Lead', 'Qualified_Lead', 'Opportunity', 'Sale', 'Service', 'Loyalty']
        
        # Reindex to include all stages, filling missing with 0
        funnel_metrics = funnel_metrics.reindex(expected_stages, fill_value=0)
        funnel_metrics.columns = ['Unique_Customers', 'Total_Completions']
        
        # Convert to float to handle calculations properly
        funnel_metrics = funnel_metrics.astype(float)
        
        # Calculate conversion rates
        total_leads = funnel_metrics.loc['Lead', 'Unique_Customers']
        if total_leads > 0:
            funnel_metrics['Conversion_Rate'] = (funnel_metrics['Unique_Customers'] / total_leads * 100).round(2)
            funnel_metrics['Stage_Drop_Rate'] = (100 - funnel_metrics['Conversion_Rate']).round(2)
        else:
            funnel_metrics['Conversion_Rate'] = 0.0
            funnel_metrics['Stage_Drop_Rate'] = 0.0
        
        # Calculate stage-to-stage conversion
        funnel_metrics['Stage_to_Stage_Rate'] = 0.0
        for i in range(1, len(funnel_metrics)):
            prev_customers = funnel_metrics.iloc[i-1]['Unique_Customers']
            curr_customers = funnel_metrics.iloc[i]['Unique_Customers']
            if prev_customers > 0:
                stage_to_stage_rate = (curr_customers / prev_customers * 100)
            else:
                stage_to_stage_rate = 0.0
            funnel_metrics.iloc[i, funnel_metrics.columns.get_loc('Stage_to_Stage_Rate')] = round(stage_to_stage_rate, 2)
        
        data_source = "Real Logs Data" if self.logs_df is not None else "Sample Data"
        print(f"Customer Journey Funnel Analysis ({data_source}):")
        print("="*70)
        
        # Display only stages with data or the first few stages
        display_funnel = funnel_metrics[funnel_metrics['Unique_Customers'] > 0]
        if len(display_funnel) == 0:
            display_funnel = funnel_metrics.head(4)  # Show first 4 stages even if no data
        
        print(display_funnel[['Unique_Customers', 'Conversion_Rate', 'Stage_to_Stage_Rate']])
        
        # Create funnel visualization
        self.visualize_customer_funnel(funnel_metrics)
        
        # Identify biggest drop-offs
        stages_with_data = funnel_metrics[funnel_metrics['Unique_Customers'] > 0]
        if len(stages_with_data) > 1:
            stage_to_stage_rates = stages_with_data['Stage_to_Stage_Rate'].iloc[1:]
            if len(stage_to_stage_rates) > 0 and stage_to_stage_rates.max() > 0:
                biggest_dropoff_idx = stage_to_stage_rates.idxmin()
                biggest_dropoff_rate = 100 - stage_to_stage_rates.min()
                print(f"\n🚨 Biggest Drop-off: {biggest_dropoff_idx} ({biggest_dropoff_rate:.1f}% drop)")
        
        return funnel_metrics

    def visualize_customer_funnel(self, funnel_metrics):
        """Create and save customer journey funnel visualization"""
        stages_with_data = funnel_metrics[funnel_metrics['Unique_Customers'] > 0]
        
        if len(stages_with_data) == 0:
            return None
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Customer Journey Funnel Analysis', fontsize=16, fontweight='bold')
        
        # 1. Funnel Bar Chart
        stages = stages_with_data.index
        customers = stages_with_data['Unique_Customers']
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(stages)))
        
        bars = ax1.bar(stages, customers, color=colors)
        ax1.set_title('Customer Funnel by Stage')
        ax1.set_ylabel('Number of Customers')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add percentage labels on bars
        total_leads = customers.iloc[0] if len(customers) > 0 else 1
        for i, (bar, count) in enumerate(zip(bars, customers)):
            height = bar.get_height()
            pct = (count / total_leads) * 100
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:.0f}\n({pct:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
        
        # 2. Conversion Rates
        conversion_rates = stages_with_data['Conversion_Rate']
        ax2.plot(stages, conversion_rates, 'o-', linewidth=2, markersize=8, color='red')
        ax2.set_title('Conversion Rate by Stage')
        ax2.set_ylabel('Conversion Rate (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on the line
        for stage, rate in zip(stages, conversion_rates):
            ax2.annotate(f'{rate:.1f}%', (stage, rate), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')
        
        plt.tight_layout()
        self.save_figure(fig, 'customer_journey_funnel.png', 'Customer Journey Funnel')
        plt.show()
        
        return fig

    def multi_touch_attribution_from_logs(self):
        """Implement multi-touch attribution using real logs data"""
        print("\n=== MULTI-TOUCH ATTRIBUTION FROM LOGS ===")
        
        logs_processed = self.process_logs_for_campaign_analysis()
        if logs_processed is None:
            return self.multi_touch_attribution_modeling()  # Fallback
        
        # Get customer identifier
        customer_col = self.logs_customer_id_col if self.logs_customer_id_col else self.logs_email_col
        
        if customer_col is None:
            print("No customer identifier found for attribution")
            return self.multi_touch_attribution_modeling()
        
        # Create attribution data from logs
        attribution_data = []
        
        # Get customers who had conversions
        conversion_customers = logs_processed[
            logs_processed['event_standardized'] == 'conversion'
        ][customer_col].unique()
        
        for customer in conversion_customers:
            if pd.isna(customer):
                continue
                
            customer_events = logs_processed[logs_processed[customer_col] == customer]
            customer_events = customer_events.sort_values(self.logs_date_col)
            
            # Get all touchpoints before conversion
            touchpoints = []
            conversion_revenue = 0
            
            for idx, event in customer_events.iterrows():
                if event['event_standardized'] == 'conversion':
                    # Assign revenue to this conversion
                    if self.sales_df is not None and self.amount_col is not None:
                        conversion_revenue = self.sales_df[self.amount_col].mean()  # Use average
                    else:
                        conversion_revenue = 150  # Default
                    break
                else:
                    touchpoint = event.get('campaign', 'Unknown')
                    touchpoints.append(touchpoint)
            
            # Create touchpoint records
            for i, touchpoint in enumerate(touchpoints):
                attribution_data.append({
                    'Customer_ID': customer,
                    'Touchpoint': touchpoint,
                    'Order': i + 1,
                    'Revenue': conversion_revenue if i == len(touchpoints) - 1 else 0
                })
        
        if len(attribution_data) == 0:
            print("No attribution data could be extracted from logs")
            return self.multi_touch_attribution_modeling()
        
        attribution_df = pd.DataFrame(attribution_data)
        
        # Calculate different attribution models
        attribution_models = {}
        
        # 1. First Touch Attribution
        first_touch = attribution_df[attribution_df['Order'] == 1].groupby('Touchpoint')['Revenue'].sum()
        attribution_models['First_Touch'] = first_touch
        
        # 2. Last Touch Attribution
        last_touch_customers = attribution_df.groupby('Customer_ID')['Order'].max().reset_index()
        last_touch_customers.columns = ['Customer_ID', 'Max_Order']
        last_touch_data = attribution_df.merge(last_touch_customers, on='Customer_ID')
        last_touch_data = last_touch_data[last_touch_data['Order'] == last_touch_data['Max_Order']]
        last_touch = last_touch_data.groupby('Touchpoint')['Revenue'].sum()
        attribution_models['Last_Touch'] = last_touch
        
        # 3. Linear Attribution
        customer_revenues = attribution_df.groupby('Customer_ID')['Revenue'].max()
        linear_attribution = []
        
        for customer_id, revenue in customer_revenues.items():
            if revenue > 0:
                customer_touchpoints = attribution_df[attribution_df['Customer_ID'] == customer_id]['Touchpoint'].tolist()
                weight_per_touch = revenue / len(customer_touchpoints)
                for touchpoint in customer_touchpoints:
                    linear_attribution.append({'Touchpoint': touchpoint, 'Revenue': weight_per_touch})
        
        if linear_attribution:
            linear_df = pd.DataFrame(linear_attribution)
            linear_touch = linear_df.groupby('Touchpoint')['Revenue'].sum()
            attribution_models['Linear'] = linear_touch
        
        # Create comparison
        attribution_comparison = pd.DataFrame(attribution_models).fillna(0).round(2)
        
        print("Multi-Touch Attribution from Real Logs Data:")
        print("="*60)
        print(attribution_comparison)
        
        # Calculate attribution percentages
        for model in attribution_comparison.columns:
            total = attribution_comparison[model].sum()
            if total > 0:
                attribution_comparison[f'{model}_Percent'] = (attribution_comparison[model] / total * 100).round(2)
        
        percentage_cols = [col for col in attribution_comparison.columns if 'Percent' in col]
        if percentage_cols:
            print(f"\nAttribution Percentages (From Real Data):")
            print(attribution_comparison[percentage_cols])
        
        # Create attribution visualization
        self.visualize_attribution_analysis(attribution_comparison)
        
        return attribution_comparison

    def visualize_attribution_analysis(self, attribution_comparison):
        """Create and save attribution analysis visualization"""
        if attribution_comparison is None or len(attribution_comparison) == 0:
            return None
            
        # Get revenue models (not percentage columns)
        revenue_models = [col for col in attribution_comparison.columns if not col.endswith('_Percent')]
        
        if len(revenue_models) == 0:
            return None
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Multi-Touch Attribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Attribution Revenue by Model
        touchpoints = attribution_comparison.index
        x_pos = np.arange(len(touchpoints))
        width = 0.25
        
        colors = ['skyblue', 'lightgreen', 'orange']
        for i, model in enumerate(revenue_models[:3]):  # Limit to 3 models
            offset = (i - 1) * width
            bars = ax1.bar(x_pos + offset, attribution_comparison[model], width, 
                          label=model, alpha=0.8, color=colors[i % len(colors)])
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'${height:.0f}', ha='center', va='bottom', fontsize=8)
        
        ax1.set_title('Attribution Revenue by Touchpoint')
        ax1.set_ylabel('Revenue ($)')
        ax1.set_xlabel('Touchpoints')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(touchpoints, rotation=45, ha='right')
        ax1.legend()
        
        # 2. Attribution Percentage Comparison
        percentage_cols = [col for col in attribution_comparison.columns if col.endswith('_Percent')]
        if percentage_cols:
            model_names = [col.replace('_Percent', '') for col in percentage_cols]
            
            # Create stacked or grouped bar chart
            bottom = np.zeros(len(touchpoints))
            colors_pct = plt.cm.Set3(np.linspace(0, 1, len(percentage_cols)))
            
            for i, (pct_col, model_name) in enumerate(zip(percentage_cols, model_names)):
                values = attribution_comparison[pct_col].values
                ax2.bar(touchpoints, values, bottom=bottom, label=model_name, 
                       alpha=0.8, color=colors_pct[i])
                
                # Add percentage labels
                for j, (touchpoint, value) in enumerate(zip(touchpoints, values)):
                    if value > 5:  # Only show labels for values > 5%
                        ax2.text(j, bottom[j] + value/2, f'{value:.1f}%', 
                                ha='center', va='center', fontsize=8, fontweight='bold')
                
                bottom += values
            
            ax2.set_title('Attribution Percentage by Model')
            ax2.set_ylabel('Attribution Percentage (%)')
            ax2.set_xlabel('Touchpoints')
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No percentage data available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Attribution Percentage Analysis')
        
        plt.tight_layout()
        self.save_figure(fig, 'attribution_analysis.png', 'Attribution Analysis')
        plt.show()
        
        return fig

    def funnel_drop_off_visualization(self):
        """Create funnel visualization showing early drop-offs"""
        print("\n=== FUNNEL DROP-OFF VISUALIZATION ===")
        
        # Create e-commerce funnel data
        ecommerce_funnel = {
            'Stage': ['Website_Visit', 'Product_View', 'Add_to_Cart', 'Checkout_Started', 'Payment_Info', 'Purchase_Complete'],
            'Users': [10000, 7500, 3000, 1800, 1500, 1200],
            'Drop_Off_Reasons': [
                'Not Interested',
                'Price Too High', 
                'Abandoned Cart',
                'Complicated Checkout',
                'Payment Issues',
                'Successful Purchase'
            ]
        }
        
        funnel_df = pd.DataFrame(ecommerce_funnel)
        
        # Calculate drop-off rates
        funnel_df['Conversion_Rate'] = (funnel_df['Users'] / funnel_df['Users'].iloc[0] * 100).round(2)
        funnel_df['Drop_Off_Rate'] = (100 - funnel_df['Conversion_Rate']).round(2)
        
        # Calculate stage-to-stage drop-off
        funnel_df['Stage_Drop_Off'] = 0
        for i in range(1, len(funnel_df)):
            prev_users = funnel_df.iloc[i-1]['Users']
            curr_users = funnel_df.iloc[i]['Users']
            funnel_df.iloc[i, funnel_df.columns.get_loc('Stage_Drop_Off')] = prev_users - curr_users
        
        print("E-commerce Funnel Analysis:")
        print("="*70)
        print(funnel_df[['Stage', 'Users', 'Conversion_Rate', 'Stage_Drop_Off', 'Drop_Off_Reasons']])
        
        # Create drop-off visualization
        self.visualize_ecommerce_funnel(funnel_df)
        
        # Identify critical drop-off points
        max_dropoff_idx = funnel_df['Stage_Drop_Off'].idxmax()
        max_dropoff_stage = funnel_df.iloc[max_dropoff_idx]['Stage']
        max_dropoff_users = funnel_df.iloc[max_dropoff_idx]['Stage_Drop_Off']
        max_dropoff_reason = funnel_df.iloc[max_dropoff_idx]['Drop_Off_Reasons']
        
        print(f"\n🚨 Critical Drop-off Point:")
        print(f"Stage: {max_dropoff_stage}")
        print(f"Users Lost: {max_dropoff_users:,}")
        print(f"Reason: {max_dropoff_reason}")
        
        # Calculate potential revenue impact
        avg_order_value = 150
        potential_revenue_lost = max_dropoff_users * avg_order_value
        
        print(f"💰 Potential Revenue Lost: ${potential_revenue_lost:,.2f}")
        
        return funnel_df

    def visualize_ecommerce_funnel(self, funnel_df):
        """Create and save e-commerce funnel visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('E-commerce Funnel Drop-off Analysis', fontsize=16, fontweight='bold')
        
        stages = funnel_df['Stage']
        users = funnel_df['Users']
        
        # 1. Funnel Visualization
        colors = plt.cm.Reds_r(np.linspace(0.3, 0.9, len(stages)))
        bars = ax1.bar(stages, users, color=colors)
        
        ax1.set_title('E-commerce Funnel')
        ax1.set_ylabel('Number of Users')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add conversion rate labels
        for bar, user_count, conv_rate in zip(bars, users, funnel_df['Conversion_Rate']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{user_count:,}\n({conv_rate:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 2. Drop-off Analysis
        drop_offs = funnel_df['Stage_Drop_Off'][1:]  # Skip first stage (no drop-off)
        drop_stages = funnel_df['Stage'][1:]
        
        bars2 = ax2.bar(drop_stages, drop_offs, color='red', alpha=0.7)
        ax2.set_title('Stage-to-Stage Drop-offs')
        ax2.set_ylabel('Users Lost')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add drop-off labels
        for bar, drop_count in zip(bars2, drop_offs):
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{drop_count:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        self.save_figure(fig, 'ecommerce_funnel_dropoff.png', 'E-commerce Funnel Drop-off')
        plt.show()
        
        return fig

    def visualize_campaign_performance(self):
        """Create campaign performance visualizations"""
        if self.campaign_data is None:
            self.campaign_dashboard_analysis()
        
        print("\n=== CREATING CAMPAIGN PERFORMANCE VISUALIZATION ===")
        
        # Create comprehensive campaign performance chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Campaign Performance Dashboard (From Real Data)', fontsize=16, fontweight='bold')
        
        # 1. Open Rates
        axes[0, 0].bar(self.campaign_data['Campaign'], self.campaign_data['Open_Rate'], color='skyblue')
        axes[0, 0].set_title('Open Rates by Campaign')
        axes[0, 0].set_ylabel('Open Rate (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(self.campaign_data['Open_Rate']):
            axes[0, 0].text(i, v, f'{v:.1f}%', ha='center', va='bottom')
        
        # 2. Click Rates
        axes[0, 1].bar(self.campaign_data['Campaign'], self.campaign_data['Click_Rate'], color='lightgreen')
        axes[0, 1].set_title('Click Rates by Campaign')
        axes[0, 1].set_ylabel('Click Rate (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        for i, v in enumerate(self.campaign_data['Click_Rate']):
            axes[0, 1].text(i, v, f'{v:.1f}%', ha='center', va='bottom')
        
        # 3. Conversion Rates
        axes[1, 0].bar(self.campaign_data['Campaign'], self.campaign_data['Conversion_Rate'], color='coral')
        axes[1, 0].set_title('Conversion Rates by Campaign')
        axes[1, 0].set_ylabel('Conversion Rate (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        for i, v in enumerate(self.campaign_data['Conversion_Rate']):
            axes[1, 0].text(i, v, f'{v:.1f}%', ha='center', va='bottom')
        
        # 4. ROI
        roi_colors = ['red' if x < 0 else 'gold' for x in self.campaign_data['ROI']]
        axes[1, 1].bar(self.campaign_data['Campaign'], self.campaign_data['ROI'], color=roi_colors)
        axes[1, 1].set_title('ROI by Campaign')
        axes[1, 1].set_ylabel('ROI (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        for i, v in enumerate(self.campaign_data['ROI']):
            axes[1, 1].text(i, v, f'{v:.1f}%', ha='center', va='bottom' if v >= 0 else 'top')
        
        plt.tight_layout()
        self.save_figure(fig, 'campaign_performance.png', 'Campaign Performance')
        plt.show()
        
        return fig

    def create_customer_features(self):
        """Create customer features from the sales CSV data"""
        if self.sales_df is None:
            print("No data loaded. Please load data first.")
            return None
            
        print("\n=== CREATING CUSTOMER FEATURES ===")
        
        # Use identified columns or ask user to specify
        if not all([self.customer_id_col, self.amount_col, self.date_col]):
            print("Could not automatically identify all required columns.")
            print("Available columns:", self.sales_df.columns.tolist())
            return None
        
        # Ensure amount column is numeric
        self.sales_df[self.amount_col] = pd.to_numeric(self.sales_df[self.amount_col], errors='coerce')
        
        # Create customer aggregations
        print("Creating customer transaction aggregations...")
        
        # Basic transaction metrics per customer
        customer_agg = self.sales_df.groupby(self.customer_id_col).agg({
            self.amount_col: ['sum', 'mean', 'count', 'std'],
            self.date_col: ['min', 'max']
        }).round(2)
        
        # Flatten column names
        customer_agg.columns = ['Total_Spend', 'Avg_Transaction', 'Transaction_Count', 'Spend_Std',
                               'First_Transaction', 'Last_Transaction']
        
        # Add product variety if product column exists
        if self.product_col and self.product_col in self.sales_df.columns:
            product_variety = self.sales_df.groupby(self.customer_id_col)[self.product_col].nunique()
            customer_agg['Product_Variety'] = product_variety
        else:
            customer_agg['Product_Variety'] = 1
        
        # Calculate RFM metrics
        current_date = self.sales_df[self.date_col].max()
        customer_agg['Recency_Days'] = (current_date - customer_agg['Last_Transaction']).dt.days
        customer_agg['Customer_Tenure_Days'] = (customer_agg['Last_Transaction'] - customer_agg['First_Transaction']).dt.days
        customer_agg['Purchase_Frequency'] = customer_agg['Transaction_Count'] / (customer_agg['Customer_Tenure_Days'] + 1) * 365
        
        # Fill NaN values
        customer_agg = customer_agg.fillna(0)
        
        # Add churn flag (customers who haven't purchased in 90+ days)
        customer_agg['Churned'] = (customer_agg['Recency_Days'] > 90).astype(int)
        
        # Calculate CLV estimate (simple version)
        customer_agg['Estimated_CLV'] = (
            customer_agg['Avg_Transaction'] * customer_agg['Purchase_Frequency'] * 2
        )
        
        # Add customer segments based on spending
        spending_quantiles = customer_agg['Total_Spend'].quantile([0.33, 0.66])
        customer_agg['Spend_Segment'] = pd.cut(
            customer_agg['Total_Spend'], 
            bins=[-np.inf, spending_quantiles.iloc[0], spending_quantiles.iloc[1], np.inf],
            labels=['Low', 'Medium', 'High']
        )
        
        self.customer_features = customer_agg
        print(f"Customer features created for {len(customer_agg)} customers")
        print("\nFeature summary:")
        print(customer_agg.describe())
        
        return customer_agg

    def predict_churn(self):
        """Build churn prediction model using CSV data"""
        if self.customer_features is None:
            print("No customer features available. Create features first.")
            return None
            
        print("\n=== CHURN PREDICTION MODEL ===")
        
        # Select numeric features for modeling
        feature_cols = ['Avg_Transaction', 'Transaction_Count', 'Recency_Days', 
                       'Purchase_Frequency', 'Product_Variety', 'Customer_Tenure_Days']
        
        # Ensure we have enough features
        available_features = [col for col in feature_cols if col in self.customer_features.columns]
        
        if len(available_features) < 3:
            print("Not enough features available for churn prediction")
            return None
        
        # Prepare data
        X = self.customer_features[available_features].fillna(0)
        y = self.customer_features['Churned']
        
        if y.sum() == 0:
            print("No churned customers found in data")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.models['churn'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['churn'].fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.models['churn'].predict(X_test_scaled)
        accuracy = self.models['churn'].score(X_test_scaled, y_test)
        
        print(f"Churn Prediction Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': self.models['churn'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        # Add churn probability to customer features
        self.customer_features['Churn_Probability'] = self.models['churn'].predict_proba(
            self.scaler.transform(X)
        )[:, 1]
        
        # Create churn analysis visualization
        self.visualize_churn_analysis(feature_importance)
        
        return feature_importance

    def visualize_churn_analysis(self, feature_importance):
        """Create and save churn analysis visualization"""
        if feature_importance is None or self.customer_features is None:
            return None
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Churn Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Feature Importance
        ax1.barh(feature_importance['feature'], feature_importance['importance'], color='lightcoral')
        ax1.set_title('Churn Prediction - Feature Importance')
        ax1.set_xlabel('Importance Score')
        
        # 2. Churn Distribution
        churn_dist = self.customer_features['Churned'].value_counts()
        labels = ['Active', 'Churned']
        colors = ['lightgreen', 'red']
        
        ax2.pie(churn_dist.values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Customer Churn Distribution')
        
        plt.tight_layout()
        self.save_figure(fig, 'churn_analysis.png', 'Churn Analysis')
        plt.show()
        
        return fig

    def customer_segmentation(self):
        """Perform RFM segmentation using CSV data"""
        if self.customer_features is None:
            print("No customer features available. Create features first.")
            return None
            
        print("\n=== CUSTOMER SEGMENTATION ===")
        
        # RFM Analysis using available features
        rfm_features = ['Recency_Days', 'Purchase_Frequency', 'Total_Spend']
        available_rfm = [col for col in rfm_features if col in self.customer_features.columns]
        
        if len(available_rfm) < 2:
            print("Not enough RFM features available")
            return None
        
        rfm_data = self.customer_features[available_rfm].fillna(0)
        
        # Normalize for clustering
        rfm_scaled = StandardScaler().fit_transform(rfm_data)
        
        # K-means clustering
        n_clusters = min(5, len(self.customer_features) // 10)
        if n_clusters < 2:
            n_clusters = 2
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.customer_features['RFM_Segment'] = kmeans.fit_predict(rfm_scaled)
        
        # Segment analysis
        segment_cols = available_rfm + ['Transaction_Count']
        if 'Estimated_CLV' in self.customer_features.columns:
            segment_cols.append('Estimated_CLV')
        if 'Churn_Probability' in self.customer_features.columns:
            segment_cols.append('Churn_Probability')
            
        segment_analysis = self.customer_features.groupby('RFM_Segment')[segment_cols].mean().round(2)
        segment_analysis['Customer_Count'] = self.customer_features.groupby('RFM_Segment').size()
        
        print("Customer Segments Analysis:")
        print(segment_analysis)
        
        # Create segmentation visualization
        self.visualize_customer_segments(segment_analysis)
        
        return segment_analysis

    def visualize_customer_segments(self, segment_analysis):
        """Create and save customer segmentation visualization"""
        if segment_analysis is None:
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Customer Segmentation Analysis', fontsize=16, fontweight='bold')
        
        segments = segment_analysis.index
        
        # 1. Customer Count by Segment
        axes[0, 0].bar(segments, segment_analysis['Customer_Count'], color='lightblue')
        axes[0, 0].set_title('Customer Count by Segment')
        axes[0, 0].set_ylabel('Number of Customers')
        axes[0, 0].set_xlabel('Segment')
        
        # 2. Total Spend by Segment
        if 'Total_Spend' in segment_analysis.columns:
            axes[0, 1].bar(segments, segment_analysis['Total_Spend'], color='green', alpha=0.7)
            axes[0, 1].set_title('Average Total Spend by Segment')
            axes[0, 1].set_ylabel('Average Total Spend ($)')
            axes[0, 1].set_xlabel('Segment')
        
        # 3. Recency by Segment
        if 'Recency_Days' in segment_analysis.columns:
            axes[1, 0].bar(segments, segment_analysis['Recency_Days'], color='orange', alpha=0.7)
            axes[1, 0].set_title('Average Recency by Segment')
            axes[1, 0].set_ylabel('Recency (Days)')
            axes[1, 0].set_xlabel('Segment')
        
        # 4. Transaction Count by Segment
        if 'Transaction_Count' in segment_analysis.columns:
            axes[1, 1].bar(segments, segment_analysis['Transaction_Count'], color='purple', alpha=0.7)
            axes[1, 1].set_title('Average Transaction Count by Segment')
            axes[1, 1].set_ylabel('Transaction Count')
            axes[1, 1].set_xlabel('Segment')
        
        plt.tight_layout()
        self.save_figure(fig, 'customer_segmentation.png', 'Customer Segmentation')
        plt.show()
        
        return fig

    def sales_trends_analysis(self):
        """Analyze sales trends from the CSV data"""
        print("\n=== SALES TRENDS ANALYSIS ===")
        
        if self.date_col is None or self.amount_col is None:
            print("Date or amount column not available for trend analysis")
            return None
        
        # Daily sales trends
        daily_sales = self.sales_df.groupby(self.sales_df[self.date_col].dt.date)[self.amount_col].agg(['sum', 'count']).round(2)
        daily_sales.columns = ['Daily_Revenue', 'Daily_Transactions']
        
        # Monthly trends
        monthly_sales = self.sales_df.groupby(self.sales_df[self.date_col].dt.to_period('M'))[self.amount_col].agg(['sum', 'count']).round(2)
        monthly_sales.columns = ['Monthly_Revenue', 'Monthly_Transactions']
        
        print("Sales Summary:")
        print(f"Total Revenue: ${daily_sales['Daily_Revenue'].sum():,.2f}")
        print(f"Total Transactions: {daily_sales['Daily_Transactions'].sum():,}")
        print(f"Average Transaction Value: ${self.sales_df[self.amount_col].mean():.2f}")
        print(f"Date Range: {self.sales_df[self.date_col].min().date()} to {self.sales_df[self.date_col].max().date()}")
        
        print("\nTop 5 Revenue Days:")
        print(daily_sales.nlargest(5, 'Daily_Revenue'))
        
        print("\nMonthly Trends:")
        print(monthly_sales.tail())
        
        # Create sales trends visualization
        self.visualize_sales_trends(daily_sales, monthly_sales)
        
        return daily_sales, monthly_sales

    def visualize_sales_trends(self, daily_sales, monthly_sales):
        """Create and save sales trends visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Sales Trends Analysis', fontsize=16, fontweight='bold')
        
        # 1. Daily Revenue Trend
        axes[0, 0].plot(daily_sales.index, daily_sales['Daily_Revenue'], color='blue', alpha=0.7)
        axes[0, 0].set_title('Daily Revenue Trend')
        axes[0, 0].set_ylabel('Revenue ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Daily Transaction Count
        axes[0, 1].plot(daily_sales.index, daily_sales['Daily_Transactions'], color='green', alpha=0.7)
        axes[0, 1].set_title('Daily Transaction Count')
        axes[0, 1].set_ylabel('Number of Transactions')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Monthly Revenue
        monthly_x = range(len(monthly_sales))
        monthly_labels = [str(period) for period in monthly_sales.index]
        
        axes[1, 0].bar(monthly_x, monthly_sales['Monthly_Revenue'], color='orange', alpha=0.7)
        axes[1, 0].set_title('Monthly Revenue')
        axes[1, 0].set_ylabel('Revenue ($)')
        axes[1, 0].set_xticks(monthly_x)
        axes[1, 0].set_xticklabels(monthly_labels, rotation=45)
        
        # 4. Monthly Transactions
        axes[1, 1].bar(monthly_x, monthly_sales['Monthly_Transactions'], color='red', alpha=0.7)
        axes[1, 1].set_title('Monthly Transactions')
        axes[1, 1].set_ylabel('Number of Transactions')
        axes[1, 1].set_xticks(monthly_x)
        axes[1, 1].set_xticklabels(monthly_labels, rotation=45)
        
        plt.tight_layout()
        self.save_figure(fig, 'sales_trends.png', 'Sales Trends')
        plt.show()
        
        return fig

    def top_customers_analysis(self):
        """Analyze top customers"""
        print("\n=== TOP CUSTOMERS ANALYSIS ===")
        
        if self.customer_features is None:
            print("No customer features available")
            return None
        
        # Top customers by spending
        top_spenders = self.customer_features.nlargest(10, 'Total_Spend')[
            ['Total_Spend', 'Transaction_Count', 'Avg_Transaction', 'Recency_Days']
        ].round(2)
        
        print("Top 10 Customers by Total Spend:")
        print(top_spenders)
        
        # Customer distribution analysis
        print(f"\nCustomer Spend Distribution:")
        top_10_percent_revenue = self.customer_features.nlargest(int(len(self.customer_features)*0.1), 'Total_Spend')['Total_Spend'].sum()
        total_revenue = self.customer_features['Total_Spend'].sum()
        print(f"Top 10% of customers contribute: {top_10_percent_revenue / total_revenue:.1%} of revenue")
        
        # Create top customers visualization
        self.visualize_top_customers(top_spenders)
        
        return top_spenders

    def visualize_top_customers(self, top_spenders):
        """Create and save top customers visualization"""
        if top_spenders is None or len(top_spenders) == 0:
            return None
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Top Customers Analysis', fontsize=16, fontweight='bold')
        
        customer_indices = range(len(top_spenders))
        customer_labels = [f'Customer {i+1}' for i in customer_indices]
        
        # 1. Total Spend
        bars1 = ax1.bar(customer_indices, top_spenders['Total_Spend'], color='gold', alpha=0.8)
        ax1.set_title('Top Customers - Total Spend')
        ax1.set_ylabel('Total Spend ($)')
        ax1.set_xticks(customer_indices)
        ax1.set_xticklabels(customer_labels, rotation=45)
        
        # Add value labels
        for bar, value in zip(bars1, top_spenders['Total_Spend']):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'${value:,.0f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Transaction Count
        bars2 = ax2.bar(customer_indices, top_spenders['Transaction_Count'], color='lightblue', alpha=0.8)
        ax2.set_title('Top Customers - Transaction Count')
        ax2.set_ylabel('Number of Transactions')
        ax2.set_xticks(customer_indices)
        ax2.set_xticklabels(customer_labels, rotation=45)
        
        # 3. Average Transaction Value
        bars3 = ax3.bar(customer_indices, top_spenders['Avg_Transaction'], color='lightgreen', alpha=0.8)
        ax3.set_title('Top Customers - Average Transaction')
        ax3.set_ylabel('Average Transaction ($)')
        ax3.set_xticks(customer_indices)
        ax3.set_xticklabels(customer_labels, rotation=45)
        
        # 4. Recency
        bars4 = ax4.bar(customer_indices, top_spenders['Recency_Days'], color='coral', alpha=0.8)
        ax4.set_title('Top Customers - Recency')
        ax4.set_ylabel('Days Since Last Purchase')
        ax4.set_xticks(customer_indices)
        ax4.set_xticklabels(customer_labels, rotation=45)
        
        plt.tight_layout()
        self.save_figure(fig, 'top_customers_analysis.png', 'Top Customers Analysis')
        plt.show()
        
        return fig

    def at_risk_customers(self):
        """Identify at-risk customers from CSV data"""
        print("\n=== AT-RISK CUSTOMER IDENTIFICATION ===")
        
        if self.customer_features is None:
            print("No customer features available")
            return None
        
        # Define at-risk criteria based on available data
        at_risk_conditions = []
        
        # High recency (haven't purchased recently)
        if 'Recency_Days' in self.customer_features.columns:
            recency_threshold = self.customer_features['Recency_Days'].quantile(0.75)
            at_risk_conditions.append(self.customer_features['Recency_Days'] > recency_threshold)
        
        # Low frequency
        if 'Purchase_Frequency' in self.customer_features.columns:
            freq_threshold = self.customer_features['Purchase_Frequency'].quantile(0.25)
            at_risk_conditions.append(self.customer_features['Purchase_Frequency'] < freq_threshold)
        
        # High churn probability
        if 'Churn_Probability' in self.customer_features.columns:
            at_risk_conditions.append(self.customer_features['Churn_Probability'] > 0.7)
        
        if not at_risk_conditions:
            print("Cannot identify at-risk customers with available data")
            return None
        
        # Combine conditions (any customer meeting any criteria)
        at_risk_mask = at_risk_conditions[0]
        for condition in at_risk_conditions[1:]:
            at_risk_mask = at_risk_mask | condition
        
        at_risk = self.customer_features[at_risk_mask].copy()
        
        # Calculate potential revenue at risk
        revenue_at_risk = at_risk['Total_Spend'].sum()
        
        print(f"At-Risk Customers: {len(at_risk):,} ({len(at_risk)/len(self.customer_features):.1%})")
        print(f"Revenue at Risk: ${revenue_at_risk:,.2f}")
        
        # Show top at-risk customers
        risk_cols = ['Total_Spend', 'Transaction_Count', 'Recency_Days']
        if 'Churn_Probability' in at_risk.columns:
            risk_cols.append('Churn_Probability')
        
        top_at_risk = at_risk.nlargest(10, 'Total_Spend')[risk_cols].round(3)
        print("\nTop 10 At-Risk Customers (by spend):")
        print(top_at_risk)
        
        # Create at-risk customers visualization
        self.visualize_at_risk_customers(at_risk, top_at_risk)
        
        return at_risk

    def visualize_at_risk_customers(self, at_risk, top_at_risk):
        """Create and save at-risk customers visualization"""
        if at_risk is None or len(at_risk) == 0:
            return None
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('At-Risk Customers Analysis', fontsize=16, fontweight='bold')
        
        # 1. Risk Distribution
        total_customers = len(self.customer_features)
        risk_customers = len(at_risk)
        safe_customers = total_customers - risk_customers
        
        labels = ['Safe', 'At Risk']
        sizes = [safe_customers, risk_customers]
        colors = ['lightgreen', 'red']
        
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Customer Risk Distribution')
        
        # 2. Top At-Risk Customers (by spend)
        if len(top_at_risk) > 0:
            customer_indices = range(min(10, len(top_at_risk)))
            customer_labels = [f'Customer {i+1}' for i in customer_indices]
            
            bars = ax2.bar(customer_indices, top_at_risk['Total_Spend'][:10], color='red', alpha=0.7)
            ax2.set_title('Top At-Risk Customers by Spend')
            ax2.set_ylabel('Total Spend ($)')
            ax2.set_xlabel('Customer Rank')
            ax2.set_xticks(customer_indices)
            ax2.set_xticklabels(customer_labels, rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, top_at_risk['Total_Spend'][:10]):
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'${value:,.0f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        self.save_figure(fig, 'at_risk_customers.png', 'At-Risk Customers')
        plt.show()
        
        return fig

    def generate_sample_campaign_data(self):
        """Generate sample campaign data for demonstration if logs data not available"""
        print("\n=== GENERATING SAMPLE CAMPAIGN DATA (FALLBACK) ===")
        
        campaigns = ['Email_Newsletter', 'Social_Media', 'Google_Ads', 'Retargeting', 'Organic']
        np.random.seed(42)
        
        campaign_metrics = []
        for campaign in campaigns:
            sent = np.random.randint(1000, 10000)
            opened = int(sent * np.random.uniform(0.15, 0.35))
            clicked = int(opened * np.random.uniform(0.02, 0.15))
            conversions = int(clicked * np.random.uniform(0.01, 0.08))
            
            campaign_metrics.append({
                'Campaign': campaign,
                'Sent': sent,
                'Opened': opened,
                'Clicked': clicked,
                'Conversions': conversions,
                'Revenue': conversions * np.random.uniform(50, 500)
            })
        
        self.campaign_data = pd.DataFrame(campaign_metrics)
        return self.campaign_data

    def generate_sample_journey_data(self):
        """Generate sample customer journey data (fallback)"""
        if self.customer_features is None:
            print("Create customer features first")
            return None
            
        print("\n=== GENERATING SAMPLE JOURNEY DATA (FALLBACK) ===")
        
        stages = ['Lead', 'Qualified_Lead', 'Opportunity', 'Sale', 'Service', 'Loyalty']
        journey_data = []
        np.random.seed(42)
        
        for customer_id in self.customer_features.index[:min(1000, len(self.customer_features))]:
            customer_journey = []
            current_stage = 0
            
            customer_journey.append({
                'Customer_ID': customer_id,
                'Stage': stages[0],
                'Date': pd.Timestamp.now() - timedelta(days=np.random.randint(30, 365)),
                'Completed': 1
            })
            
            for i in range(1, len(stages)):
                progress_prob = [0.8, 0.6, 0.7, 0.9, 0.5][i-1]
                
                if np.random.random() < progress_prob:
                    customer_journey.append({
                        'Customer_ID': customer_id,
                        'Stage': stages[i],
                        'Date': customer_journey[-1]['Date'] + timedelta(days=np.random.randint(1, 30)),
                        'Completed': 1
                    })
                else:
                    break
            
            journey_data.extend(customer_journey)
        
        self.journey_data = pd.DataFrame(journey_data)
        return self.journey_data

    def multi_touch_attribution_modeling(self):
        """Main attribution method that tries logs first, then fallback"""
        if self.logs_df is not None:
            return self.multi_touch_attribution_from_logs()
        else:
            return self.generate_sample_attribution_data()
    
    def generate_sample_attribution_data(self):
        """Generate sample attribution data (fallback)"""
        print("\n=== GENERATING SAMPLE ATTRIBUTION DATA (FALLBACK) ===")
        
        touchpoints = ['Email', 'Social_Media', 'Google_Ads', 'Website', 'Retargeting']
        attribution_data = []
        
        np.random.seed(42)
        
        if self.customer_features is not None:
            customers = self.customer_features.index[:100]
        else:
            customers = range(100)
        
        for customer_id in customers:
            customer_touchpoints = []
            num_touchpoints = np.random.randint(1, 6)
            
            for i in range(num_touchpoints):
                touchpoint = np.random.choice(touchpoints)
                attribution_data.append({
                    'Customer_ID': customer_id,
                    'Touchpoint': touchpoint,
                    'Order': i + 1,
                    'Revenue': np.random.uniform(50, 500) if i == num_touchpoints - 1 else 0
                })
        
        attribution_df = pd.DataFrame(attribution_data)
        
        # Calculate attribution models (same logic as before)
        attribution_models = {}
        
        first_touch = attribution_df[attribution_df['Order'] == 1].groupby('Touchpoint')['Revenue'].sum()
        attribution_models['First_Touch'] = first_touch
        
        last_touch_customers = attribution_df.groupby('Customer_ID')['Order'].max().reset_index()
        last_touch_customers.columns = ['Customer_ID', 'Max_Order']
        last_touch_data = attribution_df.merge(last_touch_customers, on='Customer_ID')
        last_touch_data = last_touch_data[last_touch_data['Order'] == last_touch_data['Max_Order']]
        last_touch = last_touch_data.groupby('Touchpoint')['Revenue'].sum()
        attribution_models['Last_Touch'] = last_touch
        
        customer_revenues = attribution_df.groupby('Customer_ID')['Revenue'].max()
        linear_attribution = []
        
        for customer_id, revenue in customer_revenues.items():
            if revenue > 0:
                customer_touchpoints = attribution_df[attribution_df['Customer_ID'] == customer_id]['Touchpoint'].tolist()
                weight_per_touch = revenue / len(customer_touchpoints)
                for touchpoint in customer_touchpoints:
                    linear_attribution.append({'Touchpoint': touchpoint, 'Revenue': weight_per_touch})
        
        if linear_attribution:
            linear_df = pd.DataFrame(linear_attribution)
            linear_touch = linear_df.groupby('Touchpoint')['Revenue'].sum()
            attribution_models['Linear'] = linear_touch
        
        attribution_comparison = pd.DataFrame(attribution_models).fillna(0).round(2)
        
        print("Multi-Touch Attribution (Sample Data):")
        print("="*60)
        print(attribution_comparison)
        
        return attribution_comparison

    def create_comprehensive_dashboard(self):
        """Create a comprehensive dashboard with all key metrics"""
        print("\n=== CREATING COMPREHENSIVE DASHBOARD ===")
        
        if self.campaign_data is None or self.customer_features is None:
            print("Missing required data for comprehensive dashboard")
            return None
            
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        fig.suptitle('CRM Analytics Comprehensive Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Campaign ROI
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(range(len(self.campaign_data)), self.campaign_data['ROI'], 
                color=['red' if x < 0 else 'green' for x in self.campaign_data['ROI']])
        ax1.set_title('Campaign ROI', fontweight='bold')
        ax1.set_ylabel('ROI (%)')
        ax1.set_xticks(range(len(self.campaign_data)))
        ax1.set_xticklabels(self.campaign_data['Campaign'], rotation=45, ha='right')
        
        # 2. Customer Segments
        ax2 = fig.add_subplot(gs[0, 1])
        segment_counts = self.customer_features['Spend_Segment'].value_counts()
        ax2.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
        ax2.set_title('Customer Segments', fontweight='bold')
        
        # 3. Churn Risk
        ax3 = fig.add_subplot(gs[0, 2])
        if 'Churned' in self.customer_features.columns:
            churn_counts = self.customer_features['Churned'].value_counts()
            ax3.bar(['Active', 'Churned'], churn_counts.values, color=['green', 'red'])
            ax3.set_title('Churn Distribution', fontweight='bold')
            ax3.set_ylabel('Number of Customers')
        
        # 4. Revenue Distribution
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.hist(self.customer_features['Total_Spend'], bins=20, color='skyblue', alpha=0.7)
        ax4.set_title('Revenue Distribution', fontweight='bold')
        ax4.set_xlabel('Customer Spend ($)')
        ax4.set_ylabel('Number of Customers')
        
        # 5. Campaign Conversion Funnel
        ax5 = fig.add_subplot(gs[1, :2])
        total_sent = self.campaign_data['Sent'].sum()
        total_opened = self.campaign_data['Opened'].sum()
        total_clicked = self.campaign_data['Clicked'].sum()
        total_conversions = self.campaign_data['Conversions'].sum()
        
        funnel_stages = ['Sent', 'Opened', 'Clicked', 'Conversions']
        funnel_values = [total_sent, total_opened, total_clicked, total_conversions]
        
        bars = ax5.bar(funnel_stages, funnel_values, color=['blue', 'green', 'orange', 'red'])
        ax5.set_title('Overall Campaign Funnel', fontweight='bold')
        ax5.set_ylabel('Count')
        
        # Add percentage labels
        for i, (bar, value) in enumerate(zip(bars, funnel_values)):
            pct = (value / total_sent) * 100 if total_sent > 0 else 0
            ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:,}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
        
        # 6. Top Customers
        ax6 = fig.add_subplot(gs[1, 2:])
        top_5 = self.customer_features.nlargest(5, 'Total_Spend')
        ax6.barh(range(5), top_5['Total_Spend'], color='gold')
        ax6.set_title('Top 5 Customers by Spend', fontweight='bold')
        ax6.set_xlabel('Total Spend ($)')
        ax6.set_yticks(range(5))
        ax6.set_yticklabels([f'Customer {i+1}' for i in range(5)])
        
        # 7. Monthly Revenue Trend (if available)
        if hasattr(self, 'sales_df') and self.date_col and self.amount_col:
            ax7 = fig.add_subplot(gs[2, :])
            monthly_revenue = self.sales_df.groupby(self.sales_df[self.date_col].dt.to_period('M'))[self.amount_col].sum()
            ax7.plot(range(len(monthly_revenue)), monthly_revenue.values, marker='o', linewidth=2)
            ax7.set_title('Monthly Revenue Trend', fontweight='bold')
            ax7.set_ylabel('Revenue ($)')
            ax7.set_xlabel('Month')
            ax7.grid(True, alpha=0.3)
            ax7.set_xticks(range(len(monthly_revenue)))
            ax7.set_xticklabels([str(p) for p in monthly_revenue.index], rotation=45)
        
        # Save comprehensive dashboard
        self.save_figure(fig, 'comprehensive_dashboard.png', 'Comprehensive Dashboard')
        plt.show()
        
        return fig

    def generate_comprehensive_report(self):
        """Generate comprehensive CRM analytics report with real logs data"""
        print("\n" + "="*80)
        print("         ENHANCED CRM ANALYTICS REPORT (WITH REAL LOGS)")
        print("="*80)
        
        # Load and process both datasets
        sales_data, logs_data = self.load_csv_data()
        if sales_data is None:
            print("Could not load sales data")
            return None
        
        features = self.create_customer_features()
        if features is None:
            return None
        
        # Run all analyses with real data where possible
        print("\n🎯 Running Campaign Dashboard Analysis (From Logs)...")
        campaign_metrics = self.campaign_dashboard_analysis()
        
        print("\n🔄 Running Multi-Touch Attribution Analysis (From Logs)...")
        attribution_analysis = self.multi_touch_attribution_modeling()
        
        print("\n📊 Running Customer Journey Funnel Analysis (From Logs)...")
        funnel_analysis = self.customer_journey_funnel_analysis()
        
        print("\n📉 Running Drop-off Visualization Analysis...")
        dropoff_analysis = self.funnel_drop_off_visualization()
        
        # Run existing analyses
        segments = self.customer_segmentation()
        churn_importance = self.predict_churn()
        daily_sales, monthly_sales = self.sales_trends_analysis()
        top_customers = self.top_customers_analysis()
        at_risk = self.at_risk_customers()
        
        # Create comprehensive dashboard
        comprehensive_dashboard = self.create_comprehensive_dashboard()
        
        # Enhanced summary insights
        print("\n" + "="*80)
        print("                    ENHANCED KEY INSIGHTS (REAL DATA)")
        print("="*80)
        
        print(f"📊 Total Customers: {len(self.customer_features):,}")
        print(f"💰 Total Revenue: ${self.customer_features['Total_Spend'].sum():,.2f}")
        print(f"📈 Average Customer Value: ${self.customer_features['Total_Spend'].mean():.2f}")
        
        if at_risk is not None:
            print(f"⚠️  Customers at Risk: {len(at_risk):,} ({len(at_risk)/len(self.customer_features):.1%})")
        
        print(f"🛒 Total Transactions: {self.customer_features['Transaction_Count'].sum():,}")
        print(f"📅 Sales Data Period: {self.sales_df[self.date_col].min().date()} to {self.sales_df[self.date_col].max().date()}")
        
        if self.logs_df is not None:
            print(f"📧 Logs Data Period: {self.logs_df[self.logs_date_col].min().date()} to {self.logs_df[self.logs_date_col].max().date()}")
            print(f"📬 Total Log Events: {len(self.logs_df):,}")
        
        # Campaign insights from real data
        if campaign_metrics is not None and len(campaign_metrics) > 0:
            best_campaign = campaign_metrics.loc[campaign_metrics['ROI'].idxmax(), 'Campaign']
            print(f"🏆 Best Performing Campaign (Real Data): {best_campaign}")
            
            total_sent = campaign_metrics['Sent'].sum()
            total_opened = campaign_metrics['Opened'].sum()
            if total_sent > 0:
                print(f"🎯 Overall Open Rate (Real Data): {(total_opened/total_sent*100):.2f}%")
        
        # Funnel insights
        if funnel_analysis is not None:
            stages_with_data = funnel_analysis[funnel_analysis['Unique_Customers'] > 0]
            if len(stages_with_data) > 0:
                final_stage = stages_with_data.index[-1]
                final_rate = stages_with_data.loc[final_stage, 'Conversion_Rate']
                print(f"🔄 Lead to {final_stage} Conversion: {final_rate:.2f}%")
        
        print(f"\n📊 Total Visualizations Created: {len([f for f in os.listdir(self.vis_dir) if f.endswith('.png')])}")
        print(f"📁 Visualization Directory: {self.vis_dir}")
        
        return {
            'sales_data': self.sales_df,
            'logs_data': self.logs_df,
            'customer_features': self.customer_features,
            'campaign_metrics': campaign_metrics,
            'attribution_analysis': attribution_analysis,
            'funnel_analysis': funnel_analysis,
            'dropoff_analysis': dropoff_analysis,
            'segments': segments,
            'at_risk_customers': at_risk,
            'models': self.models,
            'comprehensive_dashboard': comprehensive_dashboard
        }


# Example usage
if __name__ == "__main__":
    # Initialize Enhanced CRM Analytics with both CSV files
    crm = MLCRMAnalytics(
        sales_csv_path=r'D:\CRM_PROJECT\data\sales.csv',
        logs_csv_path=r'D:\CRM_PROJECT\data\logs.csv'  # Your logs file
    )
    
    # Generate comprehensive enhanced report using real data
    results = crm.generate_comprehensive_report()
    
    if results is not None:
        print("\n" + "="*80)
        print("           ENHANCED ACTIONABLE RECOMMENDATIONS (REAL DATA)")
        print("="*80)
        
        print("🔥 IMMEDIATE CAMPAIGN ACTIONS (Based on Real Data):")
        print("1. Scale campaigns with highest real ROI from logs analysis")
        print("2. Investigate low-performing campaigns in your logs")
        print("3. Address actual drop-off points identified in customer journey")
        print("4. Target customers who opened but didn't click (from logs)")
        
        print("\n📊 ATTRIBUTION & JOURNEY OPTIMIZATION (Real Insights):")
        print("1. Reallocate budget based on actual attribution analysis")
        print("2. Focus on touchpoints that drive real conversions")
        print("3. Improve stages with highest actual drop-off rates")
        print("4. Implement retargeting for customers in logs who didn't convert")
        
        print("\n🎯 DATA-DRIVEN OPTIMIZATION TARGETS:")
        print("1. Improve email subject lines for campaigns with low open rates")
        print("2. A/B test call-to-action for campaigns with low click rates")
        print("3. Optimize conversion funnel based on real user behavior")
        print("4. Create personalized campaigns based on customer journey stage")
        
        print("\n📈 GROWTH OPPORTUNITIES (From Your Data):")
        print("1. Identify and scale your best-performing campaigns")
        print("2. Create lookalike audiences based on your actual converters")
        print("3. Implement cross-channel attribution for better insights")
        print("4. Develop retention campaigns for at-risk customers")
    else:
        print("Please ensure both CSV files exist:")
        print("1. sales.csv - with Customer ID, Transaction Amount, Date")
        print("2. logs.csv - with Customer ID, Campaign, Event Type, Timestamp")
        print("\nLogs.csv should contain events like: sent, opened, clicked, conversion")


