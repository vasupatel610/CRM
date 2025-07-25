import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# NLP Library
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure VADER lexicon is available
try:
    nltk.data.find('sentiment/vader_lexicon')
    sia = SentimentIntensityAnalyzer()
except LookupError:
    print("Downloading VADER lexicon...")
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

class SocialListeningDashboard:
    def __init__(self, sentiment_csv_path=r'D:\CRM_PROJECT\data\sentiment.csv'):
        self.sentiment_csv_path = sentiment_csv_path
        self.vis_dir = r'D:\CRM_PROJECT\visulization'
        self.sentiment_df = None
        self.social_metrics = None
        
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
    
    def load_sentiment_data(self):
        """Load and process sentiment.csv data with NLTK-VADER sentiment analysis"""
        try:
            print(f"Loading sentiment data from {self.sentiment_csv_path}...")
            self.sentiment_df = pd.read_csv(self.sentiment_csv_path)
            print(f"Sentiment data loaded successfully! Shape: {self.sentiment_df.shape}")
            
            print("\n=== SENTIMENT DATA ===")
            print("Column names:")
            print(self.sentiment_df.columns.tolist())
            print("\nFirst few rows:")
            print(self.sentiment_df.head())
            print("\nData types:")
            print(self.sentiment_df.dtypes)
            
            # Basic data cleaning
            self.sentiment_df.columns = self.sentiment_df.columns.str.strip()
            
            # Convert timestamp to datetime
            if 'timestamp' in self.sentiment_df.columns:
                self.sentiment_df['timestamp'] = pd.to_datetime(self.sentiment_df['timestamp'])
                print("Converted timestamp to datetime")
            
            # Clean data
            self.sentiment_df['brand_mentioned'] = self.sentiment_df['brand_mentioned'].fillna('Unknown')
            self.sentiment_df['platform'] = self.sentiment_df['platform'].fillna('Unknown')
            self.sentiment_df['hashtags'] = self.sentiment_df['hashtags'].fillna('')
            self.sentiment_df['mention_text'] = self.sentiment_df['mention_text'].fillna('')
            
            # *** RECOMPUTE SENTIMENT WITH NLTK-VADER ***
            if 'mention_text' in self.sentiment_df.columns:
                print("🔄 Recomputing sentiment scores with NLTK-VADER...")
                
                # Apply VADER sentiment analysis to each mention_text
                sentiment_results = self.sentiment_df['mention_text'].apply(self._analyze_sentiment_nltk)
                
                # Extract compound score and sentiment label
                self.sentiment_df['sentiment_score'] = sentiment_results.apply(lambda x: x['compound'])
                self.sentiment_df['sentiment'] = sentiment_results.apply(lambda x: self._classify_sentiment(x['compound']))
                
                print("✅ Sentiment analysis complete!")
            else:
                print("⚠️ 'mention_text' column not found - using existing sentiment data")
                if 'sentiment' not in self.sentiment_df.columns:
                    self.sentiment_df['sentiment'] = 'neutral'
                if 'sentiment_score' not in self.sentiment_df.columns:
                    self.sentiment_df['sentiment_score'] = 0.0
            
            return self.sentiment_df
            
        except FileNotFoundError as e:
            print(f"Error: File not found! {str(e)}")
            return None
        except Exception as e:
            print(f"Error loading sentiment data: {str(e)}")
            return None

    def _analyze_sentiment_nltk(self, text):
        """Analyze sentiment using NLTK-VADER"""
        if not isinstance(text, str) or text.strip() == "":
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
        
        return sia.polarity_scores(text)
    
    @staticmethod
    def _classify_sentiment(compound_score):
        """Classify sentiment based on compound score"""
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    def brand_mentions_analysis(self):
        """Analyze brand mentions"""
        print("\n=== BRAND MENTIONS ANALYSIS ===")
        
        if self.sentiment_df is None:
            print("No sentiment data available")
            return None
        
        # Brand mention counts
        brand_mentions = self.sentiment_df['brand_mentioned'].value_counts()
        
        # Brand mention trends over time
        if 'timestamp' in self.sentiment_df.columns:
            self.sentiment_df['date'] = self.sentiment_df['timestamp'].dt.date
            daily_mentions = self.sentiment_df.groupby(['date', 'brand_mentioned']).size().unstack(fill_value=0)
            
            # Get top brands for trending analysis
            top_brands = brand_mentions.head(5).index.tolist()
            daily_mentions_top = daily_mentions[top_brands] if len(top_brands) > 0 else daily_mentions
        else:
            daily_mentions_top = None
        
        # Platform distribution by brand
        platform_brand = self.sentiment_df.groupby(['brand_mentioned', 'platform']).size().unstack(fill_value=0)
        
        print(f"Total Mentions: {len(self.sentiment_df):,}")
        print(f"Unique Brands: {brand_mentions.nunique():,}")
        print(f"\nTop 10 Brand Mentions:")
        print(brand_mentions.head(10))
        
        # Create brand mentions visualization
        self.visualize_brand_mentions(brand_mentions, daily_mentions_top, platform_brand)
        
        return {
            'brand_mentions': brand_mentions,
            'daily_mentions': daily_mentions_top,
            'platform_brand': platform_brand
        }

    def visualize_brand_mentions(self, brand_mentions, daily_mentions, platform_brand):
        """Create and save brand mentions visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Brand Mentions Dashboard', fontsize=18, fontweight='bold')
        
        # 1. Top Brand Mentions
        top_brands = brand_mentions.head(15)
        axes[0, 0].bar(range(len(top_brands)), top_brands.values, color='dodgerblue')
        axes[0, 0].set_title('Top 15 Brand Mentions')
        axes[0, 0].set_ylabel('Mention Count')
        axes[0, 0].set_xticks(range(len(top_brands)))
        axes[0, 0].set_xticklabels(top_brands.index, rotation=45, ha='right')
        
        # Add value labels
        for i, v in enumerate(top_brands.values):
            axes[0, 0].text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=8)
        
        # 2. Brand Mentions Over Time
        if daily_mentions is not None and not daily_mentions.empty:
            for brand in daily_mentions.columns[:5]:  # Top 5 brands
                axes[0, 1].plot(daily_mentions.index, daily_mentions[brand], marker='o', label=brand, linewidth=2)
            axes[0, 1].set_title('Brand Mention Trends Over Time')
            axes[0, 1].set_ylabel('Daily Mentions')
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No time data available', ha='center', va='center', 
                           transform=axes[0, 1].transAxes, fontsize=12)
            axes[0, 1].set_title('Brand Mention Trends')
        
        # 3. Platform Distribution for Top Brands
        if not platform_brand.empty:
            top_5_brands = brand_mentions.head(5).index
            platform_subset = platform_brand.loc[top_5_brands]
            
            # Stacked bar chart
            bottom = np.zeros(len(platform_subset))
            colors = plt.cm.Set3(np.linspace(0, 1, len(platform_subset.columns)))
            
            for i, platform in enumerate(platform_subset.columns):
                axes[1, 0].bar(range(len(platform_subset)), platform_subset[platform], 
                              bottom=bottom, label=platform, color=colors[i])
                bottom += platform_subset[platform]
            
            axes[1, 0].set_title('Platform Distribution - Top 5 Brands')
            axes[1, 0].set_ylabel('Mention Count')
            axes[1, 0].set_xlabel('Brands')
            axes[1, 0].set_xticks(range(len(platform_subset)))
            axes[1, 0].set_xticklabels(platform_subset.index, rotation=45, ha='right')
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            axes[1, 0].text(0.5, 0.5, 'No platform data available', ha='center', va='center', 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_title('Platform Distribution')
        
        # 4. Brand Mention Share (Pie Chart)
        top_10_brands = brand_mentions.head(10)
        others_count = brand_mentions[10:].sum() if len(brand_mentions) > 10 else 0
        
        if others_count > 0:
            pie_data = list(top_10_brands.values) + [others_count]
            pie_labels = list(top_10_brands.index) + ['Others']
        else:
            pie_data = top_10_brands.values
            pie_labels = top_10_brands.index
        
        axes[1, 1].pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Brand Mention Share')
        
        plt.tight_layout()
        self.save_figure(fig, 'brand_mentions_dashboard.png', 'Brand Mentions Dashboard')
        plt.show()
        
        return fig

    def hashtags_analysis(self):
        """Analyze hashtags trends"""
        print("\n=== HASHTAGS ANALYSIS ===")
        
        if self.sentiment_df is None:
            print("No sentiment data available")
            return None
        
        # Extract and count hashtags
        all_hashtags = []
        for hashtags in self.sentiment_df['hashtags'].dropna():
            if hashtags:
                # Split by space and clean hashtags
                tags = str(hashtags).split()
                for tag in tags:
                    if tag.startswith('#'):
                        all_hashtags.append(tag)
        
        if not all_hashtags:
            print("No hashtags found in data")
            return None
        
        hashtag_counts = pd.Series(all_hashtags).value_counts()
        
        # Hashtags by brand
        hashtag_brand_analysis = {}
        top_brands = self.sentiment_df['brand_mentioned'].value_counts().head(5).index
        
        for brand in top_brands:
            brand_data = self.sentiment_df[self.sentiment_df['brand_mentioned'] == brand]
            brand_hashtags = []
            for hashtags in brand_data['hashtags'].dropna():
                if hashtags:
                    tags = str(hashtags).split()
                    for tag in tags:
                        if tag.startswith('#'):
                            brand_hashtags.append(tag)
            if brand_hashtags:
                hashtag_brand_analysis[brand] = pd.Series(brand_hashtags).value_counts().head(10)
        
        # Hashtag trends over time
        hashtag_trends = None
        if 'timestamp' in self.sentiment_df.columns:
            self.sentiment_df['date'] = self.sentiment_df['timestamp'].dt.date
            hashtag_daily = {}
            
            for date, group in self.sentiment_df.groupby('date'):
                daily_hashtags = []
                for hashtags in group['hashtags'].dropna():
                    if hashtags:
                        tags = str(hashtags).split()
                        for tag in tags:
                            if tag.startswith('#'):
                                daily_hashtags.append(tag)
                hashtag_daily[date] = pd.Series(daily_hashtags).value_counts()
            
            # Convert to DataFrame for trending analysis
            if hashtag_daily:
                hashtag_trends = pd.DataFrame(hashtag_daily).fillna(0).T
        
        print(f"Total Hashtags Found: {len(all_hashtags):,}")
        print(f"Unique Hashtags: {len(hashtag_counts):,}")
        print(f"\nTop 15 Hashtags:")
        print(hashtag_counts.head(15))
        
        # Create hashtags visualization
        self.visualize_hashtags(hashtag_counts, hashtag_brand_analysis, hashtag_trends)
        
        return {
            'hashtag_counts': hashtag_counts,
            'hashtag_brand_analysis': hashtag_brand_analysis,
            'hashtag_trends': hashtag_trends
        }

    def visualize_hashtags(self, hashtag_counts, hashtag_brand_analysis, hashtag_trends):
        """Create and save hashtags visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Hashtags Analysis Dashboard', fontsize=18, fontweight='bold')
        
        # 1. Top Hashtags
        top_hashtags = hashtag_counts.head(20)
        axes[0, 0].bar(range(len(top_hashtags)), top_hashtags.values, color='orange')
        axes[0, 0].set_title('Top 20 Hashtags')
        axes[0, 0].set_ylabel('Usage Count')
        axes[0, 0].set_xticks(range(len(top_hashtags)))
        axes[0, 0].set_xticklabels(top_hashtags.index, rotation=45, ha='right')
        
        # Add value labels
        for i, v in enumerate(top_hashtags.values):
            axes[0, 0].text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=8)
        
        # 2. Hashtag Trends Over Time
        if hashtag_trends is not None and not hashtag_trends.empty:
            top_trending = hashtag_trends.sum().nlargest(5)
            for hashtag in top_trending.index:
                if hashtag in hashtag_trends.columns:
                    axes[0, 1].plot(hashtag_trends.index, hashtag_trends[hashtag], 
                                   marker='o', label=hashtag, linewidth=2)
            axes[0, 1].set_title('Hashtag Trends Over Time')
            axes[0, 1].set_ylabel('Daily Usage')
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No time trend data available', ha='center', va='center', 
                           transform=axes[0, 1].transAxes, fontsize=12)
            axes[0, 1].set_title('Hashtag Trends Over Time')
        
        # 3. Hashtags by Brand
        if hashtag_brand_analysis:
            brands = list(hashtag_brand_analysis.keys())[:3]  # Top 3 brands
            colors = ['skyblue', 'lightgreen', 'coral']
            
            for i, brand in enumerate(brands):
                brand_hashtags = hashtag_brand_analysis[brand].head(8)
                x_pos = np.arange(len(brand_hashtags)) + i * 0.25
                axes[1, 0].bar(x_pos, brand_hashtags.values, width=0.25, 
                              label=brand, color=colors[i % len(colors)])
            
            axes[1, 0].set_title('Top Hashtags by Brand')
            axes[1, 0].set_ylabel('Usage Count')
            axes[1, 0].set_xlabel('Hashtags')
            if brands:
                sample_hashtags = hashtag_brand_analysis[brands[0]].head(8).index
                axes[1, 0].set_xticks(np.arange(len(sample_hashtags)))
                axes[1, 0].set_xticklabels(sample_hashtags, rotation=45, ha='right')
            axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, 'No brand-hashtag data available', ha='center', va='center', 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_title('Hashtags by Brand')
        
        # 4. Hashtag Word Cloud Style (Bar chart)
        top_15_hashtags = hashtag_counts.head(15)
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_15_hashtags)))
        bars = axes[1, 1].barh(range(len(top_15_hashtags)), top_15_hashtags.values, color=colors)
        axes[1, 1].set_title('Top 15 Hashtags (Horizontal)')
        axes[1, 1].set_xlabel('Usage Count')
        axes[1, 1].set_yticks(range(len(top_15_hashtags)))
        axes[1, 1].set_yticklabels(top_15_hashtags.index)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_15_hashtags.values)):
            axes[1, 1].text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                           f'{value:,}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        self.save_figure(fig, 'hashtags_dashboard.png', 'Hashtags Dashboard')
        plt.show()
        
        return fig

    def sentiment_analysis(self):
        """Analyze sentiment data with NLTK-VADER results"""
        print("\n=== SENTIMENT ANALYSIS (NLTK-VADER) ===")
        
        if self.sentiment_df is None:
            print("No sentiment data available")
            return None
        
        # Overall sentiment distribution
        sentiment_dist = self.sentiment_df['sentiment'].value_counts()
        sentiment_percentages = (sentiment_dist / len(self.sentiment_df) * 100).round(2)
        
        # Sentiment by brand
        sentiment_by_brand = self.sentiment_df.groupby(['brand_mentioned', 'sentiment']).size().unstack(fill_value=0)
        
        # Calculate sentiment percentages for each brand
        sentiment_brand_pct = sentiment_by_brand.div(sentiment_by_brand.sum(axis=1), axis=0) * 100
        sentiment_brand_pct = sentiment_brand_pct.fillna(0).round(2)
        
        # Sentiment by platform
        sentiment_by_platform = self.sentiment_df.groupby(['platform', 'sentiment']).size().unstack(fill_value=0)
        
        # Sentiment over time
        sentiment_trends = None
        if 'timestamp' in self.sentiment_df.columns:
            self.sentiment_df['date'] = self.sentiment_df['timestamp'].dt.date
            sentiment_trends = self.sentiment_df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
        
        # Sentiment scores analysis (VADER compound scores)
        sentiment_scores = None
        if 'sentiment_score' in self.sentiment_df.columns:
            sentiment_scores = self.sentiment_df.groupby('sentiment')['sentiment_score'].describe()
        
        print("Sentiment Distribution (NLTK-VADER Analysis):")
        for sentiment, count in sentiment_dist.items():
            pct = sentiment_percentages[sentiment]
            print(f"  {sentiment.title()}: {count:,} ({pct:.1f}%)")
        
        print(f"\nSentiment by Top 10 Brands:")
        top_brands = self.sentiment_df['brand_mentioned'].value_counts().head(10).index
        brand_sentiment_summary = sentiment_brand_pct.loc[top_brands] if len(top_brands) > 0 else sentiment_brand_pct.head(10)
        print(brand_sentiment_summary)
        
        if sentiment_scores is not None:
            print(f"\nSentiment Score Statistics (VADER Compound Scores):")
            print(sentiment_scores)
        
        # Create sentiment visualization
        self.visualize_sentiment_analysis(sentiment_dist, sentiment_by_brand, sentiment_by_platform, 
                                         sentiment_trends, sentiment_brand_pct)
        
        return {
            'sentiment_dist': sentiment_dist,
            'sentiment_by_brand': sentiment_by_brand,
            'sentiment_by_platform': sentiment_by_platform,
            'sentiment_trends': sentiment_trends,
            'sentiment_brand_pct': sentiment_brand_pct,
            'sentiment_scores': sentiment_scores
        }

    def visualize_sentiment_analysis(self, sentiment_dist, sentiment_by_brand, sentiment_by_platform, 
                                    sentiment_trends, sentiment_brand_pct):
        """Create and save sentiment analysis visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(24, 12))
        fig.suptitle('Sentiment Analysis Dashboard (NLTK-VADER)', fontsize=18, fontweight='bold')
        
        # 1. Overall Sentiment Distribution
        colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
        pie_colors = [colors.get(sent, 'blue') for sent in sentiment_dist.index]
        
        axes[0, 0].pie(sentiment_dist.values, labels=sentiment_dist.index, autopct='%1.1f%%', 
                       colors=pie_colors, startangle=90)
        axes[0, 0].set_title('Overall Sentiment Distribution')
        
        # 2. Sentiment by Top Brands
        top_5_brands = sentiment_by_brand.sum(axis=1).nlargest(5).index
        brand_subset = sentiment_by_brand.loc[top_5_brands]
        
        x_pos = np.arange(len(brand_subset))
        width = 0.25
        
        sentiments = brand_subset.columns
        colors_bar = [colors.get(sent, 'blue') for sent in sentiments]
        
        for i, sentiment in enumerate(sentiments):
            offset = (i - len(sentiments)//2) * width
            bars = axes[0, 1].bar(x_pos + offset, brand_subset[sentiment], width, 
                                 label=sentiment.title(), color=colors_bar[i], alpha=0.8)
        
        axes[0, 1].set_title('Sentiment by Top 5 Brands')
        axes[0, 1].set_ylabel('Mention Count')
        axes[0, 1].set_xlabel('Brands')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(brand_subset.index, rotation=45, ha='right')
        axes[0, 1].legend()
        
        # 3. Sentiment by Platform
        if not sentiment_by_platform.empty:
            platform_subset = sentiment_by_platform
            
            bottom = np.zeros(len(platform_subset))
            for sentiment in platform_subset.columns:
                axes[0, 2].bar(range(len(platform_subset)), platform_subset[sentiment], 
                              bottom=bottom, label=sentiment.title(), 
                              color=colors.get(sentiment, 'blue'), alpha=0.8)
                bottom += platform_subset[sentiment]
            
            axes[0, 2].set_title('Sentiment by Platform')
            axes[0, 2].set_ylabel('Mention Count')
            axes[0, 2].set_xlabel('Platforms')
            axes[0, 2].set_xticks(range(len(platform_subset)))
            axes[0, 2].set_xticklabels(platform_subset.index, rotation=45, ha='right')
            axes[0, 2].legend()
        else:
            axes[0, 2].text(0.5, 0.5, 'No platform data available', ha='center', va='center', 
                           transform=axes[0, 2].transAxes, fontsize=12)
            axes[0, 2].set_title('Sentiment by Platform')
        
        # 4. Sentiment Trends Over Time
        if sentiment_trends is not None and not sentiment_trends.empty:
            for sentiment in sentiment_trends.columns:
                axes[1, 0].plot(sentiment_trends.index, sentiment_trends[sentiment], 
                               marker='o', label=sentiment.title(), 
                               color=colors.get(sentiment, 'blue'), linewidth=2)
            axes[1, 0].set_title('Sentiment Trends Over Time')
            axes[1, 0].set_ylabel('Daily Mentions')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].legend()
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No time trend data available', ha='center', va='center', 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_title('Sentiment Trends Over Time')
        
        # 5. Net Sentiment Score by Brand
        if 'positive' in sentiment_brand_pct.columns and 'negative' in sentiment_brand_pct.columns:
            net_sentiment = sentiment_brand_pct['positive'] - sentiment_brand_pct['negative']
            top_10_net = net_sentiment.nlargest(10)
            
            colors_net = ['green' if x > 0 else 'red' for x in top_10_net.values]
            axes[1, 1].bar(range(len(top_10_net)), top_10_net.values, color=colors_net, alpha=0.7)
            axes[1, 1].set_title('Net Sentiment Score (Top 10 Brands)')
            axes[1, 1].set_ylabel('Net Sentiment (%)')
            axes[1, 1].set_xlabel('Brands')
            axes[1, 1].set_xticks(range(len(top_10_net)))
            axes[1, 1].set_xticklabels(top_10_net.index, rotation=45, ha='right')
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Cannot calculate net sentiment', ha='center', va='center', 
                           transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('Net Sentiment Score')
        
        # 6. Sentiment Percentage Breakdown for Top Brands
        top_5_sentiment_pct = sentiment_brand_pct.loc[top_5_brands] if len(top_5_brands) > 0 else sentiment_brand_pct.head(5)
        
        # Stacked percentage bar chart
        bottom_pct = np.zeros(len(top_5_sentiment_pct))
        for sentiment in top_5_sentiment_pct.columns:
            axes[1, 2].bar(range(len(top_5_sentiment_pct)), top_5_sentiment_pct[sentiment], 
                          bottom=bottom_pct, label=sentiment.title(), 
                          color=colors.get(sentiment, 'blue'), alpha=0.8)
            bottom_pct += top_5_sentiment_pct[sentiment]
        
        axes[1, 2].set_title('Sentiment % Breakdown (Top 5 Brands)')
        axes[1, 2].set_ylabel('Sentiment Percentage (%)')
        axes[1, 2].set_xlabel('Brands')
        axes[1, 2].set_xticks(range(len(top_5_sentiment_pct)))
        axes[1, 2].set_xticklabels(top_5_sentiment_pct.index, rotation=45, ha='right')
        axes[1, 2].legend()
        
        plt.tight_layout()
        self.save_figure(fig, 'sentiment_analysis_dashboard.png', 'Sentiment Analysis Dashboard')
        plt.show()
        
        return fig

    def social_listening_comprehensive_dashboard(self):
        """Create comprehensive social listening dashboard"""
        print("\n=== COMPREHENSIVE SOCIAL LISTENING DASHBOARD ===")
        
        # Load data
        sentiment_data = self.load_sentiment_data()
        if sentiment_data is None:
            print("Cannot load sentiment data for social listening analysis")
            return None
        
        # Run all analyses
        brand_metrics = self.brand_mentions_analysis()
        hashtag_metrics = self.hashtags_analysis()
        sentiment_metrics = self.sentiment_analysis()
        
        # Store metrics
        self.social_metrics = {
            'brand_metrics': brand_metrics,
            'hashtag_metrics': hashtag_metrics,
            'sentiment_metrics': sentiment_metrics,
            'total_mentions': len(self.sentiment_df)
        }
        
        # Create combined dashboard
        self.create_combined_social_dashboard()
        
        return self.social_metrics

    def create_combined_social_dashboard(self):
        """Create combined social listening dashboard"""
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
        fig.suptitle('Social Listening Comprehensive Dashboard (NLTK-VADER)', fontsize=20, fontweight='bold')
        
        # Row 1: Brand Metrics
        # 1. Top Brand Mentions
        ax1 = fig.add_subplot(gs[0, 0])
        if self.social_metrics['brand_metrics']:
            brand_mentions = self.social_metrics['brand_metrics']['brand_mentions'].head(10)
            ax1.bar(range(len(brand_mentions)), brand_mentions.values, color='dodgerblue')
            ax1.set_title('Top 10 Brand Mentions', fontweight='bold')
            ax1.set_ylabel('Mentions')
            ax1.set_xticks(range(len(brand_mentions)))
            ax1.set_xticklabels(brand_mentions.index, rotation=45, ha='right')
            
            # Add value labels
            for i, v in enumerate(brand_mentions.values):
                ax1.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=8)
        
        # 2. Sentiment Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        if self.social_metrics['sentiment_metrics']:
            sentiment_dist = self.social_metrics['sentiment_metrics']['sentiment_dist']
            colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
            pie_colors = [colors.get(sent, 'blue') for sent in sentiment_dist.index]
            ax2.pie(sentiment_dist.values, labels=sentiment_dist.index, autopct='%1.1f%%', 
                   colors=pie_colors, startangle=90)
            ax2.set_title('Overall Sentiment Distribution', fontweight='bold')
        
        # 3. Platform Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        platform_dist = self.sentiment_df['platform'].value_counts().head(8)
        ax3.bar(range(len(platform_dist)), platform_dist.values, color='mediumorchid')
        ax3.set_title('Top Platforms', fontweight='bold')
        ax3.set_ylabel('Mentions')
        ax3.set_xticks(range(len(platform_dist)))
        ax3.set_xticklabels(platform_dist.index, rotation=45, ha='right')
        
        # Row 2: Hashtag Metrics
        # 4. Top Hashtags
        ax4 = fig.add_subplot(gs[1, 0])
        if self.social_metrics['hashtag_metrics']:
            hashtag_counts = self.social_metrics['hashtag_metrics']['hashtag_counts'].head(10)
            ax4.bar(range(len(hashtag_counts)), hashtag_counts.values, color='orange')
            ax4.set_title('Top 10 Hashtags', fontweight='bold')
            ax4.set_ylabel('Usage Count')
            ax4.set_xticks(range(len(hashtag_counts)))
            ax4.set_xticklabels(hashtag_counts.index, rotation=45, ha='right')
        
        # 5. Sentiment by Brand (Top 5)
        ax5 = fig.add_subplot(gs[1, 1])
        if self.social_metrics['sentiment_metrics']:
            sentiment_by_brand = self.social_metrics['sentiment_metrics']['sentiment_by_brand']
            top_5_brands = sentiment_by_brand.sum(axis=1).nlargest(5).index
            brand_subset = sentiment_by_brand.loc[top_5_brands]
            
            bottom = np.zeros(len(brand_subset))
            colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
            
            for sentiment in brand_subset.columns:
                ax5.bar(range(len(brand_subset)), brand_subset[sentiment], 
                       bottom=bottom, label=sentiment.title(), 
                       color=colors.get(sentiment, 'blue'), alpha=0.8)
                bottom += brand_subset[sentiment]
            
            ax5.set_title('Sentiment by Top 5 Brands', fontweight='bold')
            ax5.set_ylabel('Mention Count')
            ax5.set_xticks(range(len(brand_subset)))
            ax5.set_xticklabels(brand_subset.index, rotation=45, ha='right')
            ax5.legend()
        
        # 6. Engagement Analysis
        ax6 = fig.add_subplot(gs[1, 2])
        if 'total_engagement' in self.sentiment_df.columns:
            engagement_by_brand = self.sentiment_df.groupby('brand_mentioned')['total_engagement'].sum().nlargest(8)
            ax6.bar(range(len(engagement_by_brand)), engagement_by_brand.values, color='gold')
            ax6.set_title('Top Brands by Engagement', fontweight='bold')
            ax6.set_ylabel('Total Engagement')
            ax6.set_xticks(range(len(engagement_by_brand)))
            ax6.set_xticklabels(engagement_by_brand.index, rotation=45, ha='right')
        else:
            ax6.text(0.5, 0.5, 'No engagement data', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Brand Engagement', fontweight='bold')
        
        # Row 3: Time Trends
        # 7. Daily Mentions Trend
        ax7 = fig.add_subplot(gs[2, :])
        if 'timestamp' in self.sentiment_df.columns:
            daily_mentions = self.sentiment_df.groupby(self.sentiment_df['timestamp'].dt.date).size()
            recent_mentions = daily_mentions.tail(30)  # Last 30 days
            
            ax7.plot(range(len(recent_mentions)), recent_mentions.values, marker='o', 
                    color='red', linewidth=2, markersize=4)
            ax7.set_title('Daily Mentions Trend (Last 30 Days)', fontweight='bold')
            ax7.set_ylabel('Daily Mentions')
            ax7.set_xlabel('Days')
            ax7.grid(True, alpha=0.3)
            
            # Add trend line
            if len(recent_mentions) > 1:
                z = np.polyfit(range(len(recent_mentions)), recent_mentions.values, 1)
                p = np.poly1d(z)
                ax7.plot(range(len(recent_mentions)), p(range(len(recent_mentions))), 
                        "--", color='blue', alpha=0.8, label='Trend')
                ax7.legend()
        else:
            ax7.text(0.5, 0.5, 'No time data available', ha='center', va='center', 
                    transform=ax7.transAxes, fontsize=12)
            ax7.set_title('Daily Mentions Trend', fontweight='bold')
        
        # Row 4: Additional Metrics
        # 8. Influencer Impact
        ax8 = fig.add_subplot(gs[3, 0])
        if 'is_influencer' in self.sentiment_df.columns:
            influencer_mentions = self.sentiment_df[self.sentiment_df['is_influencer'] == True]
            if len(influencer_mentions) > 0:
                inf_brand_mentions = influencer_mentions['brand_mentioned'].value_counts().head(5)
                ax8.bar(range(len(inf_brand_mentions)), inf_brand_mentions.values, color='purple')
                ax8.set_title('Influencer Mentions by Brand', fontweight='bold')
                ax8.set_ylabel('Influencer Mentions')
                ax8.set_xticks(range(len(inf_brand_mentions)))
                ax8.set_xticklabels(inf_brand_mentions.index, rotation=45, ha='right')
            else:
                ax8.text(0.5, 0.5, 'No influencer data', ha='center', va='center', transform=ax8.transAxes)
                ax8.set_title('Influencer Impact', fontweight='bold')
        else:
            ax8.text(0.5, 0.5, 'No influencer data', ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('Influencer Impact', fontweight='bold')
        
        # 9. Verified Account Mentions
        ax9 = fig.add_subplot(gs[3, 1])
        if 'verified_account' in self.sentiment_df.columns:
            verified_dist = self.sentiment_df['verified_account'].value_counts()
            ax9.pie(verified_dist.values, labels=['Unverified', 'Verified'], autopct='%1.1f%%', 
                   colors=['lightblue', 'gold'], startangle=90)
            ax9.set_title('Verified vs Unverified Accounts', fontweight='bold')
        else:
            ax9.text(0.5, 0.5, 'No verification data', ha='center', va='center', transform=ax9.transAxes)
            ax9.set_title('Account Verification', fontweight='bold')
        
        # 10. Reach Potential
        ax10 = fig.add_subplot(gs[3, 2])
        if 'reach_potential' in self.sentiment_df.columns:
            reach_by_brand = self.sentiment_df.groupby('brand_mentioned')['reach_potential'].sum().nlargest(8)
            ax10.bar(range(len(reach_by_brand)), reach_by_brand.values, color='teal')
            ax10.set_title('Reach Potential by Brand', fontweight='bold')
            ax10.set_ylabel('Total Reach Potential')
            ax10.set_xticks(range(len(reach_by_brand)))
            ax10.set_xticklabels(reach_by_brand.index, rotation=45, ha='right')
        else:
            ax10.text(0.5, 0.5, 'No reach data', ha='center', va='center', transform=ax10.transAxes)
            ax10.set_title('Reach Potential', fontweight='bold')
        
        plt.tight_layout()
        self.save_figure(fig, 'social_listening_comprehensive_dashboard.png', 'Social Listening Comprehensive Dashboard')
        plt.show()
        
        return fig

    def generate_social_listening_report(self):
        """Generate comprehensive social listening report"""
        print("\n" + "="*80)
        print("    SOCIAL LISTENING DASHBOARD REPORT (NLTK-VADER)")
        print("="*80)
        
        # Run comprehensive analysis
        results = self.social_listening_comprehensive_dashboard()
        
        if results is None:
            print("Could not generate social listening report")
            return None
        
        # Print summary insights
        print("\n" + "="*80)
        print("              KEY SOCIAL INSIGHTS (NLTK-VADER)")
        print("="*80)
        
        print(f"📱 Total Social Mentions: {self.social_metrics['total_mentions']:,}")
        
        if self.social_metrics['brand_metrics']:
            brand_mentions = self.social_metrics['brand_metrics']['brand_mentions']
            print(f"🏢 Unique Brands Mentioned: {len(brand_mentions)}")
            top_brand = brand_mentions.index[0]
            top_brand_count = brand_mentions.iloc[0]
            print(f"🥇 Most Mentioned Brand: {top_brand} ({top_brand_count:,} mentions)")
        
        if self.social_metrics['sentiment_metrics']:
            sentiment_dist = self.social_metrics['sentiment_metrics']['sentiment_dist']
            if len(sentiment_dist) > 0:
                dominant_sentiment = sentiment_dist.idxmax()
                sentiment_pct = (sentiment_dist.max() / sentiment_dist.sum() * 100)
                print(f"💭 Dominant Sentiment: {dominant_sentiment.title()} ({sentiment_pct:.1f}%)")
                
                # Show NLTK-VADER specific insights
                if 'sentiment_scores' in self.social_metrics['sentiment_metrics'] and self.social_metrics['sentiment_metrics']['sentiment_scores'] is not None:
                    scores_df = self.social_metrics['sentiment_metrics']['sentiment_scores']
                    print(f"🎯 Average Sentiment Scores (VADER):")
                    for sentiment in scores_df.index:
                        avg_score = scores_df.loc[sentiment, 'mean']
                        print(f"  {sentiment.title()}: {avg_score:.3f}")
        
        platform_dist = self.sentiment_df['platform'].value_counts()
        top_platform = platform_dist.index[0]
        platform_count = platform_dist.iloc[0]
        print(f"📱 Most Active Platform: {top_platform} ({platform_count:,} mentions)")
        
        if self.social_metrics['hashtag_metrics']:
            hashtag_counts = self.social_metrics['hashtag_metrics']['hashtag_counts']
            if len(hashtag_counts) > 0:
                top_hashtag = hashtag_counts.index[0]
                hashtag_count = hashtag_counts.iloc[0]
                print(f"#️⃣ Top Hashtag: {top_hashtag} ({hashtag_count:,} uses)")
        
        print(f"\n📊 Total Visualizations Created: {len([f for f in os.listdir(self.vis_dir) if f.endswith('.png')])}")
        print(f"📁 Visualization Directory: {self.vis_dir}")
        
        # Actionable insights
        print("\n" + "="*80)
        print("         ACTIONABLE SOCIAL LISTENING INSIGHTS")
        print("="*80)
        
        print("🔥 IMMEDIATE ACTIONS:")
        print("1. Monitor and respond to negative sentiment mentions immediately")
        print("2. Engage with positive mentions to build brand advocacy")
        print("3. Track competitor mentions for competitive intelligence")
        print("4. Leverage trending hashtags for content marketing")
        
        print("\n📱 PLATFORM-SPECIFIC STRATEGIES:")
        print("1. Focus marketing efforts on highest-volume platforms")
        print("2. Tailor content strategy for each platform's audience")
        print("3. Monitor emerging platforms for early brand presence")
        
        print("\n💭 SENTIMENT-DRIVEN OPTIMIZATIONS (NLTK-VADER):")
        print("1. Address negative sentiment patterns proactively")
        print("2. Amplify positive sentiment through social proof")
        print("3. Use VADER compound scores to prioritize response urgency")
        print("4. Develop crisis management protocols for sentiment drops")
        
        print("\n🎯 HASHTAG & TREND OPPORTUNITIES:")
        print("1. Create content around trending hashtags")
        print("2. Monitor hashtag performance for campaign optimization")
        print("3. Identify brand-specific hashtag opportunities")
        print("4. Track hashtag sentiment evolution over time")
        
        print("\n🤖 NLTK-VADER SPECIFIC BENEFITS:")
        print("1. Real-time sentiment scoring without external APIs")
        print("2. Compound scores provide nuanced sentiment measurement")
        print("3. Handles social media text patterns effectively")
        print("4. Consistent and reproducible sentiment analysis")
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize Social Listening Dashboard with NLTK-VADER
    social_dashboard = SocialListeningDashboard(
        sentiment_csv_path=r'D:\CRM_PROJECT\data\sentiment.csv'
    )
    
    # Generate comprehensive social listening report
    results = social_dashboard.generate_social_listening_report()
    
    if results is not None:
        print("\n✅ Social Listening Dashboard Analysis Complete!")
        print("🔬 Sentiment Analysis: NLTK-VADER (Real-time NLP)")
        print("📊 All visualizations saved to: D:\CRM_PROJECT\visulization")
        
        print("\n" + "="*80)
        print("            NEXT STEPS & RECOMMENDATIONS")
        print("="*80)
        print("1. Schedule automated sentiment monitoring")
        print("2. Set up alerts for negative sentiment spikes")  
        print("3. Create response templates for different sentiment levels")
        print("4. Integrate findings with marketing campaign planning")
        print("5. Track sentiment score improvements over time")
    else:
        print("❌ Please ensure sentiment.csv exists with columns:")
        print("   - mention_id, timestamp, platform, brand_mentioned")
        print("   - mention_text (for NLTK sentiment analysis)")
        print("   - hashtags, total_engagement (optional)")
