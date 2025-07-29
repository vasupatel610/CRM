import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium # Make sure folium is installed: pip install folium
from folium.plugins import HeatMap
import base64
from io import BytesIO
from datetime import datetime, timedelta
import calendar

def load_and_process_data(file_path):
    """Load and preprocess the CRM data"""
    df = pd.read_csv(file_path)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['month'] = df['transaction_date'].dt.month
    df['year'] = df['transaction_date'].dt.year
    df['month_year'] = df['transaction_date'].dt.to_period('M')
    # Add month_name column
    df['month_name'] = df['transaction_date'].dt.strftime('%b %Y') # e.g., 'Jan 2023'
    return df

def create_sales_trendline(df):
    """Create MTD/QTD/YTD sales trendline"""
    # Group by month and calculate total sales
    monthly_sales = df.groupby('month_year').agg({
        'grand_total': 'sum',
        'month_name': 'first' # Get the month name for display
    }).reset_index()
    monthly_sales = monthly_sales.sort_values('month_year') # Ensure correct order
    monthly_sales['month_year_str'] = monthly_sales['month_year'].astype(str)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_sales['month_name'], # Use month name for display
        y=monthly_sales['grand_total'],
        mode='lines+markers',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.1)'
    ))

    fig.update_layout(
        title="Total Sales (MTD/QTD/YTD) Value Trendline",
        xaxis_title="", # Removed x-axis title
        yaxis_title="Sales Value",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='white',
        showlegend=False,
        xaxis=dict(tickangle=45) # Keep tickangle for readability if many months
    )

    return fig.to_html(div_id="sales_trend", include_plotlyjs=True)

def create_sales_vs_target(df):
    """
    Create an enhanced Sales vs Target visualization showing:
    - Actual vs target sales amounts only
    - Last 6 months data only
    """

    # Step 1: Calculate monthly sales and targets
    monthly_data = df.groupby('month_year').agg({
        'grand_total': 'sum',
        'target_multiplier': 'first',  # Get target multiplier from data
        'month_name': 'first' # Get the month name for display
    }).reset_index()

    # Calculate realistic targets
    monthly_data['target_amount'] = monthly_data['grand_total'] * monthly_data['target_multiplier']

    # Step 2: Get last 6 months only
    monthly_data = monthly_data.sort_values('month_year')
    last_6_months = monthly_data.tail(6)

    # Step 3: Create the visualization
    fig = go.Figure()

    # Actual sales bars
    fig.add_trace(go.Bar(
        x=last_6_months['month_name'], # Use month name for display
        y=last_6_months['grand_total'],
        name='Actual Sales (KES)',
        marker_color='#3498db',
        opacity=0.8,
        showlegend=True
    ))

    # Target sales bars
    fig.add_trace(go.Bar(
        x=last_6_months['month_name'], # Use month name for display
        y=last_6_months['target_amount'],
        name='Target Sales (KES)',
        marker_color='#95a5a6',
        opacity=0.8,
        showlegend=True
    ))

    # Step 4: Configure layout for better visibility
    fig.update_layout(
        title={
            'text': "Actual vs Target Sales (Last 6 Months)",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#2c3e50'}
        },

        # Primary Y-axis (Sales amounts)
        yaxis=dict(
            title="Sales Amount (KES)",
            side="left",
            tickformat=',.0f',
            tickprefix='KES',
            gridcolor='rgba(128,128,128,0.2)',
            title_font={'color': '#2c3e50'},
            tickfont={'color': '#2c3e50'}
        ),

        # X-axis
        xaxis=dict(
            title="", # Removed x-axis title
            tickangle=45,
            title_font={'color': '#2c3e50'},
            tickfont={'color': '#2c3e50'}
        ),

        # General layout
        height=300,
        margin=dict(l=60, r=60, t=60, b=80),
        plot_bgcolor='rgba(248,249,250,0.8)',
        paper_bgcolor='white',

        # Legend
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(128,128,128,0.3)',
            borderwidth=1
        ),

        # Hover settings
        hovermode='x unified',
        barmode='group' # This ensures bars are grouped side-by-side for each month
    )

    return fig.to_html(div_id="sales_target", include_plotlyjs=False)

def create_transaction_count(df):
    """Create transaction count visualization"""
    # Count distinct transactions
    transaction_count = df.groupby('month_year').agg({
        'transaction_id': 'nunique',
        'month_name': 'first' # Get the month name for display
    }).reset_index()
    transaction_count = transaction_count.sort_values('month_year')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=transaction_count['month_name'], # Use month name for display
        y=transaction_count['transaction_id'],
        marker_color='#3498db',
        text=transaction_count['transaction_id'],
        textposition='outside'
    ))

    fig.update_layout(
        title="# of Transactions (distinct count of invoices)",
        xaxis_title="", # Removed x-axis title
        yaxis_title="Transaction Count",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='white',
        showlegend=False,
        xaxis=dict(tickangle=45)
    )

    return fig.to_html(div_id="transaction_count", include_plotlyjs=False)

def create_geo_heatmap(df):
    """Create geographical heatmap of top performing branches"""
    # Group by branch and calculate performance metrics
    branch_performance = df.groupby(['branch_id', 'latitude', 'longitude']).agg({
        'grand_total': 'sum',
        'transaction_id': 'nunique'
    }).reset_index()

    # Create folium map
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()

    try:
        import folium
        from folium.plugins import HeatMap
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

        # Add heat map
        heat_data = [[row['latitude'], row['longitude'], row['grand_total']]
                     for idx, row in branch_performance.iterrows()]
        HeatMap(heat_data).add_to(m)

        # Add markers for top branches
        top_branches = branch_performance.nlargest(5, 'grand_total')
        for idx, branch in top_branches.iterrows():
            folium.Marker(
                [branch['latitude'], branch['longitude']],
                popup=f"Branch: {branch['branch_id']}<br>Sales: KES{branch['grand_total']:,.2f}",
                tooltip=f"Top Branch: {branch['branch_id']}"
            ).add_to(m)

        # Convert map to HTML
        map_html = m._repr_html_()
        return f'<div style="height: 250px; overflow: hidden;">{map_html}</div>'
    except ImportError:
        return "<p>Folium library not found. Please install it to view the geographical heatmap.</p>"

def create_sales_by_channel(df):
    """Create sales by channel bar chart"""
    channel_sales = df.groupby('channel_id')['grand_total'].sum().reset_index()
    channel_sales = channel_sales.sort_values('grand_total', ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=channel_sales['channel_id'],
        x=channel_sales['grand_total'],
        orientation='h',
        marker_color=['#e74c3c', '#f39c12', '#2ecc71'],
        text=[f"KES{x:,.0f}" for x in channel_sales['grand_total']],
        textposition='outside'
    ))

    fig.update_layout(
        title="Sales by Channel",
        xaxis_title="Sales Value",
        yaxis_title="Channel",
        height=250,
        margin=dict(l=80, r=20, t=40, b=20),
        plot_bgcolor='white',
        showlegend=False
    )

    return fig.to_html(div_id="sales_channel", include_plotlyjs=False)

def create_top_products_pie(df):
    """Create top 5 products pie chart"""
    product_sales = df.groupby('product_name')['grand_total'].sum().reset_index()
    top_products = product_sales.nlargest(5, 'grand_total')

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=top_products['product_name'],
        values=top_products['grand_total'],
        hole=0.4,
        textinfo='label+percent',
        textposition='outside'
    ))

    fig.update_layout(
        title="Top 5 Products",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
    )

    return fig.to_html(div_id="top_products", include_plotlyjs=False)

def create_return_rate_trend(df):
    """Create return rate trendline"""
    monthly_returns = df.groupby('month_year').agg({
        'is_returned': lambda x: (x == 'Yes').sum(),
        'transaction_line_id': 'count',
        'month_name': 'first' # Get the month name for display
    }).reset_index()
    monthly_returns = monthly_returns.sort_values('month_year')

    monthly_returns['return_rate'] = (monthly_returns['is_returned'] / monthly_returns['transaction_line_id']) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_returns['month_name'], # Use month name for display
        y=monthly_returns['return_rate'],
        mode='lines+markers',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=8),
        fill='tonexty',
        fillcolor='rgba(231, 76, 60, 0.1)'
    ))

    fig.update_layout(
        title="Return Rate Trendline",
        xaxis_title="", # Removed x-axis title
        yaxis_title="Return Rate (%)",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='white',
        showlegend=False,
        xaxis=dict(tickangle=45)
    )

    return fig.to_html(div_id="return_rate", include_plotlyjs=False)

def create_loyalty_heatmap(df):
    """Create loyalty engagement heatmap per store per customer age bin using loyalty card percentage"""
    
    # Step 1: Create age bins
    df['age_bin'] = pd.cut(df['customer_age'], 
                          bins=[0, 27, 37, 47, 57, 67, 77, 100], 
                          labels=['18-27', '28-37', '38-47', '48-57', '58-67', '68-77', '78-80'],
                          right=False)
    
    # Step 2: Identify loyalty card transactions (assuming voucher_redeemed_flag indicates loyalty card usage)
    # You can modify this condition based on your loyalty card identification logic
    df['has_loyalty_card'] = df['voucher_redeemed_flag'] == 'Yes'
    
    # Step 3: Calculate loyalty engagement percentage by store and age bin
    loyalty_data = df.groupby(['branch_id', 'age_bin']).agg({
        'has_loyalty_card': 'sum',  # Count of transactions with loyalty card
        'transaction_line_id': 'count',  # Total transactions
        'latitude': 'first',  # Get branch coordinates
        'longitude': 'first'
    }).reset_index()
    
    # Calculate loyalty engagement percentage
    loyalty_data['loyalty_percentage'] = (loyalty_data['has_loyalty_card'] / loyalty_data['transaction_line_id']) * 100
    
    # Handle NaN values
    loyalty_data['loyalty_percentage'] = loyalty_data['loyalty_percentage'].fillna(0)
    
    # Step 4: Create pivot table for heatmap data (store_id vs age_bin)
    heatmap_data = loyalty_data.pivot(index='branch_id', columns='age_bin', values='loyalty_percentage')
    heatmap_data = heatmap_data.fillna(0)  # Fill NaN with 0
    
    # Step 5: Create Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdYlBu_r',  # Red-Yellow-Blue reversed (red for high values)
        colorbar=dict(title="Loyalty Engagement %"),
        text=heatmap_data.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 8},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Loyalty Engagement Percentage Heatmap<br><sub>Store ID vs Customer Age Bin</sub>",
        xaxis_title="Customer Age Bin",
        yaxis_title="Store ID",
        height=250,
        margin=dict(l=60, r=20, t=60, b=40),
        plot_bgcolor='white',
        font=dict(size=10)
    )
    
    return fig.to_html(div_id="loyalty_heatmap", include_plotlyjs=False)

def create_bnpl_adoption_boxplot(df):
    """Create BNPL adoption rate monthly boxplot"""
    # Calculate BNPL adoption by month and branch
    bnpl_data = df.groupby(['month_year', 'branch_id']).agg({
        'payment_mode': lambda x: (x == 'BNPL').sum(),
        'transaction_line_id': 'count',
        'month_name': 'first' # Get the month name for display
    }).reset_index()
    bnpl_data = bnpl_data.sort_values('month_year')

    bnpl_data['bnpl_rate'] = (bnpl_data['payment_mode'] / bnpl_data['transaction_line_id']) * 100

    fig = go.Figure()

    for month_year_period in bnpl_data['month_year'].unique():
        month_data = bnpl_data[bnpl_data['month_year'] == month_year_period]
        # Get the representative month_name for this month_year_period
        month_name_for_plot = month_data['month_name'].iloc[0] if not month_data.empty else str(month_year_period)
        fig.add_trace(go.Box(
            y=month_data['bnpl_rate'],
            name=month_name_for_plot, # Use month name for the boxplot group
            boxpoints='outliers'
        ))

    fig.update_layout(
        title="BNPL Adoption Rate (Monthly Adoption Rate Boxplot)",
        xaxis_title="", # Removed x-axis title
        yaxis_title="BNPL Adoption Rate (%)",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='white',
        showlegend=False,
        xaxis=dict(tickangle=45)
    )

    return fig.to_html(div_id="bnpl_adoption", include_plotlyjs=False)

def generate_dashboard_html(df):
    """Generate the complete 3x3 dashboard HTML"""

    # Generate all chart components
    sales_trend = create_sales_trendline(df)
    sales_target = create_sales_vs_target(df)
    transaction_count = create_transaction_count(df)
    geo_heatmap = create_geo_heatmap(df)
    sales_channel = create_sales_by_channel(df)
    top_products = create_top_products_pie(df)
    return_rate = create_return_rate_trend(df)
    loyalty_heatmap = create_loyalty_heatmap(df)
    bnpl_adoption = create_bnpl_adoption_boxplot(df)

    # Create the HTML template
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CRM Sales Dashboard</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .dashboard {{
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                grid-template-rows: 1fr 1fr 1fr;
                gap: 15px;
                height: 100vh;
                max-height: 900px;
            }}
            .widget {{
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 15px;
                overflow: hidden;
                display: flex;
                flex-direction: column;
            }}
            .widget h3 {{
                margin: 0 0 10px 0;
                color: #333;
                font-size: 14px;
                font-weight: bold;
            }}
            .chart-container {{
                flex: 1;
                min-height: 0;
            }}
            .blue-bg {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }}
            .blue-bg h3 {{
                color: white;
            }}
            @media (max-width: 1200px) {{
                .dashboard {{
                    grid-template-columns: 1fr 1fr;
                    height: auto;
                }}
            }}
            @media (max-width: 768px) {{
                .dashboard {{
                    grid-template-columns: 1fr;
                    height: auto;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="dashboard">
            <div class="widget blue-bg">
                <h3>Total Sales (MTD/QTD/YTD) Value<br>Trendline as backdrop</h3>
                <div class="chart-container">
                    {sales_trend}
                </div>
            </div>

            <div class="widget blue-bg">
                <h3>Sales vs Target %<br>Last 6 months two bar per one X</h3>
                <div class="chart-container">
                    {sales_target}
                </div>
            </div>

            <div class="widget blue-bg">
                <h3># of Transactions<br>distinct count of invoices</h3>
                <div class="chart-container">
                    {transaction_count}
                </div>
            </div>

            <div class="widget">
                <h3>Top Performing Branches<br>geo view</h3>
                <div class="chart-container">
                    {geo_heatmap}
                </div>
            </div>

            <div class="widget">
                <h3>Sales by Channel<br>Bar chart</h3>
                <div class="chart-container">
                    {sales_channel}
                </div>
            </div>

            <div class="widget">
                <h3>Top 5 Products<br>Pie chart</h3>
                <div class="chart-container">
                    {top_products}
                </div>
            </div>

            <div class="widget">
                <h3>Return Rate<br>trendline</h3>
                <div class="chart-container">
                    {return_rate}
                </div>
            </div>

            <div class="widget">
                <h3>Loyalty Engagement<br>per store per customer age bin</h3>
                <div class="chart-container">
                    {loyalty_heatmap}
                </div>
            </div>

            <div class="widget">
                <h3>BNPL Adoption Rate<br>monthwise adoption rate boxplot</h3>
                <div class="chart-container">
                    {bnpl_adoption}
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    return html_template

def main():
    """Main function to generate the dashboard"""
    # File path - update this to your actual file location
    file_path = r"D:\Mockup\synthetic_transaction_data.csv"

    try:
        # Load and process data
        print("Loading data...")
        df = load_and_process_data(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")

        # Generate dashboard HTML
        print("Generating dashboard...")
        dashboard_html = generate_dashboard_html(df)

        # Save to file
        output_file = "crm_sales_dashboard.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)

        print(f"Dashboard generated successfully! Open '{output_file}' in your browser to view.")

    except FileNotFoundError:
        print(f"Error: Could not find the file at {file_path}")
        print("Please make sure the file path is correct.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()


# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots
# import folium # Make sure folium is installed: pip install folium
# from folium.plugins import HeatMap
# import base64
# from io import BytesIO
# from datetime import datetime, timedelta
# import calendar

# def load_and_process_data(file_path):
#     """Load and preprocess the CRM data"""
#     df = pd.read_csv(file_path)
#     df['transaction_date'] = pd.to_datetime(df['transaction_date'])
#     df['month'] = df['transaction_date'].dt.month
#     df['year'] = df['transaction_date'].dt.year
#     df['month_year'] = df['transaction_date'].dt.to_period('M')
#     # Add month_name column
#     df['month_name'] = df['transaction_date'].dt.strftime('%b %Y') # e.g., 'Jan 2023'
#     return df

# def create_sales_trendline(df):
#     """Create MTD/QTD/YTD sales trendline"""
#     # Group by month and calculate total sales
#     monthly_sales = df.groupby('month_year').agg({
#         'grand_total': 'sum',
#         'month_name': 'first' # Get the month name for display
#     }).reset_index()
#     monthly_sales = monthly_sales.sort_values('month_year') # Ensure correct order
#     monthly_sales['month_year_str'] = monthly_sales['month_year'].astype(str)

#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=monthly_sales['month_name'], # Use month name for display
#         y=monthly_sales['grand_total'],
#         mode='lines+markers',
#         line=dict(color='#1f77b4', width=3),
#         marker=dict(size=8),
#         fill='tonexty',
#         fillcolor='rgba(31, 119, 180, 0.1)'
#     ))

#     fig.update_layout(
#         title="Total Sales (MTD/QTD/YTD) Value Trendline",
#         xaxis_title="", # Removed x-axis title
#         yaxis_title="Sales Value",
#         height=250,
#         margin=dict(l=20, r=20, t=40, b=20),
#         plot_bgcolor='white',
#         showlegend=False,
#         xaxis=dict(tickangle=45) # Keep tickangle for readability if many months
#     )

#     return fig.to_html(div_id="sales_trend", include_plotlyjs=True)

# def create_sales_vs_target(df):
#     """
#     Create an enhanced Sales vs Target visualization showing:
#     - Actual vs target sales amounts only
#     - Last 6 months data only
#     """

#     # Step 1: Calculate monthly sales and targets
#     monthly_data = df.groupby('month_year').agg({
#         'grand_total': 'sum',
#         'target_multiplier': 'first',  # Get target multiplier from data
#         'month_name': 'first' # Get the month name for display
#     }).reset_index()

#     # Calculate realistic targets
#     monthly_data['target_amount'] = monthly_data['grand_total'] * monthly_data['target_multiplier']

#     # Step 2: Get last 6 months only
#     monthly_data = monthly_data.sort_values('month_year')
#     last_6_months = monthly_data.tail(6)

#     # Step 3: Create the visualization
#     fig = go.Figure()

#     # Actual sales bars
#     fig.add_trace(go.Bar(
#         x=last_6_months['month_name'], # Use month name for display
#         y=last_6_months['grand_total'],
#         name='Actual Sales (KES)',
#         marker_color='#3498db',
#         opacity=0.8,
#         showlegend=True
#     ))

#     # Target sales bars
#     fig.add_trace(go.Bar(
#         x=last_6_months['month_name'], # Use month name for display
#         y=last_6_months['target_amount'],
#         name='Target Sales (KES)',
#         marker_color='#95a5a6',
#         opacity=0.8,
#         showlegend=True
#     ))

#     # Step 4: Configure layout for better visibility
#     fig.update_layout(
#         title={
#             'text': "Actual vs Target Sales (Last 6 Months)",
#             'x': 0.5,
#             'xanchor': 'center',
#             'font': {'size': 16, 'color': '#2c3e50'}
#         },

#         # Primary Y-axis (Sales amounts)
#         yaxis=dict(
#             title="Sales Amount (KES)",
#             side="left",
#             tickformat=',.0f',
#             tickprefix='KES',
#             gridcolor='rgba(128,128,128,0.2)',
#             title_font={'color': '#2c3e50'},
#             tickfont={'color': '#2c3e50'}
#         ),

#         # X-axis
#         xaxis=dict(
#             title="", # Removed x-axis title
#             tickangle=45,
#             title_font={'color': '#2c3e50'},
#             tickfont={'color': '#2c3e50'}
#         ),

#         # General layout
#         height=300,
#         margin=dict(l=60, r=60, t=60, b=80),
#         plot_bgcolor='rgba(248,249,250,0.8)',
#         paper_bgcolor='white',

#         # Legend
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=-0.3,
#             xanchor="center",
#             x=0.5,
#             bgcolor='rgba(255,255,255,0.8)',
#             bordercolor='rgba(128,128,128,0.3)',
#             borderwidth=1
#         ),

#         # Hover settings
#         hovermode='x unified',
#         barmode='group' # This ensures bars are grouped side-by-side for each month
#     )

#     return fig.to_html(div_id="sales_target", include_plotlyjs=False)


# def create_transaction_count(df):
#     """Create transaction count visualization"""
#     # Count distinct transactions
#     transaction_count = df.groupby('month_year').agg({
#         'transaction_id': 'nunique',
#         'month_name': 'first' # Get the month name for display
#     }).reset_index()
#     transaction_count = transaction_count.sort_values('month_year')

#     fig = go.Figure()
#     fig.add_trace(go.Bar(
#         x=transaction_count['month_name'], # Use month name for display
#         y=transaction_count['transaction_id'],
#         marker_color='#3498db',
#         text=transaction_count['transaction_id'],
#         textposition='outside'
#     ))

#     fig.update_layout(
#         title="# of Transactions (distinct count of invoices)",
#         xaxis_title="", # Removed x-axis title
#         yaxis_title="Transaction Count",
#         height=250,
#         margin=dict(l=20, r=20, t=40, b=20),
#         plot_bgcolor='white',
#         showlegend=False,
#         xaxis=dict(tickangle=45)
#     )

#     return fig.to_html(div_id="transaction_count", include_plotlyjs=False)

# def create_geo_heatmap(df):
#     """Create geographical heatmap of top performing branches"""
#     # Group by branch and calculate performance metrics
#     branch_performance = df.groupby(['branch_id', 'latitude', 'longitude']).agg({
#         'grand_total': 'sum',
#         'transaction_id': 'nunique'
#     }).reset_index()

#     # Create folium map
#     center_lat = df['latitude'].mean()
#     center_lon = df['longitude'].mean()

#     try:
#         import folium
#         from folium.plugins import HeatMap
#         m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

#         # Add heat map
#         heat_data = [[row['latitude'], row['longitude'], row['grand_total']]
#                      for idx, row in branch_performance.iterrows()]
#         HeatMap(heat_data).add_to(m)

#         # Add markers for top branches
#         top_branches = branch_performance.nlargest(5, 'grand_total')
#         for idx, branch in top_branches.iterrows():
#             folium.Marker(
#                 [branch['latitude'], branch['longitude']],
#                 popup=f"Branch: {branch['branch_id']}<br>Sales: KES{branch['grand_total']:,.2f}",
#                 tooltip=f"Top Branch: {branch['branch_id']}"
#             ).add_to(m)

#         # Convert map to HTML
#         map_html = m._repr_html_()
#         return f'<div style="height: 250px; overflow: hidden;">{map_html}</div>'
#     except ImportError:
#         return "<p>Folium library not found. Please install it to view the geographical heatmap.</p>"


# def create_sales_by_channel(df):
#     """Create sales by channel bar chart"""
#     channel_sales = df.groupby('channel_id')['grand_total'].sum().reset_index()
#     channel_sales = channel_sales.sort_values('grand_total', ascending=True)

#     fig = go.Figure()
#     fig.add_trace(go.Bar(
#         y=channel_sales['channel_id'],
#         x=channel_sales['grand_total'],
#         orientation='h',
#         marker_color=['#e74c3c', '#f39c12', '#2ecc71'],
#         text=[f"KES{x:,.0f}" for x in channel_sales['grand_total']],
#         textposition='outside'
#     ))

#     fig.update_layout(
#         title="Sales by Channel",
#         xaxis_title="Sales Value",
#         yaxis_title="Channel",
#         height=250,
#         margin=dict(l=80, r=20, t=40, b=20),
#         plot_bgcolor='white',
#         showlegend=False
#     )

#     return fig.to_html(div_id="sales_channel", include_plotlyjs=False)

# def create_top_products_pie(df):
#     """Create top 5 products pie chart"""
#     product_sales = df.groupby('product_name')['grand_total'].sum().reset_index()
#     top_products = product_sales.nlargest(5, 'grand_total')

#     fig = go.Figure()
#     fig.add_trace(go.Pie(
#         labels=top_products['product_name'],
#         values=top_products['grand_total'],
#         hole=0.4,
#         textinfo='label+percent',
#         textposition='outside'
#     ))

#     fig.update_layout(
#         title="Top 5 Products",
#         height=250,
#         margin=dict(l=20, r=20, t=40, b=20),
#         showlegend=True,
#         legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
#     )

#     return fig.to_html(div_id="top_products", include_plotlyjs=False)

# def create_return_rate_trend(df):
#     """Create return rate trendline"""
#     monthly_returns = df.groupby('month_year').agg({
#         'is_returned': lambda x: (x == 'Yes').sum(),
#         'transaction_line_id': 'count',
#         'month_name': 'first' # Get the month name for display
#     }).reset_index()
#     monthly_returns = monthly_returns.sort_values('month_year')


#     monthly_returns['return_rate'] = (monthly_returns['is_returned'] / monthly_returns['transaction_line_id']) * 100

#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=monthly_returns['month_name'], # Use month name for display
#         y=monthly_returns['return_rate'],
#         mode='lines+markers',
#         line=dict(color='#e74c3c', width=3),
#         marker=dict(size=8),
#         fill='tonexty',
#         fillcolor='rgba(231, 76, 60, 0.1)'
#     ))

#     fig.update_layout(
#         title="Return Rate Trendline",
#         xaxis_title="", # Removed x-axis title
#         yaxis_title="Return Rate (%)",
#         height=250,
#         margin=dict(l=20, r=20, t=40, b=20),
#         plot_bgcolor='white',
#         showlegend=False,
#         xaxis=dict(tickangle=45)
#     )

#     return fig.to_html(div_id="return_rate", include_plotlyjs=False)

# def create_loyalty_heatmap(df):
#     """Create loyalty engagement heatmap per store on geographical map using Customer Lifetime Value (CLV)."""

#     # Step 0: Create a definitive branch location mapping
#     # Ensure unique branch_id, latitude, longitude sets.
#     # If there are duplicate branch_ids with different lat/lon, this takes the first one found.
#     branch_locations = df[['branch_id', 'latitude', 'longitude']].drop_duplicates(subset=['branch_id'])

#     # Step 1: Calculate customer-level metrics
#     customer_summary = df.groupby('customer_id').agg(
#         first_purchase_date=('transaction_date', 'min'),
#         last_purchase_date=('transaction_date', 'max'),
#         total_spend=('grand_total', 'sum'),
#         total_transactions=('transaction_id', 'nunique'),
#         # Assign the customer to the branch where they had the most transactions
#         primary_branch_id=('branch_id', lambda x: x.value_counts().idxmax())
#     ).reset_index()

#     # Merge branch coordinates to customer_summary based on primary_branch_id
#     # The 'branch_id' column from branch_locations will be added to customer_summary.
#     # We will then use this 'branch_id' for subsequent grouping.
#     customer_summary = pd.merge(
#         customer_summary,
#         branch_locations,
#         left_on='primary_branch_id',
#         right_on='branch_id',
#         how='left'
#     )

#     # Now, drop the 'primary_branch_id' if you prefer to use the 'branch_id' from branch_locations
#     # as the consistent branch identifier for the customer.
#     customer_summary.drop(columns=['primary_branch_id'], inplace=True)


#     # Calculate Average Purchase Value for each customer
#     customer_summary['average_purchase_value'] = customer_summary['total_spend'] / customer_summary['total_transactions']

#     # Calculate Customer Lifespan in days
#     customer_summary['customer_lifespan_days'] = (customer_summary['last_purchase_date'] - customer_summary['first_purchase_date']).dt.days

#     # Handle single-purchase customers for lifespan (set lifespan to 1 day to avoid division by zero later)
#     customer_summary['customer_lifespan_days'] = customer_summary['customer_lifespan_days'].replace(0, 1)

#     # Calculate Average Purchase Frequency (transactions per day of lifespan)
#     customer_summary['average_purchase_frequency'] = np.where(
#         customer_summary['customer_lifespan_days'] > 0,
#         customer_summary['total_transactions'] / customer_summary['customer_lifespan_days'],
#         customer_summary['total_transactions'] # If lifespan is 0/1, frequency is just total transactions (e.g., 1 trans in 1 day)
#     )

#     # Calculate CLV for each customer
#     customer_summary['clv'] = customer_summary['average_purchase_value'] * \
#                                   customer_summary['average_purchase_frequency'] * \
#                                   customer_summary['customer_lifespan_days']

#     # Step 2: Aggregate average CLV by branch (using the unique branch_id from branch_locations)
#     # Ensure 'branch_id', 'latitude', 'longitude' are used as grouping keys.
#     # These columns are now correctly in `customer_summary` after the merge.
#     loyalty_data = customer_summary.groupby(['branch_id', 'latitude', 'longitude']).agg(
#         average_clv=('clv', 'mean'),
#         total_unique_customers=('customer_id', 'nunique'),
#         total_grand_total=('total_spend', 'sum'), # Total sales through customers assigned to this branch
#         total_transactions_branch=('total_transactions', 'sum') # Total transactions through customers assigned to this branch
#     ).reset_index()

#     # Handle cases where a branch might have no customers calculated, resulting in NaN CLV
#     loyalty_data['average_clv'] = loyalty_data['average_clv'].fillna(0)

#     # Get min and max CLV for dynamic color scaling
#     min_clv = loyalty_data['average_clv'].min()
#     max_clv = loyalty_data['average_clv'].max()

#     try:
#         import folium
#         from folium.plugins import HeatMap

#         # Create folium map centered on data
#         center_lat = loyalty_data['latitude'].mean()
#         center_lon = loyalty_data['longitude'].mean()

#         m = folium.Map(
#             location=[center_lat, center_lon],
#             zoom_start=7,
#             tiles='OpenStreetMap'
#         )

#         # Dynamic color scale based on actual data range
#         def get_color_dynamic(clv_value, min_val, max_val):
#             if max_val == min_val:  # Handle edge case where all values are the same
#                 return '#FFD700' # Default to a neutral color if no range

#             # Normalize the CLV value to 0-1 scale
#             normalized = (clv_value - min_val) / (max_val - min_val)

#             # Create gradient from red (lowest) to green (highest)
#             if normalized >= 0.8:
#                 return '#006400'  # Dark Green - Highest 20%
#             elif normalized >= 0.6:
#                 return '#228B22'  # Forest Green - High 20%
#             elif normalized >= 0.4:
#                 return '#32CD32'  # Lime Green - Middle-High 20%
#             elif normalized >= 0.2:
#                 return '#FFD700'  # Gold - Middle-Low 20%
#             else:
#                 return '#FF4500'  # Red Orange - Lowest 20%

#         # Sort data by CLV to get ranking
#         loyalty_data_sorted = loyalty_data.sort_values('average_clv', ascending=False).reset_index(drop=True)
#         loyalty_data_sorted['rank'] = loyalty_data_sorted.index + 1
#         loyalty_data_sorted['percentile'] = (loyalty_data_sorted['rank'] / len(loyalty_data_sorted)) * 100

#         # Add circle markers for each store with dynamic color coding
#         for idx, store in loyalty_data_sorted.iterrows():
#             # Create popup content with ranking
#             popup_content = f"""
#             <div style="font-family: Arial; min-width: 220px;">
#                 <h4 style="margin: 0; color: #333;">🏪 {store['branch_id']}</h4>
#                 <hr style="margin: 5px 0;">
#                 <p style="margin: 2px 0;"><b>📊 Average CLV:</b> KES{store['average_clv']:,.2f}</p>
#                 <p style="margin: 2px 0;"><b>🏆 Rank:</b> #{store['rank']} of {len(loyalty_data_sorted)}</p>
#                 <p style="margin: 2px 0;"><b>📈 Percentile:</b> Top {store['percentile']:.0f}%</p>
#                 <p style="margin: 2px 0;"><b>🛒 Total Unique Customers:</b> {int(store['total_unique_customers'])}</p>
#                 <p style="margin: 2px 0;"><b>💰 Total Branch Sales:</b> KES{store['total_grand_total']:,.0f}</p>
#                 <p style="margin: 2px 0;"><b>📋 Total Branch Transactions:</b> {int(store['total_transactions_branch'])}</p>
#             </div>
#             """

#             # Calculate marker size based on total unique customers at the branch
#             max_customers_size = loyalty_data['total_unique_customers'].max()
#             min_customers_size = loyalty_data['total_unique_customers'].min()
#             # Avoid division by zero if all customer counts are the same
#             normalized_size = (store['total_unique_customers'] - min_customers_size) / (max_customers_size - min_customers_size) if (max_customers_size - min_customers_size) != 0 else 0.5
#             marker_size = 8 + (normalized_size * 20)  # Size range from 8 to 28

#             # Add CircleMarker for visual representation (color based on CLV, size based on unique customers)
#             folium.CircleMarker(
#                 location=[store['latitude'], store['longitude']],
#                 radius=marker_size,
#                 popup=folium.Popup(popup_content, max_width=300),
#                 tooltip=f"Store: {store['branch_id']} | Avg CLV: KES{store['average_clv']:,.0f} (Rank #{store['rank']})",
#                 color='white', # Border color
#                 weight=2,
#                 fillColor=get_color_dynamic(store['average_clv'], min_clv, max_clv),
#                 fillOpacity=0.8
#             ).add_to(m)

#             # Add a separate Marker with DivIcon for the branch_id label
#             folium.Marker(
#                 location=[store['latitude'], store['longitude']],
#                 icon=folium.DivIcon(
#                     html=f'<div style="font-size: 10px; font-weight: bold; color: white; text-shadow: 1px 1px 1px black;">{store["branch_id"]}</div>',
#                     icon_size=(50, 20), # Adjust size as needed
#                     icon_anchor=(25, 10) # Adjust anchor to center the text
#                 )
#             ).add_to(m)

#         # Create dynamic legend based on actual data quantiles
#         # Check if min_clv == max_clv to prevent division by zero in quantile calculations
#         if min_clv == max_clv:
#             # If all CLV values are the same, simplify the legend
#             legend_html = f'''
#             <div style="position: fixed;
#                         bottom: 50px; left: 50px; width: 220px; height: 100px;
#                         background-color: white; border:2px solid grey; z-index:9999;
#                         font-size:11px; font-family: Arial;
#                         padding: 10px; border-radius: 5px;">
#             <h4 style="margin: 0 0 8px 0;">Average CLV</h4>
#             <p style="margin: 2px 0;"><span style="color: {get_color_dynamic(min_clv, min_clv, max_clv)}; font-size: 14px;">●</span> All Stores: KES{min_clv:,.0f}</p>
#             <small><i>Circle size = # of unique customers</i></small>
#             </div>
#             '''
#         else:
#             quintile_1 = loyalty_data['average_clv'].quantile(0.8)
#             quintile_2 = loyalty_data['average_clv'].quantile(0.6)
#             quintile_3 = loyalty_data['average_clv'].quantile(0.4)
#             quintile_4 = loyalty_data['average_clv'].quantile(0.2)

#             legend_html = f'''
#             <div style="position: fixed;
#                         bottom: 50px; left: 50px; width: 220px; height: 140px;
#                         background-color: white; border:2px solid grey; z-index:9999;
#                         font-size:11px; font-family: Arial;
#                         padding: 10px; border-radius: 5px;">
#             <h4 style="margin: 0 0 8px 0;">Average CLV (Dynamic Range)</h4>
#             <p style="margin: 2px 0;"><span style="color: #006400; font-size: 14px;">●</span> KES{quintile_1:,.0f}+ (Top 20%)</p>
#             <p style="margin: 2px 0;"><span style="color: #228B22; font-size: 14px;">●</span> KES{quintile_2:,.0f} - KES{quintile_1:,.0f} (High)</p>
#             <p style="margin: 2px 0;"><span style="color: #32CD32; font-size: 14px;">●</span> KES{quintile_3:,.0f} - KES{quintile_2:,.0f} (Medium)</p>
#             <p style="margin: 2px 0;"><span style="color: #FFD700; font-size: 14px;">●</span> KES{quintile_4:,.0f} - KES{quintile_3:,.0f} (Low)</p>
#             <p style="margin: 2px 0;"><span style="color: #FF4500; font-size: 14px;">●</span> KES{min_clv:,.0f} - KES{quintile_4:,.0f} (Bottom 20%)</p>
#             <small><i>Circle size = # of unique customers</i></small>
#             </div>
#             '''
#         m.get_root().html.add_child(folium.Element(legend_html))

#         # Convert map to HTML and wrap in container
#         map_html = m._repr_html_()
#         return f'<div style="height: 220px; overflow: hidden; border-radius: 5px;">{map_html}</div>'
#     except ImportError:
#         return "<p>Folium library not found. Please install it to view the loyalty engagement heatmap.</p>"

# def create_bnpl_adoption_boxplot(df):
#     """Create BNPL adoption rate monthly boxplot"""
#     # Calculate BNPL adoption by month and branch
#     bnpl_data = df.groupby(['month_year', 'branch_id']).agg({
#         'payment_mode': lambda x: (x == 'BNPL').sum(),
#         'transaction_line_id': 'count',
#         'month_name': 'first' # Get the month name for display
#     }).reset_index()
#     bnpl_data = bnpl_data.sort_values('month_year')


#     bnpl_data['bnpl_rate'] = (bnpl_data['payment_mode'] / bnpl_data['transaction_line_id']) * 100

#     fig = go.Figure()

#     for month_year_period in bnpl_data['month_year'].unique():
#         month_data = bnpl_data[bnpl_data['month_year'] == month_year_period]
#         # Get the representative month_name for this month_year_period
#         month_name_for_plot = month_data['month_name'].iloc[0] if not month_data.empty else str(month_year_period)
#         fig.add_trace(go.Box(
#             y=month_data['bnpl_rate'],
#             name=month_name_for_plot, # Use month name for the boxplot group
#             boxpoints='outliers'
#         ))

#     fig.update_layout(
#         title="BNPL Adoption Rate (Monthly Adoption Rate Boxplot)",
#         xaxis_title="", # Removed x-axis title
#         yaxis_title="BNPL Adoption Rate (%)",
#         height=250,
#         margin=dict(l=20, r=20, t=40, b=20),
#         plot_bgcolor='white',
#         showlegend=False,
#         xaxis=dict(tickangle=45)
#     )

#     return fig.to_html(div_id="bnpl_adoption", include_plotlyjs=False)

# def generate_dashboard_html(df):
#     """Generate the complete 3x3 dashboard HTML"""

#     # Generate all chart components
#     sales_trend = create_sales_trendline(df)
#     sales_target = create_sales_vs_target(df)
#     transaction_count = create_transaction_count(df)
#     geo_heatmap = create_geo_heatmap(df)
#     sales_channel = create_sales_by_channel(df)
#     top_products = create_top_products_pie(df)
#     return_rate = create_return_rate_trend(df)
#     loyalty_heatmap = create_loyalty_heatmap(df)
#     bnpl_adoption = create_bnpl_adoption_boxplot(df)

#     # Create the HTML template
#     html_template = f"""
#     <!DOCTYPE html>
#     <html lang="en">
#     <head>
#         <meta charset="UTF-8">
#         <meta name="viewport" content="width=device-width, initial-scale=1.0">
#         <title>CRM Sales Dashboard</title>
#         <style>
#             body {{
#                 font-family: Arial, sans-serif;
#                 margin: 0;
#                 padding: 20px;
#                 background-color: #f5f5f5;
#             }}
#             .dashboard {{
#                 display: grid;
#                 grid-template-columns: 1fr 1fr 1fr;
#                 grid-template-rows: 1fr 1fr 1fr;
#                 gap: 15px;
#                 height: 100vh;
#                 max-height: 900px;
#             }}
#             .widget {{
#                 background: white;
#                 border-radius: 8px;
#                 box-shadow: 0 2px 10px rgba(0,0,0,0.1);
#                 padding: 15px;
#                 overflow: hidden;
#                 display: flex;
#                 flex-direction: column;
#             }}
#             .widget h3 {{
#                 margin: 0 0 10px 0;
#                 color: #333;
#                 font-size: 14px;
#                 font-weight: bold;
#             }}
#             .chart-container {{
#                 flex: 1;
#                 min-height: 0;
#             }}
#             .blue-bg {{
#                 background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#                 color: white;
#             }}
#             .blue-bg h3 {{
#                 color: white;
#             }}
#             @media (max-width: 1200px) {{
#                 .dashboard {{
#                     grid-template-columns: 1fr 1fr;
#                     height: auto;
#                 }}
#             }}
#             @media (max-width: 768px) {{
#                 .dashboard {{
#                     grid-template-columns: 1fr;
#                     height: auto;
#                 }}
#             }}
#         </style>
#     </head>
#     <body>
#         <div class="dashboard">
#             <div class="widget blue-bg">
#                 <h3>Total Sales (MTD/QTD/YTD) Value<br>Trendline as backdrop</h3>
#                 <div class="chart-container">
#                     {sales_trend}
#                 </div>
#             </div>

#             <div class="widget blue-bg">
#                 <h3>Sales vs Target %<br>Last 6 months two bar per one X</h3>
#                 <div class="chart-container">
#                     {sales_target}
#                 </div>
#             </div>

#             <div class="widget blue-bg">
#                 <h3># of Transactions<br>distinct count of invoices</h3>
#                 <div class="chart-container">
#                     {transaction_count}
#                 </div>
#             </div>

#             <div class="widget">
#                 <h3>Top Performing Branches<br>geo view</h3>
#                 <div class="chart-container">
#                     {geo_heatmap}
#                 </div>
#             </div>

#             <div class="widget">
#                 <h3>Sales by Channel<br>Bar chart</h3>
#                 <div class="chart-container">
#                     {sales_channel}
#                 </div>
#             </div>

#             <div class="widget">
#                 <h3>Top 5 Products<br>Pie chart</h3>
#                 <div class="chart-container">
#                     {top_products}
#                 </div>
#             </div>

#             <div class="widget">
#                 <h3>Return Rate<br>trendline</h3>
#                 <div class="chart-container">
#                     {return_rate}
#                 </div>
#             </div>

#             <div class="widget">
#                 <h3>Loyalty Engagement<br>per store per geo heatmap</h3>
#                 <div class="chart-container">
#                     {loyalty_heatmap}
#                 </div>
#             </div>

#             <div class="widget">
#                 <h3>BNPL Adoption Rate<br>monthwise adoption rate boxplot</h3>
#                 <div class="chart-container">
#                     {bnpl_adoption}
#                 </div>
#             </div>
#         </div>
#     </body>
#     </html>
#     """

#     return html_template

# def main():
#     """Main function to generate the dashboard"""
#     # File path - update this to your actual file location
#     file_path = r"D:\Mockup\data\synthetic_transaction_data.csv"

#     try:
#         # Load and process data
#         print("Loading data...")
#         df = load_and_process_data(file_path)
#         print(f"Data loaded successfully. Shape: {df.shape}")

#         # Generate dashboard HTML
#         print("Generating dashboard...")
#         dashboard_html = generate_dashboard_html(df)

#         # Save to file
#         output_file = "crm_sales_dashboard.html"
#         with open(output_file, 'w', encoding='utf-8') as f:
#             f.write(dashboard_html)

#         print(f"Dashboard generated successfully! Open '{output_file}' in your browser to view.")

#     except FileNotFoundError:
#         print(f"Error: Could not find the file at {file_path}")
#         print("Please make sure the file path is correct.")
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")

# if __name__ == "__main__":
#     main()

