import pandas as pd
import plotly.express as px

df = pd.read_csv('D:\proj\synthetic_transaction_data.csv')

def create_bnpl_adoption_boxplot(df, csv_output_path=None):
    # Step 0: Convert transaction_date to datetime and extract month-year
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df['month'] = df['transaction_date'].dt.strftime('%b-%Y')  # e.g., Mar-2025

    # Step 1: Create BNPL flag
    df['bnpl_flag'] = df['payment_mode'].apply(lambda x: 1 if str(x).strip().upper() == 'BNPL' else 0)

    # Step 2: Group by month and branch to calculate adoption
    branch_month_stats = (
        df.groupby(['month', 'branch_id'])
        .agg(
            total_txns=('transaction_id', 'count'),
            bnpl_txns=('bnpl_flag', 'sum')
        )
        .reset_index()
    )

    # Step 3: Calculate adoption rate
    branch_month_stats['bnpl_adoption_rate'] = branch_month_stats['bnpl_txns'] / branch_month_stats['total_txns']

    # Rename columns for output
    branch_month_stats.rename(columns={'branch_id': 'store_id'}, inplace=True)

    # Optional: Save to CSV if path provided
    if csv_output_path:
        branch_month_stats.to_csv(csv_output_path, index=False)

    # Step 4: Sort month in chronological order
    branch_month_stats['month'] = pd.to_datetime(branch_month_stats['month'], format='%b-%Y')
    branch_month_stats.sort_values('month', inplace=True)
    branch_month_stats['month'] = branch_month_stats['month'].dt.strftime('%b-%Y')

    # Step 5: Plot
    fig = px.box(
        branch_month_stats,
        x="month",
        y="bnpl_adoption_rate",
        points="all",
        color="month",
        title="BNPL Adoption Rate Distribution by Month (Per Store)",
        labels={
            "month": "Month",
            "bnpl_adoption_rate": "BNPL Adoption Rate"
        },
        template="plotly_white"
    )

    fig.update_traces(jitter=0.3, marker=dict(opacity=0.5))
    fig.update_layout(
        showlegend=False,
        yaxis_tickformat=".0%",
        xaxis=dict(tickangle=45),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig, branch_month_stats

# Usage
fig, df_bnpl_summary = create_bnpl_adoption_boxplot(df, csv_output_path="bnpl_adoption_summary.csv")
fig.show()
