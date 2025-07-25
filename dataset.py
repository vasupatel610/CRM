import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid

# Configuration
num_transactions = 5000
max_lines_per_transaction = 5
num_customers = 1000
num_products = 50
num_branches = 10
channels = ["Online", "In-Store", "B2B"]
payment_modes = ["Cash", "Card", "BNPL"]

# Define a yearly growth rate for quantities or overall sales value
yearly_sales_growth_rate = 1.10 # 10% growth year-over-year

# Electronic products list (50 products)
electronic_products = [
    "iPhone 15 Pro", "Samsung Galaxy S24", "MacBook Air M3", "Dell XPS 13", "HP Pavilion",
    "Sony WH-1000XM5", "AirPods Pro", "JBL Flip 6", "Canon EOS R6", "Nikon D7500",
    "iPad Pro 12.9", "Samsung Tab S9", "Microsoft Surface", "Asus ROG Laptop", "Lenovo ThinkPad",
    "Apple Watch Series 9", "Samsung Galaxy Watch", "Fitbit Charge 6", "Garmin Venu 3", "OnePlus Watch",
    "LG OLED TV 55", "Samsung QLED 65", "Sony Bravia 43", "TCL 4K Smart TV", "Hisense ULED",
    "PlayStation 5", "Xbox Series X", "Nintendo Switch", "Steam Deck", "Asus ROG Ally",
    "Dyson V15 Detect", "Roomba i7+", "Philips Air Fryer", "Instant Pot Duo", "KitchenAid Mixer",
    "Bose SoundLink", "Marshall Acton", "Sonos One", "Echo Dot 5th Gen", "Google Nest Hub",
    "Ring Video Doorbell", "Arlo Pro 4", "Nest Cam", "Wyze Cam v3", "Eufy Security",
    "Tesla Model Y Charger", "Anker PowerBank", "Belkin Wireless Charger", "Logitech MX Master", "Razer DeathAdder"
]

# Product category mapping
# Product category mapping
product_category_map = {
    "iPhone 15 Pro": "Mobile & Computing", "Samsung Galaxy S24": "Mobile & Computing",
    "MacBook Air M3": "Mobile & Computing", "Dell XPS 13": "Mobile & Computing",
    "HP Pavilion": "Mobile & Computing", "Microsoft Surface": "Mobile & Computing",
    "Asus ROG Laptop": "Mobile & Computing", "Lenovo ThinkPad": "Mobile & Computing",
    "iPad Pro 12.9": "Mobile & Computing", "Samsung Tab S9": "Mobile & Computing",
    "Logitech MX Master": "Mobile & Computing", "Razer DeathAdder": "Mobile & Computing",
    "Anker PowerBank": "Mobile & Computing", "Belkin Wireless Charger": "Mobile & Computing",
    
    "Apple Watch Series 9": "Wearables & Accessories", "Samsung Galaxy Watch": "Wearables & Accessories",
    "Fitbit Charge 6": "Wearables & Accessories", "Garmin Venu 3": "Wearables & Accessories",
    "OnePlus Watch": "Wearables & Accessories", "AirPods Pro": "Wearables & Accessories",
    "Sony WH-1000XM5": "Wearables & Accessories",
    
    "LG OLED TV 55": "Entertainment & Gaming", "Samsung QLED 65": "Entertainment & Gaming",
    "Sony Bravia 43": "Entertainment & Gaming", "TCL 4K Smart TV": "Entertainment & Gaming",
    "Hisense ULED": "Entertainment & Gaming", "PlayStation 5": "Entertainment & Gaming",
    "Xbox Series X": "Entertainment & Gaming", "Nintendo Switch": "Entertainment & Gaming",
    "Steam Deck": "Entertainment & Gaming", "Asus ROG Ally": "Entertainment & Gaming",
    "JBL Flip 6": "Entertainment & Gaming", "Canon EOS R6": "Entertainment & Gaming",
    "Nikon D7500": "Entertainment & Gaming",
    "Bose SoundLink": "Entertainment & Gaming", "Marshall Acton": "Entertainment & Gaming",
    "Sonos One": "Entertainment & Gaming",
    
    "Dyson V15 Detect": "Smart Home & Appliances",
    "Roomba i7+": "Smart Home & Appliances", # Corrected this line
    "Philips Air Fryer": "Smart Home & Appliances", "Instant Pot Duo": "Smart Home & Appliances",
    "KitchenAid Mixer": "Smart Home & Appliances", "Echo Dot 5th Gen": "Smart Home & Appliances",
    "Google Nest Hub": "Smart Home & Appliances", "Ring Video Doorbell": "Smart Home & Appliances",
    "Arlo Pro 4": "Smart Home & Appliances", "Nest Cam": "Smart Home & Appliances",
    "Wyze Cam v3": "Smart Home & Appliances", "Eufy Security": "Smart Home & Appliances",
    "Tesla Model Y Charger": "Smart Home & Appliances"
}


# Generate mapping tables
customer_ids = [f"CUST{str(i).zfill(4)}" for i in range(1, num_customers + 1)]
product_ids = [f"PROD{str(i).zfill(3)}" for i in range(1, num_products + 1)]
branch_ids = [f"BR{str(i).zfill(2)}" for i in range(1, num_branches + 1)]

product_name_map = {product_ids[i]: electronic_products[i] for i in range(num_products)}

online_branch_id = "BR01"
online_staff_id = "STF01"
staff_ids = {branch_id: f"STF{branch_id[-2:]}" for branch_id in branch_ids}

# Base price mapping
product_price_ranges = {name: (min_p, max_p) for name, (min_p, max_p) in zip(electronic_products, [
    (120000,180000),(90000,150000),(150000,250000),(80000,150000),(70000,160000),
    (40000,60000),(35000,50000),(12000,18000),(250000,350000),(120000,180000),
    (140000,220000),(100000,170000),(90000,180000),(150000,400000),(80000,190000),
    (60000,80000),(25000,45000),(20000,30000),(60000,75000),(15000,25000),
    (120000,250000),(180000,300000),(50000,90000),(40000,80000),(70000,200000),
    (70000,90000),(65000,85000),(40000,60000),(80000,120000),(130000,160000),
    (100000,150000),(70000,120000),(15000,30000),(20000,45000),(60000,90000),
    (25000,70000),(30000,50000),(25000,40000),(10000,15000),(30000,45000),
    (15000,30000),(40000,60000),(20000,35000),(8000,15000),(9000,20000),
    (50000,80000),(6000,20000),(4000,18000),(10000,18000),(5000,12000)
])}

product_base_price_map = {
    pid: round(random.uniform(product_price_ranges[product_name_map[pid]][0], product_price_ranges[product_name_map[pid]][1]), 2)
    for pid in product_ids
}
product_unit_price_map = {pid: product_base_price_map[pid] for pid in product_ids}

month_weights = {
    1: 0.8, 2: 0.7, 3: 1.0, 4: 1.1, 5: 0.9, 6: 1.3,
    7: 1.4, 8: 1.0, 9: 1.2, 10: 1.5, 11: 1.7, 12: 2.0
}
current_date = datetime.now().date()
date_pool = []
for year in [2023, 2024, 2025]:
    # Adjust number of entries in date_pool to simulate more transactions in later years
    base_num_days_per_month = 300
    if year == 2023:
        year_multiplier_for_transactions = 1.0
    elif year == 2024:
        year_multiplier_for_transactions = yearly_sales_growth_rate # 10% more transactions in 2024
    elif year == 2025:
        year_multiplier_for_transactions = yearly_sales_growth_rate ** 2 # 21% more transactions in 2025 (cumulative)

    for month, weight in month_weights.items():
        for _ in range(int(weight * base_num_days_per_month * year_multiplier_for_transactions)):
            day = random.randint(1, 28)
            date = datetime(year, month, day).date()
            if date <= current_date:
                date_pool.append(datetime(year, month, day, random.randint(8,20), random.randint(0,59), random.randint(0,59)))

channel_quantity_weights = {
    "Online": [0.6, 0.3, 0.08, 0.02],
    "B2B": [0.3, 0.4, 0.25, 0.05],
    "In-Store": [0.4, 0.35, 0.2, 0.05]
}

# Generate customer ages
customer_age_map = {cust_id: random.randint(18, 75) for cust_id in customer_ids}

data = []
for _ in range(num_transactions):
    transaction_id = str(uuid.uuid4())
    transaction_date = random.choice(date_pool)
    customer_id = random.choice(customer_ids)
    channel_id = random.choice(channels)

    branch_id = online_branch_id if channel_id == "Online" else random.choice(branch_ids)
    staff_id = online_staff_id if channel_id == "Online" else staff_ids[branch_id]

    allowed_payments = ["Cash", "Card"] if channel_id == "In-Store" else (["Card", "BNPL"] if channel_id == "Online" else payment_modes)
    payment_mode = random.choice(allowed_payments)
    voucher_redeemed_flag = np.random.choice(["Yes", "No"], p=[0.25, 0.75])
    is_upsell = np.random.choice(["Yes", "No"], p=[0.3, 0.7])
    is_returned = "No"

    num_lines = random.randint(1, max_lines_per_transaction)
    used_products = set()

    for line_num in range(1, num_lines + 1):
        product_id = random.choice([pid for pid in product_ids if pid not in used_products])
        used_products.add(product_id)

        unit_price = product_unit_price_map[product_id]
        
        # Apply a price increase based on the year to simulate positive revenue variation
        price_year_multiplier = 1.0
        if transaction_date.year == 2024:
            price_year_multiplier = 1.05 # 5% price increase in 2024
        elif transaction_date.year == 2025:
            price_year_multiplier = 1.10 # 10% price increase in 2025 (cumulative)
        unit_price = round(unit_price * price_year_multiplier, 2)

        quantity = np.random.choice([1, 2, 3, 5], p=channel_quantity_weights[channel_id])
        # Apply quantity increase for later years
        if transaction_date.year == 2024:
            quantity = int(quantity * 1.05) # ~5% quantity increase
        elif transaction_date.year == 2025:
            quantity = int(quantity * 1.10) # ~10% quantity increase (cumulative)
        quantity = max(1, quantity) # Ensure quantity is at least 1

        discount_rate_on_line = np.random.choice([0, 0.02, 0.05, 0.1, 0.15], p=[0.6, 0.15, 0.1, 0.1, 0.05])
        discount_applied = round(discount_rate_on_line * unit_price * quantity, 2)
        product_name = product_name_map[product_id]
        product_category = product_category_map[product_name]

        data.append({
            "transaction_id": transaction_id,
            "transaction_line_id": f"{transaction_id[:8]}-{line_num}",
            "customer_id": customer_id,
            "product_id": product_id,
            "product_name": product_name,
            "product_category": product_category_map[product_name_map[product_id]],
            "branch_id": branch_id,
            "staff_id": staff_id,
            "channel_id": channel_id,
            "transaction_date": transaction_date.strftime("%Y-%m-%d %H:%M:%S"),
            "quantity": quantity,
            "unit_price": unit_price,
            "discount_applied": discount_applied,
            "payment_mode": payment_mode,
            "voucher_redeemed_flag": voucher_redeemed_flag,
            "is_upsell": is_upsell,
            "is_returned": is_returned
        })

df = pd.DataFrame(data)
df["grand_total"] = (df["unit_price"] * df["quantity"]) - df["discount_applied"]
df["net_price"] = df["grand_total"] / df["quantity"]
df["transaction_date"] = pd.to_datetime(df["transaction_date"])
df["month"] = df["transaction_date"].dt.month

# Add customer age to the DataFrame
df["customer_age"] = df["customer_id"].map(customer_age_map)

# Add returned flags
for month in df["month"].unique():
    month_idx = df[df["month"] == month].index
    rate = random.uniform(0.08, 0.18)
    n_return = int(rate * len(month_idx))
    if n_return > 0:
        returned_idx = np.random.choice(month_idx, n_return, replace=False)
        df.loc[returned_idx, "is_returned"] = "Yes"

# Monthly targets with year-over-year growth
monthly_target_base_multipliers = {
    1: 1.10, 2: 1.15, 3: 1.05, 4: 1.08, 5: 1.12, 6: 0.95,
    7: 0.98, 8: 1.02, 9: 1.00, 10: 1.04, 11: 1.06, 12: 1.07
}
def get_target_multiplier_with_growth(month, year):
    base_multiplier = monthly_target_base_multipliers.get(month, 1.0)
    year_factor = 1.0
    if year == 2024:
        year_factor = 1.03 # 3% higher targets in 2024
    elif year == 2025:
        year_factor = 1.06 # 6% higher targets in 2025 (cumulative)
    return base_multiplier * year_factor

df["target_multiplier"] = df.apply(lambda row: get_target_multiplier_with_growth(row['month'], row['transaction_date'].year), axis=1)

# Add geolocation
random.seed(42)
branch_ids_unique = df["branch_id"].unique()
kenya_locations = [
    {"lat": -1.2921, "lon": 36.8219}, {"lat": -1.3731, "lon": 36.8569}, {"lat": -4.0435, "lon": 39.6682},
    {"lat": -0.3031, "lon": 35.2961}, {"lat": -0.0917, "lon": 34.7680}, {"lat": -0.4906, "lon": 35.2719},
    {"lat": 0.5143, "lon": 35.2698}, {"lat": -1.5177, "lon": 37.2634}, {"lat": -0.6743, "lon": 34.5615},
    {"lat": -3.2175, "lon": 40.1167}
]

branch_geo = {}
for i, branch_id in enumerate(branch_ids_unique):
    if i < len(kenya_locations):
        lat_variation = random.uniform(-0.01, 0.01)
        lon_variation = random.uniform(-0.01, 0.01)
        branch_geo[branch_id] = {
            "latitude": round(kenya_locations[i]["lat"] + lat_variation, 6),
            "longitude": round(kenya_locations[i]["lon"] + lon_variation, 6)
        }
    else:
        branch_geo[branch_id] = {
            "latitude": round(random.uniform(-4.5, 1.0), 6),
            "longitude": round(random.uniform(33.9, 41.9), 6)
        }

df["latitude"] = df["branch_id"].map(lambda x: branch_geo[x]["latitude"])
df["longitude"] = df["branch_id"].map(lambda x: branch_geo[x]["longitude"])

# --- Add Churn Flag to synthetic_transaction_data.csv ---
# Calculate recency (days since last purchase) and total returns per customer
customer_last_purchase = df.groupby('customer_id')['transaction_date'].max()
customer_total_returns = df[df['is_returned'] == 'Yes'].groupby('customer_id').size().fillna(0)
customer_data = pd.DataFrame(customer_ids, columns=['customer_id'])

customer_data['last_purchase_date'] = customer_data['customer_id'].map(customer_last_purchase)
customer_data['total_returns'] = customer_data['customer_id'].map(customer_total_returns).fillna(0)
customer_data['customer_age'] = customer_data['customer_id'].map(customer_age_map)

# Calculate recency in days relative to the latest transaction date in the dataset
latest_transaction_date = df['transaction_date'].max()
customer_data['recency'] = (latest_transaction_date - customer_data['last_purchase_date']).dt.days

# Define churn probability based on recency, returns, and age
def get_churn_probability(row):
    prob = 0.05 # Base churn probability
    if row['recency'] > 250:
        prob += 0.35
    elif row['recency'] > 120:
        prob += 0.20
    elif row['recency'] > 60:
        prob += 0.10
    
    if row['total_returns'] >= 5:
        prob += 0.30
    elif row['total_returns'] >= 2:
        prob += 0.15
        
    if row['customer_age'] >= 65:
        prob += 0.10
    elif row['customer_age'] <= 22:
        prob += 0.07
    
    return min(prob, 0.9)

customer_data['churn_probability'] = customer_data.apply(get_churn_probability, axis=1)
customer_data['is_churned'] = customer_data['churn_probability'].apply(lambda p: 1 if random.random() < p else 0)

# Map churn status back to the main DataFrame
df = df.merge(customer_data[['customer_id', 'is_churned']], on='customer_id', how='left')
df['is_churned'] = df.groupby('customer_id')['is_churned'].transform('first')


# Save main transaction data
output_file = "synthetic_transaction_data.csv"
df.to_csv(output_file, index=False)
print(f"Dataset saved to: {output_file}")
# ============================================================================
# GENERATE SENTIMENT.CSV
# ============================================================================
print("Generating sentiment.csv with correlated long and dynamic reviews...")

# First, get actual customer-product combinations from transaction data
customer_product_purchases = df[['customer_id', 'product_id', 'transaction_date', 'is_returned']].drop_duplicates()
print(f"Found {len(customer_product_purchases)} unique customer-product purchase combinations")

social_media_platforms = ["Twitter", "Facebook", "Instagram", "TikTok", "Google", "YouTube"]
hashtags_choices = ["#electronics", "#newtech", "#gadgetreview", "#productlove", "#techdeals", "#unboxing", "#smartlife", "#innovate"]
campaign_names = ["HolidayTechSale", "SpringSavings", "GamingGearLaunch", "WorkFromHomeEssentials", "BackToSchoolTech"]

# Enhanced long review samples with detailed feedback
positive_long_reviews = [
    "I absolutely love this product! The quality is unmatched and it has exceeded all my expectations. The build quality feels premium and every feature works flawlessly. Customer service was also very helpful when I had questions during setup. The packaging was excellent and delivery was faster than expected. This has definitely become my go-to brand for electronics.",
    "This device is a complete game changer for me. It works flawlessly and the battery life is absolutely fantastic, lasting way longer than advertised. The interface is intuitive and user-friendly. I've been using it daily for months now and it still performs like new. Highly recommended to anyone looking for reliability and performance.",
    "The features and performance are incredible beyond what I imagined. The design is sleek and modern, fitting perfectly with my setup. I've never been happier with a purchase - it has made my work so much more efficient. The product arrived on time, was well packaged, and came with clear instructions. Worth every single penny!",
    "Top-notch quality and excellent support from both the seller and manufacturer. The product feels solid and well-built, with attention to detail that shows. I've already recommended this to my friends, family, and colleagues. The after-sales service has been outstanding whenever I needed assistance.",
    "Fantastic product that has completely transformed my daily routine and made tasks so much easier. The performance is consistent and reliable. I appreciate the thoughtful design and the range of features available. Will definitely be buying from this brand again and have already placed another order!",
    "Outstanding value for money with premium features that actually work as advertised. The product arrived in perfect condition with all accessories included. Setup was straightforward and the user manual was comprehensive. This has become an essential part of my daily workflow.",
    "Exceptional quality and craftsmanship evident in every detail. The product exceeded my expectations in terms of both performance and durability. Customer support was responsive and knowledgeable when I had questions. I'm thoroughly impressed and satisfied with this purchase."
]

negative_long_reviews = [
    "I was really disappointed with this product after having high expectations. It did not perform as advertised and the build quality feels cheap and flimsy. Multiple features stopped working within the first week, and I had to return it. The customer service was unhelpful and took forever to respond to my complaints. Definitely not worth the premium price they're charging.",
    "The item arrived damaged with visible scratches and dents, clearly not properly packaged for shipping. When I contacted customer service, they were slow to respond and seemed unwilling to help resolve the issue. The product itself feels cheaply made and doesn't justify the high price point. Would not recommend to anyone.",
    "Had high hopes for this product based on the reviews, but it kept malfunctioning and caused constant frustration. The interface is confusing and counter-intuitive. Several key features simply don't work as described. After dealing with multiple issues for weeks, I finally gave up and returned it. Very disappointing experience overall.",
    "The delivery was delayed by several weeks without proper communication, and when the product finally arrived, it had several obvious defects and missing components. The build quality is poor and it feels like it might break at any moment. Very unhappy with this purchase and the overall experience with this seller.",
    "Poor quality materials and even worse customer support. The product started having issues within days of purchase, and getting help was like pulling teeth. Multiple attempts to contact support were ignored or met with unhelpful responses. I regret buying this and will be looking for alternatives from more reputable brands.",
    "Completely overpriced for what you actually get. The product feels like a cheap knockoff despite the premium branding. Performance is inconsistent and unreliable. The warranty process was a nightmare when issues started appearing. Save your money and look elsewhere.",
    "Terrible experience from start to finish. Product arrived late, was missing accessories, and stopped working properly within a few days. Return process was complicated and customer service was rude and unhelpful. Worst purchase I've made in years."
]

neutral_long_reviews = [
    "The product is okay for the price point, though nothing particularly special about it. It meets the basic functionality requirements but lacks some of the more advanced features I was hoping for. Build quality is adequate but not premium. Delivery was on time and packaging was sufficient to protect the item.",
    "Average experience overall with this purchase. The device does what it says it will do, but there are some minor issues and limitations that could be improved. Customer service was responsive but not particularly knowledgeable. It's a decent option if you're looking for something basic.",
    "It's decent and serves its purpose, but lacks some features I was expecting based on the product description. The interface could be more user-friendly and intuitive. Delivery was on time and packaging was adequate. Fair value but there might be better alternatives available.",
    "Good enough for everyday use, though I wouldn't recommend it if you need something more robust or feature-rich. The product works as intended but feels somewhat limited. Support documentation could be more comprehensive. Reasonable purchase for basic needs.",
    "Fair product with average performance that meets minimum expectations. Some aspects are well-designed while others feel like afterthoughts. Customer support was okay but not outstanding. It's an acceptable choice if you're not looking for premium features.",
    "The product performs adequately for basic tasks but doesn't excel in any particular area. Build quality is standard - not bad but not impressive either. Setup was straightforward enough. It's an okay purchase if you need something functional without premium expectations.",
    "Reasonable product that does the job without any major issues, though it's not particularly exciting or innovative. The price point seems fair for what you get. Some features work better than others. Overall, it's a safe but unremarkable choice."
]


# Dynamic phrase components for creating varied reviews
intro_phrases = [
    "Honestly, ", "Frankly, ", "Personally, ", "To be honest, ", "In my opinion, ",
    "After using this for weeks, ", "From my experience, ", "I have to say, ",
    "Looking back on this purchase, ", "Having tried similar products, "
]

outro_phrases = [
    "Would definitely recommend to others.", "Not sure if I would buy again.",
    "Might consider alternatives next time.", "Customer service was surprisingly helpful.",
    "Had some minor issues but overall manageable.", "Overall quite satisfied with the purchase.",
    "Will be keeping an eye on future products from this brand.", "Mixed feelings about this one.",
    "Exceeded my expectations in some areas.", "Could use some improvements but solid overall."
]

def generate_dynamic_review(sentiment, product_name):
    """Generate dynamic reviews with varied structure and product-specific details"""
    
    if sentiment == 'positive':
        base_reviews = positive_long_reviews
    elif sentiment == 'negative':
        base_reviews = negative_long_reviews
    else:
        base_reviews = neutral_long_reviews
    
    # Select base review
    base_review = random.choice(base_reviews)
    
    # Randomly modify the review structure
    review = ""
    
    # 30% chance to add intro phrase
    if random.random() < 0.3:
        review += random.choice(intro_phrases)
    
    review += base_review
    
    # 40% chance to add outro phrase
    if random.random() < 0.4:
        review += " " + random.choice(outro_phrases)
    
    # 20% chance to add product-specific mention
    if random.random() < 0.2:
        product_mentions = [
            f" The {product_name} specifically impressed me.",
            f" This particular {product_name} model stands out.",
            f" Compared to other similar products, this {product_name} is different.",
            f" The {product_name} has some unique qualities."
        ]
        review += random.choice(product_mentions)
    
    return review

sentiment_data = []

# Generate reviews for a subset of actual purchases
review_probability = 0.60
purchases_to_review = customer_product_purchases.sample(n=int(len(customer_product_purchases) * review_probability), random_state=42)

for _, purchase in purchases_to_review.iterrows():
    cust = purchase['customer_id']
    prod = purchase['product_id']
    purchase_date = purchase['transaction_date']
    is_returned = purchase['is_returned']
    
    # Get product name for dynamic mentions
    product_name = product_name_map.get(prod, "product")
    
    # Review date should be after purchase date (1-45 days later for more realistic timing)
    review_date = purchase_date + timedelta(days=random.randint(1, 45))
    
    # Sentiment bias based on return status and some randomness
    if is_returned == "Yes":
        sentiment_weights = [0.05, 0.85, 0.1] # High probability of negative if returned
    else:
        # Introduce positive variation in sentiment over time
        # Newer reviews have a slightly higher chance of being positive
        year = review_date.year
        if year == 2023:
            sentiment_weights = [0.75, 0.05, 0.20] # Base positive
        elif year == 2024:
            sentiment_weights = [0.80, 0.04, 0.16] # More positive
        elif year == 2025:
            sentiment_weights = [0.85, 0.03, 0.12] # Even more positive
        else:
            sentiment_weights = [0.75, 0.05, 0.20] # Default for other years
    
    # Choose review type based on sentiment weights
    review_type = random.choices(['positive', 'negative', 'neutral'], weights=sentiment_weights)[0]
    
    # Generate dynamic review
    review = generate_dynamic_review(review_type, product_name)
    
    platform = random.choice(social_media_platforms)
    hashtags = ' '.join(random.sample(hashtags_choices, random.randint(2,4)))
    campaign = random.choice(campaign_names)
    
    sentiment_data.append({
        "customer_id": cust,
        "product_id": prod,
        "product_name": product_name, # Added product_name here
        "reviews": review,
        "date": review_date.strftime("%Y-%m-%d"),
        "social_media_platform": platform,
        "hashtags": hashtags,
        "campaign_name": campaign,
        "review_sentiment": review_type
    })

sentiment_df = pd.DataFrame(sentiment_data)
sentiment_df['customer_age'] = sentiment_df['customer_id'].map(customer_age_map)
sentiment_df.to_csv("sentiment.csv", index=False)
print("sentiment.csv created successfully!")


# ============================================================================
# GENERATE JOURNEY_ENTRY.CSV
# ============================================================================
print("Generating journey_entry.csv with campaign and hashtag correlation...")

# First, create a mapping of customer-product combinations to their sentiment data
sentiment_mapping = {}
if 'sentiment_df' in locals():
    for _, row in sentiment_df.iterrows():
        key = (row['customer_id'], row['product_id'])
        sentiment_mapping[key] = {
            'campaign_name': row['campaign_name'],
            'hashtags': row['hashtags']
        }

funnel_stages = ["Lead", "Awareness", "Consideration", "Purchase", "Service", "Loyalty", "Advocacy"]
num_journeys = 2500
journey_data = []

# Define drop_stage_weights here, before the loop that uses it
drop_stage_weights = [0.25, 0.20, 0.18, 0.15, 0.10, 0.07, 0.05] # Total 1.0 (5% reach advocacy)

for _ in range(num_journeys):
    cust = random.choice(customer_ids)
    prod = random.choice(product_ids)
    journey_id = str(uuid.uuid4())
    base_date = datetime(2023,1,1) + timedelta(days=random.randint(0,600))
    
    # Get campaign and hashtag data if available from sentiment
    sentiment_key = (cust, prod)
    if sentiment_key in sentiment_mapping:
        campaign_name = sentiment_mapping[sentiment_key]['campaign_name']
        hashtags = sentiment_mapping[sentiment_key]['hashtags']
    else:
        campaign_name = random.choice(campaign_names)
        hashtags = ' '.join(random.sample(hashtags_choices, random.randint(2,3)))
    
    # Get product name using the product_id
    product_name = product_name_map.get(prod, "Unknown Product")

    final_stage_reached = None
    cumulative_probs = np.cumsum(drop_stage_weights)
    r = random.random()

    for i, prob in enumerate(cumulative_probs):
        if r < prob:
            final_stage_reached = funnel_stages[i]
            break
    if final_stage_reached is None:
        final_stage_reached = funnel_stages[-1]

    campaign_open = "No"
    campaign_click = "No"
    conversion_flag = "No"
    
    # Add product_in_cart with 65% 'Yes' probability
    product_in_cart = np.random.choice(["Yes", "No"], p=[0.65, 0.35])

    for i, stage in enumerate(funnel_stages):
        if final_stage_reached is not None and funnel_stages.index(stage) > funnel_stages.index(final_stage_reached):
            break

        entry_date_dt = (base_date + timedelta(days=random.randint(5,15)*i, hours=random.randint(0,23), minutes=random.randint(0,59)))
        entry_date = entry_date_dt.strftime("%Y-%m-%d %H:%M:%S")

        # Introduce positive variation in campaign metrics over time
        year = entry_date_dt.year
        open_prob_increase = 0.0
        click_prob_increase = 0.0
        conversion_prob_increase = 0.0
        cart_prob_increase = 0.0 # for product_in_cart
        
        if year == 2024:
            open_prob_increase = 0.03 # 3% higher chance of opening in 2024
            click_prob_increase = 0.05 # 5% higher chance of clicking in 2024
            conversion_prob_increase = 0.02 # 2% higher chance of conversion in 2024
            cart_prob_increase = 0.05 # 5% higher chance of adding to cart in 2024
        elif year == 2025:
            open_prob_increase = 0.06 # 6% higher chance of opening in 2025 (cumulative)
            click_prob_increase = 0.10 # 10% higher chance of clicking in 2025 (cumulative)
            conversion_prob_increase = 0.05 # 5% higher chance of conversion in 2025 (cumulative)
            cart_prob_increase = 0.10 # 10% higher chance of adding to cart in 2025 (cumulative)


        if stage == "Lead":
            # Apply increased probability for campaign_open
            if random.random() < (0.80 + open_prob_increase):
                campaign_open = "Yes"
            # Apply increased probability for campaign_click
            if campaign_open == "Yes" and random.random() < (0.35 + click_prob_increase):
                campaign_click = "Yes"
        
        if stage == "Purchase":
            # Apply increased probability for conversion_flag
            if random.random() < (0.6 + conversion_prob_increase):
                conversion_flag = "Yes"
            else:
                conversion_flag = "No"
        elif stage in ["Service", "Loyalty", "Advocacy"]:
            conversion_flag = "Yes"

        # Apply increased probability for product_in_cart
        product_in_cart_base_prob = 0.65
        product_in_cart_current_prob = min(1.0, product_in_cart_base_prob + cart_prob_increase)
        product_in_cart = np.random.choice(["Yes", "No"], p=[product_in_cart_current_prob, 1 - product_in_cart_current_prob])


        journey_data.append({
            "journey_id": journey_id,
            "customer_id": cust,
            "product_id": prod,
            "product_name": product_name,
            "stage": stage,
            "stage_date": entry_date,
            "campaign_name": campaign_name,
            "hashtags": hashtags,
            "campaign_open": campaign_open,
            "campaign_click": campaign_click,
            "conversion_flag": conversion_flag,
            "product_in_cart": product_in_cart
        })

journey_df = pd.DataFrame(journey_data)
journey_df['customer_age'] = journey_df['customer_id'].map(customer_age_map)
journey_df.to_csv("journey_entry.csv", index=False)
print("journey_entry.csv created successfully!")


# ============================================================================
# GENERATE CUSTOMER_SEGMENTATION.CSV
# ============================================================================
print("Generating customer_segmentation.csv with positive variation...")

# --- Load Existing Data and Convert Date Columns ---
# This section is the primary fix for the TypeError
try:
    df = pd.read_csv("synthetic_transaction_data.csv")
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
except FileNotFoundError:
    print("synthetic_transaction_data.csv not found. Please run the main data generator script first.")
    exit()

try:
    journey_df = pd.read_csv("journey_entry.csv")
    journey_df['stage_date'] = pd.to_datetime(journey_df['stage_date']) # FIX: Convert to datetime
except FileNotFoundError:
    print("journey_entry.csv not found. Please run the main data generator script first.")
    exit()

try:
    sentiment_df = pd.read_csv("sentiment.csv")
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date']) # FIX: Convert to datetime
except FileNotFoundError:
    print("sentiment.csv not found. Please run the main data generator script first.")
    exit()

# Ensure customer_id is consistent across dataframes
customer_ids = df['customer_id'].unique()


segment_names = ["New", "Churn Risk", "Loyal", "High-Value", "Promising"]
segment_descriptions = {
    "New": "Recently acquired customers, often with 1-2 purchases.",
    "Churn Risk": "Customers showing signs of disengagement, high recency or returns.",
    "Loyal": "Consistent purchasers, frequent interactions.",
    "High-Value": "Customers with high total spending and frequent purchases.",
    "Promising": "Newer customers with high initial engagement or spending potential."
}

customer_segment_data = []

# Simulate customer demographics for segmentation features
customer_demographics = {
    cust_id: {
        'age': random.randint(18, 75),
        'gender': random.choice(['Male', 'Female', 'Other']),
        'location_region': random.choice(['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret'])
    } for cust_id in customer_ids
}

# Base probabilities for segmentation, will be adjusted for positive variation
base_segment_probabilities = {
    "New": 0.25,
    "Churn Risk": 0.20,
    "Loyal": 0.30,
    "High-Value": 0.15,
    "Promising": 0.10
}

# Track customer assignments to ensure distribution over time
customer_assigned_year = {}

for cust_id in customer_ids:
    # Get relevant data for the customer
    cust_transactions = df[df['customer_id'] == cust_id] # Use the 'df' from main transactions
    cust_journey = journey_df[journey_df['customer_id'] == cust_id] # Use the 'journey_df' from journey generation
    cust_sentiment = sentiment_df[sentiment_df['customer_id'] == cust_id] # Use the 'sentiment_df' from sentiment generation

    total_spent = cust_transactions['grand_total'].sum()
    total_transactions = cust_transactions['transaction_id'].nunique()
    avg_quantity = cust_transactions['quantity'].mean()
    num_returns = cust_transactions[cust_transactions['is_returned'] == 'Yes'].shape[0]
    
    # Calculate recency relative to the *latest date in the transaction data*
    latest_transaction_date_overall = df['transaction_date'].max()
    last_purchase_date = cust_transactions['transaction_date'].max()
    recency = (latest_transaction_date_overall - last_purchase_date).days if pd.notna(last_purchase_date) else 365 # Default for no purchases
    
    # Sentiment score (e.g., simple aggregation)
    positive_reviews = cust_sentiment[cust_sentiment['review_sentiment'] == 'positive'].shape[0]
    negative_reviews = cust_sentiment[cust_sentiment['review_sentiment'] == 'negative'].shape[0]
    sentiment_score = (positive_reviews - negative_reviews) / (positive_reviews + negative_reviews + 1e-6) if (positive_reviews + negative_reviews) > 0 else 0 # Avoid division by zero
    
    is_churned_flag = cust_transactions['is_churned'].iloc[0] if not cust_transactions.empty else 0

    # Determine assigned year based on earliest activity
    earliest_activity_date = pd.NaT
    if not cust_transactions.empty:
        earliest_activity_date = cust_transactions['transaction_date'].min()
    if not cust_journey.empty:
        current_min_journey_date = cust_journey['stage_date'].min()
        if pd.isna(earliest_activity_date) or (pd.notna(current_min_journey_date) and current_min_journey_date < earliest_activity_date):
            earliest_activity_date = current_min_journey_date
    if not cust_sentiment.empty:
        current_min_sentiment_date = cust_sentiment['date'].min()
        if pd.isna(earliest_activity_date) or (pd.notna(current_min_sentiment_date) and current_min_sentiment_date < earliest_activity_date):
            earliest_activity_date = current_min_sentiment_date

    assigned_year = earliest_activity_date.year if pd.notna(earliest_activity_date) else random.choice([2023, 2024, 2025])
    customer_assigned_year[cust_id] = assigned_year

    # Adjust segment probabilities for positive variation over years
    current_segment_probabilities = base_segment_probabilities.copy()
    if assigned_year == 2024:
        current_segment_probabilities["High-Value"] = min(1.0, current_segment_probabilities["High-Value"] + 0.05)
        current_segment_probabilities["Loyal"] = min(1.0, current_segment_probabilities["Loyal"] + 0.05)
        current_segment_probabilities["Churn Risk"] = max(0.0, current_segment_probabilities["Churn Risk"] - 0.05) # Reduced churn risk
        current_segment_probabilities["New"] = max(0.0, current_segment_probabilities["New"] - 0.05) # Fewer truly "new" by proportion
    elif assigned_year == 2025:
        current_segment_probabilities["High-Value"] = min(1.0, current_segment_probabilities["High-Value"] + 0.10)
        current_segment_probabilities["Loyal"] = min(1.0, current_segment_probabilities["Loyal"] + 0.10)
        current_segment_probabilities["Churn Risk"] = max(0.0, current_segment_probabilities["Churn Risk"] - 0.10)
        current_segment_probabilities["New"] = max(0.0, current_segment_probabilities["New"] - 0.10)
    
    # Normalize probabilities to ensure they sum to 1
    sum_probs = sum(current_segment_probabilities.values())
    current_segment_probabilities = {k: v / sum_probs for k, v in current_segment_probabilities.items()}

    # Assign segment based on characteristics and adjusted probabilities
    segment = "New" # Default
    
    # More nuanced assignment logic
    if total_spent > 500000 and total_transactions > 5 and recency < 60:
        segment_options = ["High-Value", "Loyal"]
        weights = [0.7 + (0.1 * (assigned_year - 2023)), 0.3 - (0.1 * (assigned_year - 2023))]
        # Ensure weights are valid probabilities
        weights = [max(0.0, w) for w in weights]
        sum_weights = sum(weights)
        if sum_weights > 0:
            weights = [w / sum_weights for w in weights]
        else:
            weights = [0.5, 0.5] # Fallback
        segment = random.choices(segment_options, weights=weights)[0]
    elif is_churned_flag == 1 or recency > 180 or num_returns > 2:
        segment_options = ["Churn Risk", "New"]
        weights = [0.8 - (0.1 * (assigned_year - 2023)), 0.2 + (0.1 * (assigned_year - 2023))]
        weights = [max(0.0, w) for w in weights]
        sum_weights = sum(weights)
        if sum_weights > 0:
            weights = [w / sum_weights for w in weights]
        else:
            weights = [0.5, 0.5]
        segment = random.choices(segment_options, weights=weights)[0]
    elif total_transactions > 3 and recency < 90:
        segment_options = ["Loyal", "Promising"]
        weights = [0.6 + (0.1 * (assigned_year - 2023)), 0.4 - (0.1 * (assigned_year - 2023))]
        weights = [max(0.0, w) for w in weights]
        sum_weights = sum(weights)
        if sum_weights > 0:
            weights = [w / sum_weights for w in weights]
        else:
            weights = [0.5, 0.5]
        segment = random.choices(segment_options, weights=weights)[0]
    elif total_spent > 100000 and recency < 90:
        segment = "Promising"
    else:
        # Fallback to general probabilities for less clear-cut cases
        segment = random.choices(list(current_segment_probabilities.keys()), weights=list(current_segment_probabilities.values()))[0]


    customer_segment_data.append({
        "customer_id": cust_id,
        "segment_name": segment,
        "segment_description": segment_descriptions[segment],
        "total_spent": round(total_spent, 2),
        "total_transactions": total_transactions,
        "avg_quantity_per_transaction": round(avg_quantity if pd.notna(avg_quantity) else 0, 2),
        "recency_days": recency,
        "num_returns": num_returns,
        "avg_sentiment_score": round(sentiment_score, 2),
        "is_churned": is_churned_flag,
        "customer_age": customer_demographics[cust_id]['age'],
        "customer_gender": customer_demographics[cust_id]['gender'],
        "customer_location_region": customer_demographics[cust_id]['location_region'],
        "assigned_year": assigned_year # For analysis of trend
    })

customer_segment_df = pd.DataFrame(customer_segment_data)
customer_segment_df.to_csv("customer_segmentation.csv", index=False)
print("customer_segmentation.csv created successfully!")


# ============================================================================
# GENERATE EMERGING_JOURNEY_PATTERNS.CSV
# ============================================================================
print("Generating emerging_journey_patterns.csv with positive variation...")

# Group journey data to identify common patterns
# A pattern can be a sequence of stages for a customer-product
journey_patterns = []

# To ensure positive variation, we will define a "success rate" that increases over time
# for patterns that end in 'Purchase', 'Loyalty', or 'Advocacy'.
base_success_rate = 0.60 # Base probability of a journey ending in a 'success' stage

# Calculate success rate multiplier per year
def get_success_rate_multiplier(year):
    if year == 2023:
        return 1.0
    elif year == 2024:
        return 1.05 # 5% higher success rate in 2024
    elif year == 2025:
        return 1.10 # 10% higher success rate in 2025 (cumulative)
    return 1.0

# Calculate engagement metric multiplier per year
def get_engagement_multiplier(year):
    if year == 2023:
        return 1.0
    elif year == 2024:
        return 1.03 # 3% higher engagement in 2024
    elif year == 2025:
        return 1.06 # 6% higher engagement in 2025 (cumulative)
    return 1.0

for (cust_id, prod_id, journey_id), group in journey_df.groupby(['customer_id', 'product_id', 'journey_id']):
    sorted_stages = group.sort_values('stage_date')
    pattern = " -> ".join(sorted_stages['stage'].tolist())
    
    first_stage_date = sorted_stages['stage_date'].iloc[0]
    last_stage_date = sorted_stages['stage_date'].iloc[-1]
    duration_days = (last_stage_date - first_stage_date).days
    
    final_stage = sorted_stages['stage'].iloc[-1]
    
    # Calculate average campaign metrics for the journey
    avg_campaign_open = group['campaign_open'].apply(lambda x: 1 if x == 'Yes' else 0).mean()
    avg_campaign_click = group['campaign_click'].apply(lambda x: 1 if x == 'Yes' else 0).mean()
    conversion_flag = group['conversion_flag'].iloc[-1] # From the last stage in the journey
    avg_product_in_cart = group['product_in_cart'].apply(lambda x: 1 if x == 'Yes' else 0).mean()

    # Determine the year of the pattern (based on the first stage date)
    pattern_year = first_stage_date.year

    # Introduce positive variation based on the year
    current_success_rate = min(1.0, base_success_rate * get_success_rate_multiplier(pattern_year))
    current_engagement_multiplier = get_engagement_multiplier(pattern_year)
    
    # Adjust metrics based on the trend
    # If a pattern leads to a successful stage, its actual metrics might be higher due to the multiplier
    # This is a simplification; a more complex model might adjust these during initial generation.
    # For now, we'll just ensure the output reflects the trend we want to see.

    journey_patterns.append({
        "customer_id": cust_id,
        "product_id": prod_id,
        "journey_pattern": pattern,
        "pattern_length_stages": len(sorted_stages),
        "duration_days": duration_days,
        "final_stage_reached": final_stage,
        "avg_campaign_opens": round(avg_campaign_open * current_engagement_multiplier, 2), # Apply trend
        "avg_campaign_clicks": round(avg_campaign_click * current_engagement_multiplier, 2), # Apply trend
        "conversion_flag_journey": conversion_flag,
        "avg_product_in_cart": round(avg_product_in_cart * current_engagement_multiplier, 2), # Apply trend
        "pattern_year": pattern_year, # For analysis
        "success_rate_for_year": round(current_success_rate, 2) # For analysis
    })

emerging_journey_df = pd.DataFrame(journey_patterns)
emerging_journey_df.to_csv("emerging_journey_patterns.csv", index=False)
print("emerging_journey_patterns.csv created successfully!")


print("\n=== ALL FILES GENERATED ===")
print("1. synthetic_transaction_data.csv - Main transaction dataset (includes 'is_churned' and 'customer_age' columns)")
print("2. sentiment.csv - Customer sentiment and social media data (includes 'product_name' and 'customer_age' column)")
print("3. journey_entry.csv - Customer journey funnel data (includes campaign metrics, conversion flag, 'product_name', 'product_in_cart', and 'customer_age' column)")
print("4. customer_segmentation.csv - Customer segments with enhanced probabilities for high-value segments over time.")
print("5. emerging_journey_patterns.csv - Customer journey patterns with improved success and engagement metrics over time.")