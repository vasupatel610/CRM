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
    "Roomba i7+": "Smart Home & Appliances",
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
# GENERATE JOURNEY_ENTRY.CSV WITH CHANNEL_ID, SOCIAL_MEDIA_PLATFORM
# ============================================================================
print("Generating journey_entry.csv with campaign, hashtag correlation, channel_id, social_media_platform, and offers...")

# Define offers specifically for electronics
# General offers for electronics
general_offers_electronics = [
    "Flat 10% off on any electronics purchase",
    "Free expedited shipping on all orders over $1000",
    "Bundle and save: 5% off when you buy 2 or more electronic items",
    "Limited time flash sale: 15% off sitewide on all electronics",
    "Sign up for our newsletter and get $50 off your first electronics order"
]

# Category-specific offers for electronics (based on your product_category_map)
category_specific_offers_electronics = {
    "Mobile & Computing": [
        "Buy any smartphone, get a screen protector and case free",
        "Upgrade your laptop: Trade-in bonus + 10% off new model",
        "Bundle a tablet with a keyboard cover and get 20% off accessories",
        "Student discount: 10% off on all computing devices"
    ],
    "Wearables & Accessories": [
        "Buy a smartwatch, get an extra strap free",
        "25% off on all headphones when purchased with any mobile device",
        "Health tech special: 15% off any fitness tracker"
    ],
    "Entertainment & Gaming": [
        "Purchase any Smart TV and get a soundbar at 30% off",
        "Gaming console + 2 games bundle: Save $100",
        "Free 3-month streaming subscription with any TV purchase",
        "Buy a camera, get a free starter kit (bag, SD card)"
    ],
    "Smart Home & Appliances": [
        "Automate your home: 20% off any smart home hub with 2 devices",
        "Buy a robotic vacuum, get a free brush replacement kit",
        "Kitchen appliance bundle: Save 15% on any two items",
        "Security camera installation discount: 50% off labor"
    ]
}


# First, create a mapping of customer-product combinations to their sentiment data
sentiment_mapping = {}
# Ensure sentiment_df is defined and available from the prior block execution
if 'sentiment_df' in locals() and not sentiment_df.empty:
    for _, row in sentiment_df.iterrows():
        key = (row['customer_id'], row['product_id'])
        sentiment_mapping[key] = {
            'campaign_name': row['campaign_name'],
            'hashtags': row['hashtags'],
            'social_media_platform': row['social_media_platform']
        }

# Create a mapping of customer-product combinations to their transaction channel
transaction_channel_mapping = {}
# Ensure df is defined and available from the prior block execution
if 'df' in locals() and not df.empty:
    for _, row in df.iterrows():
        key = (row['customer_id'], row['product_id'])
        # If multiple transactions exist for same customer-product, use the most recent channel
        if key not in transaction_channel_mapping:
            transaction_channel_mapping[key] = row['channel_id']


# MODIFIED FUNNEL STAGES
funnel_stages = ["sent", "viewed", "clicked", "addedtpcart", "purchased"]
num_journeys = 2500
journey_data = []

# Define drop_stage_weights here, before the loop that uses it
# Adjust weights for the new, shorter funnel
drop_stage_weights = [0.30, 0.25, 0.20, 0.15, 0.10] # Total 1.0

for _ in range(num_journeys):
    cust = random.choice(customer_ids)
    prod = random.choice(product_ids)
    journey_id = str(uuid.uuid4())
    base_date = datetime(2023,1,1) + timedelta(days=random.randint(0,600))
    
    # Get campaign, hashtag and social media platform data if available from sentiment
    sentiment_key = (cust, prod)
    if sentiment_key in sentiment_mapping:
        campaign_name = sentiment_mapping[sentiment_key]['campaign_name']
        hashtags = sentiment_mapping[sentiment_key]['hashtags']
        social_media_platform = sentiment_mapping[sentiment_key]['social_media_platform']
    else:
        campaign_name = random.choice(campaign_names)
        hashtags = ' '.join(random.sample(hashtags_choices, random.randint(2,3)))
        social_media_platform = random.choice(social_media_platforms)
    
    # Get channel_id from transaction data if available, otherwise assign randomly
    channel_id = None
    if sentiment_key in transaction_channel_mapping:
        channel_id = transaction_channel_mapping[sentiment_key]
    else:
        # Assign channel based on stage and some logic
        channel_weights = {
            "Online": 0.6,
            "In-Store": 0.25,
            "B2B": 0.15
        }
        channel_id = random.choices(list(channel_weights.keys()), weights=list(channel_weights.values()))[0]

    # Determine branch_id, latitude, and longitude based on channel_id
    current_branch_id = None
    current_latitude = None
    current_longitude = None

    if channel_id == "Online":
        current_branch_id = online_branch_id # Always BR01 for online
        current_latitude = branch_geo[online_branch_id]["latitude"]
        current_longitude = branch_geo[online_branch_id]["longitude"]
    else:
        # For In-Store or B2B, pick a random physical branch
        # Exclude BR01 from physical store choices, assuming BR01 is solely online hub
        physical_branch_ids = [bid for bid in branch_ids if bid != online_branch_id]
        if physical_branch_ids: # Ensure there are physical branches to choose from
            current_branch_id = random.choice(physical_branch_ids)
            current_latitude = branch_geo[current_branch_id]["latitude"]
            current_longitude = branch_geo[current_branch_id]["longitude"]
        else: # Fallback if no physical branches are defined (shouldn't happen with your current setup)
            current_branch_id = "UNKNOWN"
            current_latitude = 0.0
            current_longitude = 0.0


    # Get product name and category using the product_id
    product_name = product_name_map.get(prod, "Unknown Product")
    product_category = product_category_map.get(product_name, "Unknown Category")

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
    product_in_cart = "No"
    
    for i, stage in enumerate(funnel_stages):
        if final_stage_reached is not None and funnel_stages.index(stage) > funnel_stages.index(final_stage_reached):
            break

        entry_date_dt = (base_date + timedelta(days=random.randint(5,15)*i, hours=random.randint(0,23), minutes=random.randint(0,59)))
        entry_date = entry_date_dt.strftime("%Y-%m-%d %H:%M:%S")

        year = entry_date_dt.year
        open_prob_increase = 0.0
        click_prob_increase = 0.0
        conversion_prob_increase = 0.0
        cart_prob_increase = 0.0
        
        if year == 2024:
            open_prob_increase = 0.03
            click_prob_increase = 0.05
            conversion_prob_increase = 0.02
            cart_prob_increase = 0.05
        elif year == 2025:
            open_prob_increase = 0.06
            click_prob_increase = 0.10
            conversion_prob_increase = 0.05
            cart_prob_increase = 0.10

        current_offer_applied = "No Offer" 

        if stage == "sent":
            campaign_open = "Yes"
            campaign_click = "No"
            product_in_cart = "No"
            conversion_flag = "No"
        elif stage == "viewed":
            if random.random() < (0.80 + open_prob_increase):
                campaign_open = "Yes"
            campaign_click = "No"
            product_in_cart = "No"
            conversion_flag = "No"
        elif stage == "clicked":
            if campaign_open == "Yes" and random.random() < (0.35 + click_prob_increase):
                campaign_click = "Yes"
            product_in_cart = "No"
            conversion_flag = "No"
        elif stage == "addedtpcart":
            product_in_cart_base_prob = 0.65
            product_in_cart_current_prob = min(1.0, product_in_cart_base_prob + cart_prob_increase)
            product_in_cart = np.random.choice(["Yes", "No"], p=[product_in_cart_current_prob, 1 - product_in_cart_current_prob])
            campaign_open = "Yes"
            campaign_click = "Yes"
            conversion_flag = "No"
            
            # Apply offer if product is added to cart (50% chance)
            if product_in_cart == "Yes" and random.random() < 0.5:
                # Prioritize category-specific offers if available
                if product_category in category_specific_offers_electronics and random.random() < 0.7:
                    current_offer_applied = random.choice(category_specific_offers_electronics[product_category])
                else:
                    current_offer_applied = random.choice(general_offers_electronics)

        elif stage == "purchased":
            if random.random() < (0.6 + conversion_prob_increase):
                conversion_flag = "Yes"
            else:
                conversion_flag = "No"
            campaign_open = "Yes"
            campaign_click = "Yes"
            product_in_cart = "Yes" # Assume added to cart if purchased

            # Apply offer if purchased (higher chance, 70%)
            if conversion_flag == "Yes" and random.random() < 0.7:
                if product_category in category_specific_offers_electronics and random.random() < 0.8:
                    current_offer_applied = random.choice(category_specific_offers_electronics[product_category])
                else:
                    current_offer_applied = random.choice(general_offers_electronics)

        journey_data.append({
            "journey_id": journey_id,
            "customer_id": cust,
            "product_id": prod,
            "product_name": product_name,
            "product_category": product_category,
            "channel_id": channel_id,
            "branch_id": current_branch_id, # Added branch_id
            "latitude": current_latitude,     # Added latitude
            "longitude": current_longitude,   # Added longitude
            "social_media_platform": social_media_platform,
            "stage": stage,
            "stage_date": entry_date,
            "campaign_name": campaign_name,
            "hashtags": hashtags,
            "campaign_open": campaign_open,
            "campaign_click": campaign_click,
            "conversion_flag": conversion_flag,
            "product_in_cart": product_in_cart,
            "offer_applied": current_offer_applied,
        })

journey_df = pd.DataFrame(journey_data)
journey_df['customer_age'] = journey_df['customer_id'].map(customer_age_map)
journey_df.to_csv("journey_entry.csv", index=False)
print("journey_entry.csv created successfully!")