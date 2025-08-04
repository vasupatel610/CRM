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
yearly_sales_growth_rate = 1.15 # 15% growth year-over-year

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
    base_num_days_per_month = 300
    if year == 2023:
        year_multiplier_for_transactions = 1.0
    elif year == 2024:
        year_multiplier_for_transactions = yearly_sales_growth_rate
    elif year == 2025:
        year_multiplier_for_transactions = yearly_sales_growth_rate ** 2

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

# ============================================================================
# STEP 1: GENERATE TRANSACTION DATA (Base for all correlations)
# ============================================================================
print("Generating correlated transaction data...")

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
    voucher_redeemed_flag = np.random.choice(["Yes", "No"], p=[0.35, 0.65])
    is_upsell = np.random.choice(["Yes", "No"], p=[0.45, 0.55])
    is_returned = "No"

    num_lines = random.randint(1, max_lines_per_transaction)
    used_products = set()

    for line_num in range(1, num_lines + 1):
        product_id = random.choice([pid for pid in product_ids if pid not in used_products])
        used_products.add(product_id)

        unit_price = product_unit_price_map[product_id]
        
        price_year_multiplier = 1.0
        if transaction_date.year == 2024:
            price_year_multiplier = 1.05
        elif transaction_date.year == 2025:
            price_year_multiplier = 1.10
        unit_price = round(unit_price * price_year_multiplier, 2)

        quantity = np.random.choice([1, 2, 3, 5], p=channel_quantity_weights[channel_id])
        if transaction_date.year == 2024:
            quantity = int(quantity * 1.05)
        elif transaction_date.year == 2025:
            quantity = int(quantity * 1.10)
        quantity = max(1, quantity)

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

# Add returned flags with improved logic
def get_monthly_return_rate(year):
    if year == 2023: return random.uniform(0.10, 0.18)
    elif year == 2024: return random.uniform(0.06, 0.12)
    else: return random.uniform(0.03, 0.08)

for month in df["month"].unique():
    month_idx = df[df["month"] == month].index
    if len(month_idx) > 0:
        year = df.loc[month_idx[0], "transaction_date"].year
        rate = get_monthly_return_rate(year)
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
        year_factor = 1.03
    elif year == 2025:
        year_factor = 1.06
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

# Add Churn Flag
customer_last_purchase = df.groupby('customer_id')['transaction_date'].max()
customer_total_returns = df[df['is_returned'] == 'Yes'].groupby('customer_id').size().fillna(0)
customer_data = pd.DataFrame(customer_ids, columns=['customer_id'])

customer_data['last_purchase_date'] = customer_data['customer_id'].map(customer_last_purchase)
customer_data['total_returns'] = customer_data['customer_id'].map(customer_total_returns).fillna(0)
customer_data['customer_age'] = customer_data['customer_id'].map(customer_age_map)

latest_transaction_date = df['transaction_date'].max()
customer_data['recency'] = (latest_transaction_date - customer_data['last_purchase_date']).dt.days

def get_churn_probability(row):
    prob = 0.05
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

df = df.merge(customer_data[['customer_id', 'is_churned']], on='customer_id', how='left')
df['is_churned'] = df.groupby('customer_id')['is_churned'].transform('first')

# Save main transaction data
output_file = "synthetic_transaction_data.csv"
df.to_csv(output_file, index=False)
print(f"Transaction dataset saved to: {output_file}")

# ============================================================================
# STEP 2: CREATE CUSTOMER-PRODUCT PURCHASE MAPPING (Base for correlations)
# ============================================================================
print("Creating customer-product purchase mapping...")

# Get unique customer-product combinations that actually purchased
actual_purchases = df[['customer_id', 'product_id', 'transaction_date', 'is_returned']].drop_duplicates()
print(f"Found {len(actual_purchases)} unique customer-product purchase combinations")

# Create a set for quick lookup
purchase_combinations = set(zip(actual_purchases['customer_id'], actual_purchases['product_id']))
print(f"Total unique customer-product pairs that made purchases: {len(purchase_combinations)}")

# ============================================================================
# STEP 3: GENERATE SENTIMENT DATA (Only for actual purchases, max 1 per customer-product)
# ============================================================================
print("Generating sentiment data only for actual purchases...")

# Import NLTK for sentiment analysis
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except nltk.downloader.DownloadError:
        nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
except ImportError:
    print("NLTK not available, creating simple sentiment scores")
    sia = None

social_media_platforms = ["Twitter", "Facebook", "Instagram", "TikTok", "Google", "YouTube"]
hashtags_choices = ["#electronics", "#newtech", "#gadgetreview", "#productlove", "#techdeals", "#unboxing", "#smartlife", "#innovate"]
campaign_names = ["HolidayTechSale", "SpringSavings", "GamingGearLaunch", "WorkFromHomeEssentials", "BackToSchoolTech"]

# Enhanced review samples
positive_long_reviews = [
    "I absolutely love this product! The quality is unmatched and it has exceeded all my expectations. The build quality feels premium and every feature works flawlessly.",
    "This device is a complete game changer for me. It works flawlessly and the battery life is absolutely fantastic, lasting way longer than advertised.",
    "The features and performance are incredible beyond what I imagined. The design is sleek and modern, fitting perfectly with my setup.",
    "Top-notch quality and excellent support from both the seller and manufacturer. The product feels solid and well-built.",
    "Fantastic product that has completely transformed my daily routine and made tasks so much easier. The performance is consistent and reliable.",
]

negative_long_reviews = [
    "I was really disappointed with this product after having high expectations. It did not perform as advertised and the build quality feels cheap.",
    "The item arrived damaged with visible scratches and dents, clearly not properly packaged for shipping. Customer service was unhelpful.",
    "Had high hopes for this product based on the reviews, but it kept malfunctioning and caused constant frustration.",
    "The delivery was delayed by several weeks without proper communication, and when the product finally arrived, it had several defects.",
    "Poor quality materials and even worse customer support. The product started having issues within days of purchase.",
]

neutral_long_reviews = [
    "The product is okay for the price point, though nothing particularly special about it. It meets the basic functionality requirements.",
    "Average experience overall with this purchase. The device does what it says it will do, but there are some minor issues.",
    "It's decent and serves its purpose, but lacks some features I was expecting based on the product description.",
    "Good enough for everyday use, though I wouldn't recommend it if you need something more robust or feature-rich.",
    "Fair product with average performance that meets minimum expectations. Some aspects are well-designed while others feel like afterthoughts.",
]

def generate_dynamic_review(sentiment, product_name):
    if sentiment == 'positive':
        base_reviews = positive_long_reviews
    elif sentiment == 'negative':
        base_reviews = negative_long_reviews
    else:
        base_reviews = neutral_long_reviews
    
    base_review = random.choice(base_reviews)
    
    # Add product-specific mention 40% of the time
    if random.random() < 0.4:
        product_mentions = [
            f" The {product_name} specifically impressed me.",
            f" This particular {product_name} model stands out.",
            f" Compared to other similar products, this {product_name} is different.",
        ]
        base_review += random.choice(product_mentions)
    
    return base_review

sentiment_data = []

# Only generate reviews for actual purchases (with 60% probability)
review_probability = 0.60
purchases_to_review = actual_purchases.sample(n=int(len(actual_purchases) * review_probability), random_state=42)

# Ensure only one review per customer-product combination
reviewed_combinations = set()

for _, purchase in purchases_to_review.iterrows():
    cust = purchase['customer_id']
    prod = purchase['product_id']
    combination_key = (cust, prod)
    
    # Skip if this customer-product combination already has a review
    if combination_key in reviewed_combinations:
        continue
    
    reviewed_combinations.add(combination_key)
    
    purchase_date = purchase['transaction_date']
    is_returned = purchase['is_returned']
    
    product_name = product_name_map.get(prod, "product")
    
    # Review date should be 1-15 days after purchase
    review_date = purchase_date + timedelta(days=random.randint(1, 15))
    
    if review_date.date() > current_date:
        review_date = datetime.combine(current_date, datetime.min.time())
    
    # Sentiment bias based on return status
    if is_returned == "Yes":
        sentiment_weights = [0.05, 0.85, 0.1]  # High probability of negative if returned
    else:
        year = review_date.year
        if year == 2023:
            sentiment_weights = [0.75, 0.05, 0.20]
        elif year == 2024:
            sentiment_weights = [0.80, 0.04, 0.16]
        elif year == 2025:
            sentiment_weights = [0.85, 0.03, 0.12]
        else:
            sentiment_weights = [0.75, 0.05, 0.20]
    
    review_type = random.choices(['positive', 'negative', 'neutral'], weights=sentiment_weights)[0]
    review = generate_dynamic_review(review_type, product_name)
    
    platform = random.choice(social_media_platforms)
    hashtags = ' '.join(random.sample(hashtags_choices, random.randint(2,4)))
    campaign = random.choice(campaign_names)
    
    sentiment_data.append({
        "customer_id": cust,
        "product_id": prod,
        "product_name": product_name,
        "reviews": review,
        "date": review_date.strftime("%Y-%m-%d"),
        "social_media_platform": platform,
        "hashtags": hashtags,
        "campaign_name": campaign,
        "review_sentiment": review_type
    })

sentiment_df = pd.DataFrame(sentiment_data)

# Apply sentiment scoring
if sia:
    sentiment_df['sentiment_score'] = sentiment_df['reviews'].apply(lambda x: sia.polarity_scores(x)['compound'])
else:
    # Simple sentiment scoring if NLTK not available
    def simple_sentiment_score(sentiment_type):
        if sentiment_type == 'positive':
            return random.uniform(0.3, 1.0)
        elif sentiment_type == 'negative':
            return random.uniform(-1.0, -0.2)
        else:
            return random.uniform(-0.2, 0.3)
    
    sentiment_df['sentiment_score'] = sentiment_df['review_sentiment'].apply(simple_sentiment_score)

sentiment_df['customer_age'] = sentiment_df['customer_id'].map(customer_age_map)
sentiment_df.to_csv("sentiment.csv", index=False)
print(f"Sentiment dataset saved with {len(sentiment_df)} reviews for actual purchases only")

# ============================================================================
# STEP 4: GENERATE JOURNEY DATA (Only for actual purchases)
# ============================================================================
print("Generating journey data only for actual customer-product purchases...")

# General offers for electronics
general_offers_electronics_map = {
    "Flat 10% off on any electronics purchase": "ELEC10",
    "Free expedited shipping on all orders over KES10000": "FREESHIP10K",
    "Bundle and save: 5% off when you buy 2 or more electronic items": "BUNDLE5",
    "Limited time flash sale: 15% off sitewide on all electronics": "FLASH15",
    "Sign up for our newsletter and get KES50 off your first electronics order": "NEWS50"
}

# Category-specific offers
category_specific_offers_electronics_map = {
    "Mobile & Computing": {
        "Buy any smartphone, get a screen protector and case free": "MOBACCSS",
        "Upgrade your laptop: Trade-in bonus + 10% off new model": "LAPTOPUPG",
        "Bundle a tablet with a keyboard cover and get 20% off accessories": "TABBUNDLE",
        "Student discount: 10% off on all computing devices": "STUDENTCOMP"
    },
    "Wearables & Accessories": {
        "Buy a smartwatch, get an extra strap free": "WATCHSTRAP",
        "25% off on all headphones when purchased with any mobile device": "HDPBUNDLE",
        "Health tech special: 15% off any fitness tracker": "FITNESS15"
    },
    "Entertainment & Gaming": {
        "Purchase any Smart TV and get a soundbar at 30% off": "TVSOUND30",
        "Gaming console + 2 games bundle: Save KES1000": "GAMECONS",
        "Free 3-month streaming subscription with any TV purchase": "TVSTREAM",
        "Buy a camera, get a free starter kit (bag, SD card)": "CAMKIT"
    },
    "Smart Home & Appliances": {
        "Automate your home: 20% off any smart home hub with 2 devices": "SMARTHUB20",
        "Buy a robotic vacuum, get a free brush replacement kit": "ROBOVAC",
        "Kitchen appliance bundle: Save 15% on any two items": "KITCHENBND",
        "Security camera installation discount: 50% off labor": "CAMINSTALL"
    }
}

# Flatten the offer maps
all_offers_descriptions = list(general_offers_electronics_map.keys())
all_offers_codes = list(general_offers_electronics_map.values())

for category, offers_dict in category_specific_offers_electronics_map.items():
    all_offers_descriptions.extend(list(offers_dict.keys()))
    all_offers_codes.extend(list(offers_dict.values()))

offer_description_to_code = {desc: code for desc, code in general_offers_electronics_map.items()}
for category, offers_dict in category_specific_offers_electronics_map.items():
    offer_description_to_code.update(offers_dict)

# Create sentiment mapping for journey correlation
sentiment_mapping = {}
if not sentiment_df.empty:
    for _, row in sentiment_df.iterrows():
        key = (row['customer_id'], row['product_id'])
        sentiment_mapping[key] = {
            'campaign_name': row['campaign_name'],
            'hashtags': row['hashtags'],
            'social_media_platform': row['social_media_platform']
        }

# Create transaction channel mapping
transaction_channel_mapping = {}
for _, row in df.iterrows():
    key = (row['customer_id'], row['product_id'])
    transaction_channel_mapping[key] = row['channel_id']

# Journey funnel stages
funnel_stages = ["sent", "viewed", "clicked", "addedtocart", "purchased"]
# drop_stage_weights = [0.20, 0.05, 0.15, 0.30, 0.30]  # Strong loyalty pattern
drop_stage_weights = [0.05, 0.03, 0.20, 0.50, 0.22]
journey_data = []

# Generate journeys only for actual customer-product purchases
# Use a subset to avoid too many journey entries
num_journeys_from_purchases = min(3000, len(actual_purchases))
journey_purchases = actual_purchases.sample(n=num_journeys_from_purchases, random_state=42)

print(f"Generating {num_journeys_from_purchases} journey entries from actual purchases...")

for _, purchase_row in journey_purchases.iterrows():
    cust = purchase_row['customer_id']
    prod = purchase_row['product_id']
    purchase_date = purchase_row['transaction_date']
    
    journey_id = str(uuid.uuid4())
    
    # Journey should start before the actual purchase date
    # Generate journey base date 1-30 days before purchase
    days_before_purchase = random.randint(1, 30)
    base_date = purchase_date - timedelta(days=days_before_purchase)
    
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
    
    # Get channel_id from transaction data
    channel_id = transaction_channel_mapping.get(sentiment_key, "Online")
    
    # Determine branch_id, latitude, and longitude based on channel_id
    if channel_id == "Online":
        current_branch_id = online_branch_id
        current_latitude = branch_geo[online_branch_id]["latitude"]
        current_longitude = branch_geo[online_branch_id]["longitude"]
    else:
        physical_branch_ids = [bid for bid in branch_ids if bid != online_branch_id]
        if physical_branch_ids:
            current_branch_id = random.choice(physical_branch_ids)
            current_latitude = branch_geo[current_branch_id]["latitude"]
            current_longitude = branch_geo[current_branch_id]["longitude"]
        else:
            current_branch_id = "UNKNOWN"
            current_latitude = 0.0
            current_longitude = 0.0

    # Get product details
    product_name = product_name_map.get(prod, "Unknown Product")
    product_category = product_category_map.get(product_name, "Unknown Category")

    # Since this customer actually purchased, they should complete the full journey
    # But we can still add some variation for realism
    complete_journey_prob = 0.45  # 85% chance of completing full journey for actual purchasers
    
    if random.random() < complete_journey_prob:
        final_stage_reached = "purchased"  # Complete journey
    else:
        # Incomplete journey - stop at random stage before purchase
        final_stage_reached = random.choice(funnel_stages[:-1])

    campaign_open = "No"
    campaign_click = "No"
    conversion_flag = "No"
    product_in_cart = "No"
    
    for i, stage in enumerate(funnel_stages):
        if final_stage_reached is not None and funnel_stages.index(stage) > funnel_stages.index(final_stage_reached):
            break

        # Journey stages should progress over time, ending before/at purchase date
        stage_date_dt = base_date + timedelta(
            days=random.randint(0, 5) * i,  # Shorter intervals between stages
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        # Ensure stage date doesn't exceed purchase date
        if stage_date_dt > purchase_date:
            stage_date_dt = purchase_date - timedelta(hours=random.randint(1, 24))
        
        stage_date = stage_date_dt.strftime("%Y-%m-%d %H:%M:%S")

        year = stage_date_dt.year
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
        current_offer_code = "NO_OFFER"

        if stage == "sent":
            campaign_open = "Yes"
            campaign_click = "No"
            product_in_cart = "No"
            conversion_flag = "No"
        elif stage == "viewed":
            if random.random() < (0.60 + open_prob_increase):
                campaign_open = "Yes"
            campaign_click = "No"
            product_in_cart = "No"
            conversion_flag = "No"
        elif stage == "clicked":
            if campaign_open == "Yes" and random.random() < (0.45 + click_prob_increase):
                campaign_click = "Yes"
            product_in_cart = "No"
            conversion_flag = "No"
        elif stage == "addedtocart":
            product_in_cart_base_prob = 0.45
            product_in_cart_current_prob = min(1.0, product_in_cart_base_prob + cart_prob_increase)
            product_in_cart = np.random.choice(["Yes", "No"], p=[product_in_cart_current_prob, 1 - product_in_cart_current_prob])
            campaign_open = "Yes"
            campaign_click = "Yes"
            conversion_flag = "No"
            
            # Apply offer if product is added to cart (50% chance)
            if product_in_cart == "Yes" and random.random() < 0.5:
                if product_category in category_specific_offers_electronics_map and random.random() < 0.7:
                    current_offer_applied = random.choice(list(category_specific_offers_electronics_map[product_category].keys()))
                else:
                    current_offer_applied = random.choice(list(general_offers_electronics_map.keys()))
                current_offer_code = offer_description_to_code[current_offer_applied]

        elif stage == "purchased":
            # Since we know this customer actually purchased, set conversion to Yes
            conversion_flag = "Yes"
            campaign_open = "Yes"
            campaign_click = "Yes"
            product_in_cart = "Yes"

            # Apply offer if purchased (higher chance, 85%)
            if random.random() < 0.85:
                if product_category in category_specific_offers_electronics_map and random.random() < 0.8:
                    current_offer_applied = random.choice(list(category_specific_offers_electronics_map[product_category].keys()))
                else:
                    current_offer_applied = random.choice(list(general_offers_electronics_map.keys()))
                current_offer_code = offer_description_to_code[current_offer_applied]

        journey_data.append({
            "journey_id": journey_id,
            "customer_id": cust,
            "product_id": prod,
            "product_name": product_name,
            "product_category": product_category,
            "channel_id": channel_id,
            "branch_id": current_branch_id,
            "latitude": current_latitude,
            "longitude": current_longitude,
            "social_media_platform": social_media_platform,
            "stage": stage,
            "stage_date": stage_date,
            "campaign_name": campaign_name,
            "hashtags": hashtags,
            "campaign_open": campaign_open,
            "campaign_click": campaign_click,
            "conversion_flag": conversion_flag,
            "product_in_cart": product_in_cart,
            "offer_applied": current_offer_applied,
            "offer_code": current_offer_code,
        })

journey_df = pd.DataFrame(journey_data)
journey_df['customer_age'] = journey_df['customer_id'].map(customer_age_map)
journey_df.to_csv("journey_entry.csv", index=False)
print(f"Journey dataset saved with {len(journey_df)} entries from actual purchases only")

# ============================================================================
# STEP 5: GENERATE AFTER-SALES DATA (Only for actual purchases)
# ============================================================================
print("Generating after-sales data only for actual purchases...")

num_after_sales_interactions = 1500
interaction_types = ["Call", "Chat", "Email", "Social Media DM", "In-Person Support"]
issue_categories_map = {
    "Product Defect": ["Hardware Failure", "Software Bug", "Manufacturing Defect"],
    "Technical Support": ["Setup Help", "Troubleshooting", "Connectivity Issue", "Software Update", "Compatibility"],
    "Billing Inquiry": ["Incorrect Charge", "Refund Status", "Subscription Issue", "Payment Error"],
    "Order Status": ["Delivery Delay", "Missing Item", "Wrong Item Received", "Cancellation"],
    "Return/Refund": ["Return Process", "Refund Delay", "Exchange Request", "Warranty Claim"],
    "General Inquiry": ["Product Information", "Store Hours", "Warranty Info", "How-To Guide"],
    "Service Delay": ["Long Wait Time", "Slow Resolution", "Agent Unresponsive"]
}
resolution_statuses = ["Resolved", "Pending", "Escalated", "Unresolved"]
agent_ids = [f"AGT{str(i).zfill(3)}" for i in range(1, 60)]

# Convert actual_purchases to list for easier sampling
customer_product_pairs_from_transactions = actual_purchases[['customer_id', 'product_id', 'transaction_date']].values.tolist()

# Combine sentiment data for prior sentiment lookup
customer_product_prior_sentiment_map = {}
if not sentiment_df.empty:
    customer_product_prior_sentiment_map = sentiment_df.set_index(['customer_id', 'product_id'])['sentiment_score'].to_dict()

def generate_sentiment_score(base_score=0.0):
    score = np.clip(random.gauss(base_score, 0.3), -1.0, 1.0)
    score = round(score, 2)
    if score > 0.4:
        category = "Positive"
    elif score < -0.2:
        category = "Negative"
    else:
        category = "Neutral"
    return score, category

def generate_nps_score(sentiment_cat):
    if sentiment_cat == "Negative":
        return random.randint(0, 6)
    elif sentiment_cat == "Neutral":
        return random.randint(7, 8)
    else:
        return random.randint(9, 10)

def generate_feedback_score(sentiment_cat, agent_quality=1.0):
    base_score = 3.0
    if sentiment_cat == "Negative":
        base_score = 1.5
    elif sentiment_cat == "Neutral":
        base_score = 3.0
    else:
        base_score = 4.5
    
    score = random.gauss(base_score * agent_quality, 0.8)
    return int(np.clip(round(score), 1, 5))

# Simulate Agent Performance Baseline
agent_performance_baseline = {
    agent_id: {
        'avg_resolution_time': random.randint(3, 20),
        'sla_met_prob': random.uniform(0.7, 0.98),
        'fcr_prob': random.uniform(0.5, 0.9),
        'agent_quality_factor': random.uniform(0.8, 1.2)
    } for agent_id in agent_ids
}

after_sales_data = []

# Only generate after-sales for customers who actually purchased
after_sales_sample_size = min(num_after_sales_interactions, len(customer_product_pairs_from_transactions))
sampled_purchases = random.sample(customer_product_pairs_from_transactions, after_sales_sample_size)

for selected_pair in sampled_purchases:
    cust_id, prod_id, transaction_date_of_purchase = selected_pair
    
    interaction_id = str(uuid.uuid4())

    # Interaction date should be AFTER the purchase date
    min_days_after_purchase = 7
    max_days_after_purchase = 180  # 6 months
    interaction_date_dt = transaction_date_of_purchase + timedelta(
        days=random.randint(min_days_after_purchase, max_days_after_purchase),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )
    
    # Ensure interaction date is not in the future
    if interaction_date_dt > datetime.now():
        interaction_date_dt = datetime.now() - timedelta(minutes=random.randint(1, 60))

    interaction_type = random.choice(interaction_types)
    agent_id = random.choice(agent_ids)
    
    agent_baseline = agent_performance_baseline[agent_id]

    # Get product details
    product_name = product_name_map.get(prod_id, "Unknown Product")
    product_cat = product_category_map.get(product_name, "General")
    
    # Issue Category & Subcategory (Influenced by Product Category)
    issue_prob_dist = {
        "Product Defect": 0.15, "Technical Support": 0.20, "Billing Inquiry": 0.10,
        "Order Status": 0.10, "Return/Refund": 0.10, "General Inquiry": 0.25, "Service Delay": 0.10
    }

    # Adjust probabilities based on product category
    if product_cat == "Mobile & Computing":
        issue_prob_dist["Technical Support"] += 0.15
        issue_prob_dist["Product Defect"] += 0.05
        issue_prob_dist["General Inquiry"] -= 0.1
    elif product_cat == "Entertainment & Gaming":
        issue_prob_dist["Technical Support"] += 0.1
        issue_prob_dist["Product Defect"] += 0.1
        issue_prob_dist["General Inquiry"] -= 0.05
    elif product_cat == "Smart Home & Appliances":
        issue_prob_dist["Product Defect"] += 0.15
        issue_prob_dist["Technical Support"] += 0.05
        issue_prob_dist["Service Delay"] += 0.05
        issue_prob_dist["General Inquiry"] -= 0.05

    # Normalize probabilities
    total_prob = sum(issue_prob_dist.values())
    normalized_probs = {k: v / total_prob for k, v in issue_prob_dist.items()}
    
    category = random.choices(list(normalized_probs.keys()), weights=list(normalized_probs.values()))[0]
    subcategory = random.choice(issue_categories_map[category])

    # Resolution Status & Time
    resolution_time = int(random.gauss(agent_baseline['avg_resolution_time'], 5))
    resolution_time = max(5, resolution_time)

    sla_target_minutes = 30
    sla_met = "Yes" if resolution_time <= sla_target_minutes and random.random() < agent_baseline['sla_met_prob'] else "No"
    
    is_fcr = "Yes" if random.random() < agent_baseline['fcr_prob'] else "No"
    is_escalated = "Yes" if (random.random() < 0.15 and sla_met == "No") or resolution_time > (sla_target_minutes * 1.5) else "No"

    resolution_status = "Resolved"
    if sla_met == "No" or is_fcr == "No" or is_escalated == "Yes" or random.random() < 0.1:
        resolution_status = random.choices(["Pending", "Escalated", "Unresolved", "Resolved"], weights=[0.15, 0.15, 0.05, 0.65])[0]
    
    # Sentiment (Influenced by Prior Sentiment and Resolution Outcome)
    prior_sentiment_score = customer_product_prior_sentiment_map.get((cust_id, prod_id), 0.0)
    base_sentiment_for_current_interaction = prior_sentiment_score

    if resolution_status != "Resolved" or sla_met == "No" or is_fcr == "No":
        base_sentiment_for_current_interaction -= random.uniform(0.2, 0.7)
    else:
        base_sentiment_for_current_interaction += random.uniform(0.2, 0.7)
    
    sentiment_score, sentiment_category = generate_sentiment_score(base_sentiment_for_current_interaction)
    
    nps_score = generate_nps_score(sentiment_category)
    feedback_agent = generate_feedback_score(sentiment_category, agent_baseline['agent_quality_factor'])
    feedback_resolution = generate_feedback_score(sentiment_category)

    summary = f"Customer reported {category.lower()} issue regarding {product_cat} product. Issue {resolution_status.lower()}."
    if is_escalated == "Yes":
        summary += " Escalated to Tier 2."
    
    call_duration_seconds = 0
    queue_time_seconds = 0
    if interaction_type in ["Call", "Chat"]:
        call_duration_seconds = resolution_time * 60 + random.randint(0, 10)
        queue_time_seconds = random.randint(0, 180)
        if sla_met == "No" or resolution_status != "Resolved":
            queue_time_seconds += random.randint(60, 480)

    after_sales_data.append({
        "interaction_id": interaction_id,
        "customer_id": cust_id,
        "product_id": prod_id,
        "product_name": product_name,
        "product_category": product_cat,
        "interaction_date": interaction_date_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "interaction_type": interaction_type,
        "agent_id": agent_id,
        "branch_id": current_branch_id,
        "latitude": branch_geo[current_branch_id]["latitude"],
        "longitude": branch_geo[current_branch_id]["longitude"],
        "grand_total": sum(df[(df['customer_id'] == cust_id) & (df['product_id'] == prod_id)]['grand_total']),
        "issue_category": category,
        "issue_subcategory": subcategory,
        "resolution_status": resolution_status,
        "resolution_time_minutes": resolution_time,
        "sla_met": sla_met,
        "is_first_contact_resolution": is_fcr,
        "is_escalated": is_escalated,
        "sentiment_score": sentiment_score,
        "sentiment_category": sentiment_category,
        "interaction_summary": summary,
        "nps_score": nps_score,
        "feedback_score_agent": feedback_agent,
        "feedback_score_resolution": feedback_resolution,
        "call_duration_seconds": call_duration_seconds,
        "queue_time_seconds": queue_time_seconds,
        "num_open_issues_snapshot": random.randint(0, 2) if resolution_status != "Resolved" else 0,
        "time_since_last_interaction_days": (datetime.now() - interaction_date_dt).days,
        "product_ownership_flag": "Yes"
    })

after_sales_df = pd.DataFrame(after_sales_data)

# Add agent status
agent_status_map = {agent: random.choice(["Online", "Offline"]) for agent in agent_ids}
after_sales_df['agent_status'] = after_sales_df['agent_id'].map(agent_status_map)

output_file = "after_sales.csv"
after_sales_df.to_csv(output_file, index=False)
print(f"After-sales dataset saved with {len(after_sales_df)} interactions from actual purchases only")

# ============================================================================
# SUMMARY AND VALIDATION
# ============================================================================
print("\n" + "="*80)
print("DATA GENERATION SUMMARY")
print("="*80)

print(f"✅ Transaction Data: {len(df)} transaction lines")
print(f"   - Unique customer-product combinations: {len(purchase_combinations)}")

print(f"✅ Sentiment Data: {len(sentiment_df)} reviews")
print(f"   - Only for customers who actually purchased products")
print(f"   - Maximum 1 review per customer-product combination")

print(f"✅ Journey Data: {len(journey_df)} journey entries")
print(f"   - Only for customers who actually purchased products")
print(f"   - Journey dates precede actual purchase dates")

print(f"✅ After-Sales Data: {len(after_sales_df)} interactions")
print(f"   - Only for customers who actually purchased products")
print(f"   - Interaction dates follow purchase dates")

print("\n" + "="*80)
print("CORRELATION VALIDATION")
print("="*80)

# Validate correlations
print("\n1. Checking sentiment-transaction correlation...")
sentiment_combinations = set(zip(sentiment_df['customer_id'], sentiment_df['product_id']))
invalid_sentiment = sentiment_combinations - purchase_combinations
print(f"   Invalid sentiment entries (no purchase): {len(invalid_sentiment)}")

print("\n2. Checking journey-transaction correlation...")
journey_combinations = set(zip(journey_df['customer_id'], journey_df['product_id']))
invalid_journey = journey_combinations - purchase_combinations
print(f"   Invalid journey entries (no purchase): {len(invalid_journey)}")

print("\n3. Checking after-sales-transaction correlation...")
after_sales_combinations = set(zip(after_sales_df['customer_id'], after_sales_df['product_id']))
invalid_after_sales = after_sales_combinations - purchase_combinations
print(f"   Invalid after-sales entries (no purchase): {len(invalid_after_sales)}")

print("\n4. Checking for duplicate sentiment reviews...")
sentiment_duplicates = sentiment_df.groupby(['customer_id', 'product_id']).size()
duplicate_reviews = sentiment_duplicates[sentiment_duplicates > 1]
print(f"   Duplicate reviews (should be 0): {len(duplicate_reviews)}")

if len(invalid_sentiment) == 0 and len(invalid_journey) == 0 and len(invalid_after_sales) == 0 and len(duplicate_reviews) == 0:
    print("\n🎉 SUCCESS: All data is properly correlated!")
    print("   - No sentiment reviews without purchases")
    print("   - No journey entries without purchases") 
    print("   - No after-sales interactions without purchases")
    print("   - No duplicate reviews per customer-product")
else:
    print("\n⚠️  WARNING: Some correlation issues found - check the validation results above")

print("\nFiles generated:")
print("- synthetic_transaction_data.csv")
print("- sentiment.csv") 
print("- journey_entry.csv")
print("- after_sales.csv")