from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid

# Configuration
num_transactions = 1000
max_lines_per_transaction = 5
num_customers = 300
num_products = 50
num_branches = 10
channels = ["Online", "In-Store", "B2B"]
payment_modes = ["Cash", "Card", "BNPL"]

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
   
    "Dyson V15 Detect": "Smart Home & Appliances", "Roomba i7+": "Smart Home & Appliances",
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

month_weights = {1: 0.9, 2: 0.8, 3: 1.0, 4: 1.1, 5: 0.9, 6: 1.2, 7: 1.3, 8: 0.9, 9: 1.4, 10: 1.6, 11: 1.8, 12: 2.2}
current_date = datetime.now().date()
date_pool = []
for year in [2023, 2024, 2025]:
    for month, weight in month_weights.items():
        for _ in range(int(weight * 100)):
            day = random.randint(1, 28)
            date = datetime(year, month, day).date()
            if date <= current_date:
                date_pool.append(datetime(year, month, day))

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
    voucher_redeemed_flag = random.choice(["Yes", "No"])
    is_upsell = random.choice(["Yes", "No"])
    is_returned = "No"

    num_lines = random.randint(1, max_lines_per_transaction)
    used_products = set()

    for line_num in range(1, num_lines + 1):
        product_id = random.choice([pid for pid in product_ids if pid not in used_products])
        used_products.add(product_id)

        unit_price = product_unit_price_map[product_id]
        quantity = np.random.choice([1, 2, 3, 5], p=channel_quantity_weights[channel_id])
        discount_rate_on_line = np.random.choice([0, 0.02, 0.05, 0.1], p=[0.7, 0.15, 0.1, 0.05])
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
    rate = random.uniform(0.05, 0.14)
    n_return = int(rate * len(month_idx))
    if n_return > 0:
        returned_idx = np.random.choice(month_idx, n_return, replace=False)
        df.loc[returned_idx, "is_returned"] = "Yes"

# Monthly targets
monthly_target_multipliers = {1: 1.15, 2: 1.25, 3: 1.18, 4: 1.12, 5: 1.20, 6: 1.10, 7: 1.08,
                              8: 1.22, 9: 1.05, 10: 1.07, 11: 1.09, 12: 1.03}
df["target_multiplier"] = df["month"].map(monthly_target_multipliers)

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

social_media_platforms = ["Twitter", "Facebook", "Instagram", "TikTok", "LinkedIn"]
hashtags_choices = ["#electronics", "#newin", "#techsale", "#review", "#unboxing", "#mustbuy", "#promotion"]
campaign_names = ["NewYearPromo", "SummerSale", "TechFest", "EndOfYearDeal", "LaunchEvent"]

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
review_probability = 0.45  # 45% of purchases get reviewed
purchases_to_review = customer_product_purchases.sample(n=int(len(customer_product_purchases) * review_probability))

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
        # Returned items more likely to have negative reviews
        sentiment_weights = [0.05, 0.75, 0.2]  # [positive, negative, neutral]
    else:
        # Non-returned items more likely to have positive reviews
        sentiment_weights = [0.65, 0.1, 0.25]  # [positive, negative, neutral]
    
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
        "reviews": review,
        "date": review_date.strftime("%Y-%m-%d"),
        "social_media_platform": platform,
        "hashtags": hashtags,
        "campaign_name": campaign,
        "review_sentiment": review_type  # Added for analysis purposes
    })

sentiment_df = pd.DataFrame(sentiment_data)
sentiment_df.to_csv("sentiment.csv", index=False)

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

funnel_stages = ["Lead", "Sale", "Service", "Loyalty"]
num_journeys = 900
journey_data = []

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
        # If no sentiment data, assign random campaign and hashtags
        campaign_name = random.choice(campaign_names)
        hashtags = ' '.join(random.sample(hashtags_choices, random.randint(2,3)))
    
    # Simulate funnel drop-offs (25% drop at each stage, with 5% completing full journey)
    drop_stage = random.choices(funnel_stages + [None], weights=[0.25,0.25,0.25,0.20,0.05])[0]
    
    for i, stage in enumerate(funnel_stages):
        if drop_stage and funnel_stages.index(drop_stage) < i:
            break  # Customer dropped off before this stage
        entry_date = (base_date + timedelta(days=random.randint(1,10)*i)).strftime("%Y-%m-%d")
        journey_data.append({
            "journey_id": journey_id,
            "customer_id": cust,
            "product_id": prod,
            "stage": stage,
            "stage_date": entry_date,
            "campaign_name": campaign_name,
            "hashtags": hashtags
        })

journey_df = pd.DataFrame(journey_data)
journey_df.to_csv("journey_entry.csv", index=False)
print("journey_entry.csv created successfully!")

print("\n=== ALL FILES GENERATED ===")
print("1. synthetic_transaction_data.csv - Main transaction dataset")
print("2. sentiment.csv - Customer sentiment and social media data")
print("3. journey_entry.csv - Customer journey funnel data")
