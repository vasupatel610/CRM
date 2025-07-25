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
    "JBL Flip 6": "Entertainment & Gaming", "Canon EOS R6": "Entertainment & Gaming", # Added from the products list
    "Nikon D7500": "Entertainment & Gaming", # Added from the products list
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


# Save
output_file = "synthetic_transaction_data.csv"
df.to_csv(output_file, index=False)
print(f"Dataset saved to: {output_file}")

# from matplotlib import pyplot as plt
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import random
# import uuid

# # Configuration
# num_transactions = 1000
# max_lines_per_transaction = 5
# num_customers = 300
# num_products = 50
# num_branches = 10
# channels = ["Online", "In-Store", "B2B"]
# payment_modes = ["Cash", "Card", "BNPL"]

# # Electronic products list (50 products)
# electronic_products = [
#     "iPhone 15 Pro", "Samsung Galaxy S24", "MacBook Air M3", "Dell XPS 13", "HP Pavilion",
#     "Sony WH-1000XM5", "AirPods Pro", "JBL Flip 6", "Canon EOS R6", "Nikon D7500",
#     "iPad Pro 12.9", "Samsung Tab S9", "Microsoft Surface", "Asus ROG Laptop", "Lenovo ThinkPad",
#     "Apple Watch Series 9", "Samsung Galaxy Watch", "Fitbit Charge 6", "Garmin Venu 3", "OnePlus Watch",
#     "LG OLED TV 55", "Samsung QLED 65", "Sony Bravia 43", "TCL 4K Smart TV", "Hisense ULED",
#     "PlayStation 5", "Xbox Series X", "Nintendo Switch", "Steam Deck", "Asus ROG Ally",
#     "Dyson V15 Detect", "Roomba i7+", "Philips Air Fryer", "Instant Pot Duo", "KitchenAid Mixer",
#     "Bose SoundLink", "Marshall Acton", "Sonos One", "Echo Dot 5th Gen", "Google Nest Hub",
#     "Ring Video Doorbell", "Arlo Pro 4", "Nest Cam", "Wyze Cam v3", "Eufy Security",
#     "Tesla Model Y Charger", "Anker PowerBank", "Belkin Wireless Charger", "Logitech MX Master", "Razer DeathAdder"
# ]

# # Product category mapping
# product_category_map = {
#     "iPhone 15 Pro": "Mobile & Computing", "Samsung Galaxy S24": "Mobile & Computing",
#     "MacBook Air M3": "Mobile & Computing", "Dell XPS 13": "Mobile & Computing",
#     "HP Pavilion": "Mobile & Computing", "Microsoft Surface": "Mobile & Computing",
#     "Asus ROG Laptop": "Mobile & Computing", "Lenovo ThinkPad": "Mobile & Computing",
#     "iPad Pro 12.9": "Mobile & Computing", "Samsung Tab S9": "Mobile & Computing",
#     "Logitech MX Master": "Mobile & Computing", "Razer DeathAdder": "Mobile & Computing",
#     "Anker PowerBank": "Mobile & Computing", "Belkin Wireless Charger": "Mobile & Computing",
    
#     "Apple Watch Series 9": "Wearables & Accessories", "Samsung Galaxy Watch": "Wearables & Accessories",
#     "Fitbit Charge 6": "Wearables & Accessories", "Garmin Venu 3": "Wearables & Accessories",
#     "OnePlus Watch": "Wearables & Accessories", "AirPods Pro": "Wearables & Accessories",
#     "Sony WH-1000XM5": "Wearables & Accessories",
    
#     "LG OLED TV 55": "Entertainment & Gaming", "Samsung QLED 65": "Entertainment & Gaming",
#     "Sony Bravia 43": "Entertainment & Gaming", "TCL 4K Smart TV": "Entertainment & Gaming",
#     "Hisense ULED": "Entertainment & Gaming", "PlayStation 5": "Entertainment & Gaming",
#     "Xbox Series X": "Entertainment & Gaming", "Nintendo Switch": "Entertainment & Gaming",
#     "Steam Deck": "Entertainment & Gaming", "Asus ROG Ally": "Entertainment & Gaming",
#     "Bose SoundLink": "Entertainment & Gaming", "Marshall Acton": "Entertainment & Gaming",
#     "Sonos One": "Entertainment & Gaming",
    
#     "Dyson V15 Detect": "Smart Home & Appliances", "Roomba i7+": "Smart Home & Appliances",
#     "Philips Air Fryer": "Smart Home & Appliances", "Instant Pot Duo": "Smart Home & Appliances",
#     "KitchenAid Mixer": "Smart Home & Appliances", "Echo Dot 5th Gen": "Smart Home & Appliances",
#     "Google Nest Hub": "Smart Home & Appliances", "Ring Video Doorbell": "Smart Home & Appliances",
#     "Arlo Pro 4": "Smart Home & Appliances", "Nest Cam": "Smart Home & Appliances",
#     "Wyze Cam v3": "Smart Home & Appliances", "Eufy Security": "Smart Home & Appliances",
#     "Tesla Model Y Charger": "Smart Home & Appliances"
# }

# # Generate mapping tables
# customer_ids = [f"CUST{str(i).zfill(4)}" for i in range(1, num_customers + 1)]
# product_ids = [f"PROD{str(i).zfill(3)}" for i in range(1, num_products + 1)]
# branch_ids = [f"BR{str(i).zfill(2)}" for i in range(1, num_branches + 1)]

# product_name_map = {product_ids[i]: electronic_products[i] for i in range(num_products)}

# online_branch_id = "BR01"
# online_staff_id = "STF01"
# staff_ids = {branch_id: f"STF{branch_id[-2:]}" for branch_id in branch_ids}

# # Base price mapping
# product_price_ranges = {name: (min_p, max_p) for name, (min_p, max_p) in zip(electronic_products, [
#     (120000,180000),(90000,150000),(150000,250000),(80000,150000),(70000,160000),
#     (40000,60000),(35000,50000),(12000,18000),(250000,350000),(120000,180000),
#     (140000,220000),(100000,170000),(90000,180000),(150000,400000),(80000,190000),
#     (60000,80000),(25000,45000),(20000,30000),(60000,75000),(15000,25000),
#     (120000,250000),(180000,300000),(50000,90000),(40000,80000),(70000,200000),
#     (70000,90000),(65000,85000),(40000,60000),(80000,120000),(130000,160000),
#     (100000,150000),(70000,120000),(15000,30000),(20000,45000),(60000,90000),
#     (25000,70000),(30000,50000),(25000,40000),(10000,15000),(30000,45000),
#     (15000,30000),(40000,60000),(20000,35000),(8000,15000),(9000,20000),
#     (50000,80000),(6000,20000),(4000,18000),(10000,18000),(5000,12000)
# ])}

# # Add product category mapping
# product_category_map = {
#     # --- Mobile & Computing ---
#     "iPhone 15 Pro": "Mobile & Computing",
#     "Samsung Galaxy S24": "Mobile & Computing",
#     "MacBook Air M3": "Mobile & Computing",
#     "Dell XPS 13": "Mobile & Computing",
#     "HP Pavilion": "Mobile & Computing",
#     "iPad Pro 12.9": "Mobile & Computing",
#     "Samsung Tab S9": "Mobile & Computing",
#     "Microsoft Surface": "Mobile & Computing",
#     "Asus ROG Laptop": "Mobile & Computing",
#     "Lenovo ThinkPad": "Mobile & Computing",
#     "Logitech MX Master": "Mobile & Computing",
#     "Razer DeathAdder": "Mobile & Computing",
#     "Anker PowerBank": "Mobile & Computing",
#     "Belkin Wireless Charger": "Mobile & Computing",

#     # --- Wearables ---
#     "Sony WH-1000XM5": "Wearables",
#     "AirPods Pro": "Wearables",
#     "Apple Watch Series 9": "Wearables",
#     "Samsung Galaxy Watch": "Wearables",
#     "Fitbit Charge 6": "Wearables",
#     "Garmin Venu 3": "Wearables",
#     "OnePlus Watch": "Wearables",

#     # --- Entertainment & Gaming ---
#     "JBL Flip 6": "Entertainment & Gaming",
#     "Canon EOS R6": "Entertainment & Gaming",
#     "Nikon D7500": "Entertainment & Gaming",
#     "LG OLED TV 55": "Entertainment & Gaming",
#     "Samsung QLED 65": "Entertainment & Gaming",
#     "Sony Bravia 43": "Entertainment & Gaming",
#     "TCL 4K Smart TV": "Entertainment & Gaming",
#     "Hisense ULED": "Entertainment & Gaming",
#     "PlayStation 5": "Entertainment & Gaming",
#     "Xbox Series X": "Entertainment & Gaming",
#     "Nintendo Switch": "Entertainment & Gaming",
#     "Steam Deck": "Entertainment & Gaming",
#     "Asus ROG Ally": "Entertainment & Gaming",
#     "Bose SoundLink": "Entertainment & Gaming",
#     "Marshall Acton": "Entertainment & Gaming",
#     "Sonos One": "Entertainment & Gaming",

#     # --- Smart Home & Appliances ---
#     "Dyson V15 Detect": "Smart Home & Appliances",
#     "Roomba i7+": "Smart Home & Appliances",
#     "Philips Air Fryer": "Smart Home & Appliances",
#     "Instant Pot Duo": "Smart Home & Appliances",
#     "KitchenAid Mixer": "Smart Home & Appliances",
#     "Echo Dot 5th Gen": "Smart Home & Appliances",
#     "Google Nest Hub": "Smart Home & Appliances",
#     "Ring Video Doorbell": "Smart Home & Appliances",
#     "Arlo Pro 4": "Smart Home & Appliances",
#     "Nest Cam": "Smart Home & Appliances",
#     "Wyze Cam v3": "Smart Home & Appliances",
#     "Eufy Security": "Smart Home & Appliances",
#     "Tesla Model Y Charger": "Smart Home & Appliances"
# }


# product_base_price_map = {
#     pid: round(random.uniform(product_price_ranges[product_name_map[pid]][0], product_price_ranges[product_name_map[pid]][1]), 2)
#     for pid in product_ids
# }
# product_unit_price_map = {pid: product_base_price_map[pid] for pid in product_ids}

# month_weights = {1: 0.9, 2: 0.8, 3: 1.0, 4: 1.1, 5: 0.9, 6: 1.2, 7: 1.3, 8: 0.9, 9: 1.4, 10: 1.6, 11: 1.8, 12: 2.2}
# current_date = datetime.now().date()
# date_pool = []
# for year in [2023, 2024, 2025]:
#     for month, weight in month_weights.items():
#         for _ in range(int(weight * 100)):
#             day = random.randint(1, 28)
#             date = datetime(year, month, day).date()
#             if date <= current_date:
#                 date_pool.append(datetime(year, month, day))

# channel_quantity_weights = {
#     "Online": [0.6, 0.3, 0.08, 0.02],
#     "B2B": [0.3, 0.4, 0.25, 0.05],
#     "In-Store": [0.4, 0.35, 0.2, 0.05]
# }

# data = []
# for _ in range(num_transactions):
#     transaction_id = str(uuid.uuid4())
#     transaction_date = random.choice(date_pool)
#     customer_id = random.choice(customer_ids)
#     channel_id = random.choice(channels)

#     branch_id = online_branch_id if channel_id == "Online" else random.choice(branch_ids)
#     staff_id = online_staff_id if channel_id == "Online" else staff_ids[branch_id]

#     allowed_payments = ["Cash", "Card"] if channel_id == "In-Store" else (["Card", "BNPL"] if channel_id == "Online" else payment_modes)
#     payment_mode = random.choice(allowed_payments)
#     voucher_redeemed_flag = random.choice(["Yes", "No"])
#     is_upsell = random.choice(["Yes", "No"])
#     is_returned = "No"

#     num_lines = random.randint(1, max_lines_per_transaction)
#     used_products = set()

#     for line_num in range(1, num_lines + 1):
#         product_id = random.choice([pid for pid in product_ids if pid not in used_products])
#         used_products.add(product_id)

#         unit_price = product_unit_price_map[product_id]
#         quantity = np.random.choice([1, 2, 3, 5], p=channel_quantity_weights[channel_id])
#         discount_rate_on_line = np.random.choice([0, 0.02, 0.05, 0.1], p=[0.7, 0.15, 0.1, 0.05])
#         discount_applied = round(discount_rate_on_line * unit_price * quantity, 2)
#         product_name = product_name_map[product_id]
#         product_category = product_category_map[product_name]

#         data.append({
#             "transaction_id": transaction_id,
#             "transaction_line_id": f"{transaction_id[:8]}-{line_num}",
#             "customer_id": customer_id,
#             "product_id": product_id,
#             "product_name": product_name,
#             # "product_category": product_category,
#             "product_category": product_category_map[product_name_map[product_id]],
#             "branch_id": branch_id,
#             "staff_id": staff_id,
#             "channel_id": channel_id,
#             "transaction_date": transaction_date.strftime("%Y-%m-%d %H:%M:%S"),
#             "quantity": quantity,
#             "unit_price": unit_price,
#             "discount_applied": discount_applied,
#             "payment_mode": payment_mode,
#             "voucher_redeemed_flag": voucher_redeemed_flag,
#             "is_upsell": is_upsell,
#             "is_returned": is_returned
#         })

# df = pd.DataFrame(data)
# df["grand_total"] = (df["unit_price"] * df["quantity"]) - df["discount_applied"]
# df["net_price"] = df["grand_total"] / df["quantity"]
# df["transaction_date"] = pd.to_datetime(df["transaction_date"])
# df["month"] = df["transaction_date"].dt.month

# # Add returned flags
# for month in df["month"].unique():
#     month_idx = df[df["month"] == month].index
#     rate = random.uniform(0.05, 0.14)
#     n_return = int(rate * len(month_idx))
#     if n_return > 0:
#         returned_idx = np.random.choice(month_idx, n_return, replace=False)
#         df.loc[returned_idx, "is_returned"] = "Yes"

# # Monthly targets
# monthly_target_multipliers = {1: 1.15, 2: 1.25, 3: 1.18, 4: 1.12, 5: 1.20, 6: 1.10, 7: 1.08,
#                               8: 1.22, 9: 1.05, 10: 1.07, 11: 1.09, 12: 1.03}
# df["target_multiplier"] = df["month"].map(monthly_target_multipliers)

# # Add geolocation
# random.seed(42)
# branch_ids_unique = df["branch_id"].unique()
# kenya_locations = [
#     {"lat": -1.2921, "lon": 36.8219}, {"lat": -1.3731, "lon": 36.8569}, {"lat": -4.0435, "lon": 39.6682},
#     {"lat": -0.3031, "lon": 35.2961}, {"lat": -0.0917, "lon": 34.7680}, {"lat": -0.4906, "lon": 35.2719},
#     {"lat": 0.5143, "lon": 35.2698}, {"lat": -1.5177, "lon": 37.2634}, {"lat": -0.6743, "lon": 34.5615},
#     {"lat": -3.2175, "lon": 40.1167}
# ]

# branch_geo = {}
# for i, branch_id in enumerate(branch_ids_unique):
#     if i < len(kenya_locations):
#         lat_variation = random.uniform(-0.01, 0.01)
#         lon_variation = random.uniform(-0.01, 0.01)
#         branch_geo[branch_id] = {
#             "latitude": round(kenya_locations[i]["lat"] + lat_variation, 6),
#             "longitude": round(kenya_locations[i]["lon"] + lon_variation, 6)
#         }
#     else:
#         branch_geo[branch_id] = {
#             "latitude": round(random.uniform(-4.5, 1.0), 6),
#             "longitude": round(random.uniform(33.9, 41.9), 6)
#         }

# df["latitude"] = df["branch_id"].map(lambda x: branch_geo[x]["latitude"])
# df["longitude"] = df["branch_id"].map(lambda x: branch_geo[x]["longitude"])

# # Save
# output_file = "synthetic_transaction_data.csv"
# df.to_csv(output_file, index=False)
# print(f"Dataset saved to: {output_file}")



# from matplotlib import pyplot as plt
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import random
# import uuid

# # Configuration
# num_transactions = 1000
# max_lines_per_transaction = 5
# num_customers = 300
# num_products = 50
# num_branches = 10
# channels = ["Online", "In-Store", "B2B"]
# payment_modes = ["Cash", "Card", "BNPL"]

# # Electronic products list (50 products)
# electronic_products = [
#     "iPhone 15 Pro", "Samsung Galaxy S24", "MacBook Air M3", "Dell XPS 13", "HP Pavilion",
#     "Sony WH-1000XM5", "AirPods Pro", "JBL Flip 6", "Canon EOS R6", "Nikon D7500",
#     "iPad Pro 12.9", "Samsung Tab S9", "Microsoft Surface", "Asus ROG Laptop", "Lenovo ThinkPad",
#     "Apple Watch Series 9", "Samsung Galaxy Watch", "Fitbit Charge 6", "Garmin Venu 3", "OnePlus Watch",
#     "LG OLED TV 55", "Samsung QLED 65", "Sony Bravia 43", "TCL 4K Smart TV", "Hisense ULED",
#     "PlayStation 5", "Xbox Series X", "Nintendo Switch", "Steam Deck", "Asus ROG Ally",
#     "Dyson V15 Detect", "Roomba i7+", "Philips Air Fryer", "Instant Pot Duo", "KitchenAid Mixer",
#     "Bose SoundLink", "Marshall Acton", "Sonos One", "Echo Dot 5th Gen", "Google Nest Hub",
#     "Ring Video Doorbell", "Arlo Pro 4", "Nest Cam", "Wyze Cam v3", "Eufy Security",
#     "Tesla Model Y Charger", "Anker PowerBank", "Belkin Wireless Charger", "Logitech MX Master", "Razer DeathAdder"
# ]

# # Generate mapping tables
# customer_ids = [f"CUST{str(i).zfill(4)}" for i in range(1, num_customers + 1)]
# product_ids = [f"PROD{str(i).zfill(3)}" for i in range(1, num_products + 1)]
# branch_ids = [f"BR{str(i).zfill(2)}" for i in range(1, num_branches + 1)]

# # Create product mapping with electronic product names
# product_name_map = {product_ids[i]: electronic_products[i] for i in range(num_products)}

# # For online purchases, use same store_id and staff_id
# online_branch_id = "BR01"
# online_staff_id = "STF01"

# # Regular staff mapping for other branches
# staff_ids = {branch_id: f"STF{branch_id[-2:]}" for branch_id in branch_ids}

# # --- Realistic Product Price Ranges in KSh (Base Prices) ---
# product_price_ranges = {
#     "iPhone 15 Pro": (120000, 180000),
#     "Samsung Galaxy S24": (90000, 150000),
#     "MacBook Air M3": (150000, 250000),
#     "Dell XPS 13": (80000, 150000),
#     "HP Pavilion": (70000, 160000),
#     "Sony WH-1000XM5": (40000, 60000),
#     "AirPods Pro": (35000, 50000),
#     "JBL Flip 6": (12000, 18000),
#     "Canon EOS R6": (250000, 350000),
#     "Nikon D7500": (120000, 180000),
#     "iPad Pro 12.9": (140000, 220000),
#     "Samsung Tab S9": (100000, 170000),
#     "Microsoft Surface": (90000, 180000),
#     "Asus ROG Laptop": (150000, 400000),
#     "Lenovo ThinkPad": (80000, 190000),
#     "Apple Watch Series 9": (60000, 80000),
#     "Samsung Galaxy Watch": (25000, 45000),
#     "Fitbit Charge 6": (20000, 30000),
#     "Garmin Venu 3": (60000, 75000),
#     "OnePlus Watch": (15000, 25000),
#     "LG OLED TV 55": (120000, 250000),
#     "Samsung QLED 65": (180000, 300000),
#     "Sony Bravia 43": (50000, 90000),
#     "TCL 4K Smart TV": (40000, 80000),
#     "Hisense ULED": (70000, 200000),
#     "PlayStation 5": (70000, 90000),
#     "Xbox Series X": (65000, 85000),
#     "Nintendo Switch": (40000, 60000),
#     "Steam Deck": (80000, 120000),
#     "Asus ROG Ally": (130000, 160000),
#     "Dyson V15 Detect": (100000, 150000),
#     "Roomba i7+": (70000, 120000),
#     "Philips Air Fryer": (15000, 30000),
#     "Instant Pot Duo": (20000, 45000),
#     "KitchenAid Mixer": (60000, 90000),
#     "Bose SoundLink": (25000, 70000),
#     "Marshall Acton": (30000, 50000),
#     "Sonos One": (25000, 40000),
#     "Echo Dot 5th Gen": (10000, 15000),
#     "Google Nest Hub": (30000, 45000),
#     "Ring Video Doorbell": (15000, 30000),
#     "Arlo Pro 4": (40000, 60000),
#     "Nest Cam": (20000, 35000),
#     "Wyze Cam v3": (8000, 15000),
#     "Eufy Security": (9000, 20000),
#     "Tesla Model Y Charger": (50000, 80000),
#     "Anker PowerBank": (6000, 20000),
#     "Belkin Wireless Charger": (4000, 18000),
#     "Logitech MX Master": (10000, 18000),
#     "Razer DeathAdder": (5000, 12000)
# }

# # Generate base product price map
# product_base_price_map = {
#     pid: round(random.uniform(product_price_ranges[product_name_map[pid]][0], product_price_ranges[product_name_map[pid]][1]), 2)
#     for pid in product_ids
# }

# # --- NEW: Fixed unit_price for each product_id ---
# # This ensures that for a given product_id, its unit_price will always be the same.
# # We'll consider the unit_price as the standard selling price *before* any transaction-specific discounts.
# product_unit_price_map = {
#     pid: product_base_price_map[pid]
#     for pid in product_ids
# }


# # Enhanced month weights with more realistic seasonal variation
# month_weights = {
#     1: 0.9,     # Jan - Post-holiday slowdown
#     2: 0.8,     # Feb - Lowest sales
#     3: 1.0,     # Mar - Recovery
#     4: 1.1,     # Apr - Spring growth
#     5: 0.9,     # May - Slight dip
#     6: 1.2,     # Jun - Mid-year push
#     7: 1.3,     # Jul - Summer peak
#     8: 0.9,     # Aug - Summer lull
#     9: 1.4,     # Sep - Back-to-school/work
#     10: 1.6,    # Oct - Festival season
#     11: 1.8,    # Nov - Pre-holiday shopping
#     12: 2.2     # Dec - Holiday peak
# }

# # Create weighted list of dates from 2023 till today
# current_date = datetime.now().date()
# date_pool = []

# # Generate dates for 2023, 2024, and 2025 (up to today)
# for year in [2023, 2024, 2025]:
#     for month, weight in month_weights.items():
#         for _ in range(int(weight * 100)):
#             day = random.randint(1, 28)
#             try:
#                 date = datetime(year, month, day).date()
#                 if date <= current_date:  # Only add dates up to and including today
#                     date_pool.append(datetime(year, month, day))
#             except ValueError:
#                 continue

# # Initialize data container
# data = []

# # Channel quantity multipliers to create grand_total differences
# channel_quantity_weights = {
#     "Online": [0.6, 0.3, 0.08, 0.02],    # More single items (lower grand_total)
#     "B2B": [0.3, 0.4, 0.25, 0.05],       # More bulk purchases (medium grand_total)
#     "In-Store": [0.4, 0.35, 0.2, 0.05]  # Balanced quantities (higher unit engagement)
# }

# # Generate synthetic transactions
# for _ in range(num_transactions):
#     transaction_id = str(uuid.uuid4())
#     transaction_date = random.choice(date_pool)
#     customer_id = random.choice(customer_ids)
#     channel_id = random.choice(channels)

#     # Set branch_id and staff_id based on channel
#     if channel_id == "Online":
#         branch_id = online_branch_id
#         staff_id = online_staff_id
#     else:
#         branch_id = random.choice(branch_ids)
#         staff_id = staff_ids[branch_id]

#     # Logical payment_mode based on channel
#     if channel_id == "In-Store":
#         allowed_payments = ["Cash", "Card"]
#     elif channel_id == "Online":
#         allowed_payments = ["Card", "BNPL"]
#     else:  # B2B
#         allowed_payments = payment_modes

#     payment_mode = random.choice(allowed_payments)
#     voucher_redeemed_flag = random.choice(["Yes", "No"])
#     is_upsell = random.choice(["Yes", "No"])
#     is_returned = "No"  # Default, will update after DataFrame creation

#     num_lines = random.randint(1, max_lines_per_transaction)
#     used_products = set()

#     for line_num in range(1, num_lines + 1):
#         # Ensure unique product per transaction
#         product_id = random.choice([pid for pid in product_ids if pid not in used_products])
#         used_products.add(product_id)

#         # Retrieve the fixed unit_price for this product
#         unit_price = product_unit_price_map[product_id]

#         # Use channel-specific quantity distribution
#         quantity_options = [1, 2, 3, 5]
#         quantity_probs = channel_quantity_weights[channel_id]
#         quantity = np.random.choice(quantity_options, p=quantity_probs)

#         # Determine if a discount is applied to this line item (e.g., a promotional discount)
#         # This discount is applied *after* the unit_price * quantity.
#         discount_rate_on_line = np.random.choice([0, 0.02, 0.05, 0.1], p=[0.7, 0.15, 0.1, 0.05])
#         discount_applied = round(discount_rate_on_line * unit_price * quantity, 2)

#         data.append({
#             "transaction_id": transaction_id,
#             "transaction_line_id": f"{transaction_id[:8]}-{line_num}",
#             "customer_id": customer_id,
#             "product_id": product_id,
#             "product_name": product_name_map[product_id],
#             "branch_id": branch_id,
#             "staff_id": staff_id,
#             "channel_id": channel_id,
#             "transaction_date": transaction_date.strftime("%Y-%m-%d %H:%M:%S"),
#             "quantity": quantity,
#             "unit_price": unit_price, # This is now fixed per product_id
#             "discount_applied": discount_applied, # This is a transaction-level discount on the line total
#             "payment_mode": payment_mode,
#             "voucher_redeemed_flag": voucher_redeemed_flag,
#             "is_upsell": is_upsell,
#             "is_returned": is_returned
#         })

# # Create DataFrame
# df = pd.DataFrame(data)

# # Add grand_total
# df["grand_total"] = (df["unit_price"] * df["quantity"]) - df["discount_applied"]
# df["net_price"] = df["grand_total"] / df["quantity"] # Net price per unit *after* discount for that quantity

# # Add monthly targets column for more realistic target setting
# df["transaction_date"] = pd.to_datetime(df["transaction_date"])
# df["month"] = df["transaction_date"].dt.month

# # Ensure no more than 15% returned per month
# for month in df["month"].unique():
#     month_idx = df[df["month"] == month].index
#     # Randomly pick a rate between 5% and 14% (inclusive)
#     rate = random.uniform(0.05, 0.14)
#     n_return = int(rate * len(month_idx))
#     if n_return > 0:
#         returned_idx = np.random.choice(month_idx, n_return, replace=False)
#         df.loc[returned_idx, "is_returned"] = "Yes"

# # Create realistic monthly targets based on historical performance + growth expectations
# monthly_target_multipliers = {
#     1: 1.15,    # 15% growth target for Jan
#     2: 1.25,    # 25% growth target for Feb (ambitious due to low base)
#     3: 1.18,    # 18% growth target for Mar
#     4: 1.12,    # 12% growth target for Apr
#     5: 1.20,    # 20% growth target for May
#     6: 1.10,    # 10% growth target for Jun
#     7: 1.08,    # 8% growth target for Jul
#     8: 1.22,    # 22% growth target for Aug
#     9: 1.05,    # 5% growth target for Sep
#     10: 1.07,   # 7% growth target for Oct
#     11: 1.09,   # 9% growth target for Nov
#     12: 1.03    # 3% growth target for Dec (conservative due to high base)
# }

# df["target_multiplier"] = df["month"].map(monthly_target_multipliers)

# # Add geolocation data for Kenya (10 stores)
# random.seed(42) # Keeping seed for consistent branch locations

# # Unique branch IDs
# branch_ids_unique = df["branch_id"].unique()

# # Kenya coordinates (realistic locations within Kenya)
# kenya_locations = [
#     {"lat": -1.2921, "lon": 36.8219},   # Nairobi (CBD)
#     {"lat": -1.3731, "lon": 36.8569},   # Nairobi (Karen)
#     {"lat": -4.0435, "lon": 39.6682},   # Mombasa
#     {"lat": -0.3031, "lon": 35.2961},   # Eldoret
#     {"lat": -0.0917, "lon": 34.7680},   # Kisumu
#     {"lat": -0.4906, "lon": 35.2719},   # Nakuru
#     {"lat": 0.5143, "lon": 35.2698},    # Kitale
#     {"lat": -1.5177, "lon": 37.2634},   # Machakos
#     {"lat": -0.6743, "lon": 34.5615},   # Kericho
#     {"lat": -3.2175, "lon": 40.1167}    # Malindi
# ]

# # Generate geolocation mapping for branches
# branch_geo = {}
# for i, branch_id in enumerate(branch_ids_unique):
#     if i < len(kenya_locations):
#         # Add some random variation to make locations more realistic
#         lat_variation = random.uniform(-0.01, 0.01)
#         lon_variation = random.uniform(-0.01, 0.01)
#         branch_geo[branch_id] = {
#             "latitude": round(kenya_locations[i]["lat"] + lat_variation, 6),
#             "longitude": round(kenya_locations[i]["lon"] + lon_variation, 6)
#         }
#     else:
#         # If more branches than predefined locations, generate random Kenya coordinates
#         branch_geo[branch_id] = {
#             "latitude": round(random.uniform(-4.5, 1.0), 6),   # Kenya latitude range
#             "longitude": round(random.uniform(33.9, 41.9), 6)  # Kenya longitude range
#         }

# # Add lat/lon to DataFrame using mapping
# df["latitude"] = df["branch_id"].map(lambda x: branch_geo[x]["latitude"])
# df["longitude"] = df["branch_id"].map(lambda x: branch_geo[x]["longitude"])

# # Save the updated dataset to final CSV
# output_file = "synthetic_transaction_data.csv"
# df.to_csv(output_file, index=False)

# print(f"Final dataset with geolocation and realistic Kenya prices saved to: {output_file}")

# # Preview the monthly sales vs targets
# df["month_year"] = df["transaction_date"].dt.to_period("M")
# monthly_analysis = df.groupby("month_year").agg({
#     "grand_total": "sum",
#     "target_multiplier": "first"
# }).reset_index()

# monthly_analysis["target"] = monthly_analysis["grand_total"] * monthly_analysis["target_multiplier"]
# monthly_analysis["achievement_pct"] = (monthly_analysis["grand_total"] / monthly_analysis["target"]) * 100

# print("\nMonthly Sales vs Target Analysis:")
# print(monthly_analysis[["month_year", "grand_total", "target", "achievement_pct"]])

# # Verify channel pricing differences
# print("\nChannel Pricing Analysis (Average Grand Total by Channel):")
# channel_analysis = df.groupby("channel_id")["grand_total"].mean().sort_values()
# print(channel_analysis)

# print("\nSample data preview:")
# print(df[["product_id", "product_name", "channel_id", "unit_price", "discount_applied", "quantity", "grand_total", "net_price", "latitude", "longitude"]].head(10))

# # Verify same product has same unit_price across transactions
# print("\nUnit Price Consistency Check:")
# product_price_check = df.groupby('product_id').agg(
#     unit_price_min=('unit_price', 'min'),
#     unit_price_max=('unit_price', 'max'),
#     unit_price_nunique=('unit_price', 'nunique'),
#     avg_grand_total=('grand_total', 'mean')
# ).round(2)
# print("Products with consistent unit_price:")
# print(product_price_check[product_price_check['unit_price_nunique'] != 1]) # Should ideally be empty or show 1 for all if perfectly fixed
# print(product_price_check.head(10))

# print("\nChannel grand_total analysis (should show Online < B2B ≈ In-Store):")
# channel_totals = df.groupby('channel_id').agg({
#     'grand_total': ['mean', 'sum', 'count'],
#     'quantity': 'mean'
# }).round(2)
# print(channel_totals)

