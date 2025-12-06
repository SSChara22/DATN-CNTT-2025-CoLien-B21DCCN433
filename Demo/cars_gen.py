import pandas as pd
import numpy as np
from datetime import datetime
import random

# Parameters
num_users = 200
num_products = 500
num_rows = 10000

# Lists
categories = ["T-Shirt", "Pants", "Shoes", "Dress", "Jacket", "Accessories"]
brands = ["Nike", "Adidas", "Puma", "Uniqlo", "Zara", "H&M", "LV", "Gucci"]
device_types = ["Mobile", "Desktop", "Tablet"]
interaction_types = ["view", "add_to_cart", "purchase"]
genders = ["male", "female", "other"]
seasons = ["Spring", "Summer", "Autumn", "Winter"]

# Mapping purchase intent
purchase_intent_map = {
    "purchase": 1.0,
    "add_to_cart": 0.5,
    "view": 0.0
}

def random_date():
    start = datetime(2023, 1, 1)
    end = datetime(2025, 1, 1)
    return start + (end - start) * random.random()

rows = []

for _ in range(num_rows):
    user = random.randint(1, num_users)
    product = random.randint(1, num_products)
    dt = random_date()

    price = random.randint(80, 2000)
    discount = random.choice([0, 5, 10, 15, 20, 25, 30, 50])
    original_price = price + int(price * discount / 100)

    interaction = random.choice(interaction_types)
    rating = random.randint(1, 5) if interaction == "purchase" else np.nan

    row = {
        "user_id": user,
        "product_id": product,
        "rating": rating,
        "rating_count": random.randint(0, 500),
        "time_of_day": dt.strftime("%H:%M:%S"),
        "day_of_week": dt.weekday(),
        "month": dt.month,
        "season": random.choice(seasons),
        "device_type": random.choice(device_types),
        "gender": random.choice(genders),
        "category": random.choice(categories),
        "brand": random.choice(brands),
        "price": price,
        "original_price": original_price,
        "discount_percentage": discount,
        "product_views": random.randint(1, 50),
        "purchase_intent": purchase_intent_map[interaction],
        "interaction_type": interaction,
        "timestamp": dt.timestamp(),
        "day_name": dt.strftime("%A"),
        "is_weekend": 1 if dt.weekday() >= 5 else 0,
        "hour": dt.hour,
        "date": dt.strftime("%Y-%m-%d"),
    }

    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("cars_dataset.csv", index=False)
print("Generated cars_dataset.csv with", len(df), "rows")
