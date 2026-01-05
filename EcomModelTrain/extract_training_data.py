"""
Extract training data from ecom database for recommendation models
Creates implicit feedback dataset from user interactions
"""

import mysql.connector
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import pickle
import os

def extract_training_data():
    """Extract and prepare training data from database"""

    print("Connecting to database...")
    try:
        conn = mysql.connector.connect(
            host='127.0.0.1',
            user='root',
            password='',
            database='ecom'
        )
        print("Database connected")
    except Exception as e:
        print(f"Database connection failed: {e}")
        return

    try:
        # 1. Extract interactions data
        print("Extracting interactions...")
        interactions_query = """
            SELECT
                i.userId,
                i.productId,
                i.actionCode,
                i.device_type,
                i.timestamp,
                u.genderId as user_gender,
                p.categoryId as product_category,
                p.brandId as product_brand,
                p.view as product_views
            FROM interactions i
            JOIN users u ON i.userId = u.id
            JOIN products p ON i.productId = p.id
            WHERE u.statusId = 'S1' AND p.statusId = 'S1'
            ORDER BY i.timestamp
        """

        interactions_df = pd.read_sql(interactions_query, conn)
        print(f"Extracted {len(interactions_df)} interactions")

        # 2. Create implicit feedback ratings
        print("Creating implicit feedback ratings...")
        action_weights = {
            'purchase': 1.0,  # Highest interest
            'cart': 0.7,      # Medium interest
            'view': 0.3       # Low interest
        }

        interactions_df['rating'] = interactions_df['actionCode'].map(action_weights)

        # 3. Extract temporal features
        print("Extracting temporal features...")
        interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
        interactions_df['hour'] = interactions_df['timestamp'].dt.hour
        interactions_df['day_of_week'] = interactions_df['timestamp'].dt.dayofweek  # 0=Monday
        interactions_df['month'] = interactions_df['timestamp'].dt.month - 1  # 0-11 for embedding
        interactions_df['is_weekend'] = (interactions_df['day_of_week'] >= 5).astype(int)

        # Time of day categories
        interactions_df['time_of_day'] = pd.cut(
            interactions_df['hour'],
            bins=[-1, 5, 11, 17, 23],
            labels=[0, 1, 2, 3]  # night, morning, afternoon, evening
        ).astype(int)

        # Season categories
        interactions_df['season'] = pd.cut(
            interactions_df['month'],
            bins=[0, 2, 5, 8, 12],
            labels=[0, 1, 2, 3]  # winter, spring, summer, autumn
        ).astype(int)

        # 4. Create encoders
        print("Creating encoders...")
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()
        category_encoder = LabelEncoder()
        brand_encoder = LabelEncoder()
        device_encoder = LabelEncoder()

        # Fit encoders
        user_encoder.fit(interactions_df['userId'].unique())
        item_encoder.fit(interactions_df['productId'].unique())
        category_encoder.fit(interactions_df['product_category'].fillna('unknown'))
        brand_encoder.fit(interactions_df['product_brand'].fillna('unknown'))
        device_encoder.fit(interactions_df['device_type'].fillna('unknown'))

        # Encode data
        interactions_df['user_encoded'] = user_encoder.transform(interactions_df['userId'])
        interactions_df['item_encoded'] = item_encoder.transform(interactions_df['productId'])
        interactions_df['category_encoded'] = category_encoder.transform(interactions_df['product_category'].fillna('unknown'))
        interactions_df['brand_encoded'] = brand_encoder.transform(interactions_df['product_brand'].fillna('unknown'))
        interactions_df['device_encoded'] = device_encoder.transform(interactions_df['device_type'].fillna('unknown'))

        # Gender encoding
        gender_map = {'M': 0, 'FE': 1, 'O': 2}
        interactions_df['gender_encoded'] = interactions_df['user_gender'].map(gender_map).fillna(3).astype(int)

        # 5. Create context features list for ENCM
        print("Creating context features...")
        context_features = [
            ('category', len(category_encoder.classes_)),
            ('brand', len(brand_encoder.classes_)),
            ('device_type', len(device_encoder.classes_)),
            ('time_of_day', 4),  # 0-3
            ('season', 4),       # 0-3
            ('gender', 4),       # 0-3 (M, FE, O, unknown)
            ('hour', 24),        # 0-23
            ('month', 12),       # 1-12, but we'll use 0-11
            ('day_of_week', 7),  # 0-6
            ('is_weekend', 2),   # 0-1
        ]

        # 6. Create train/test split (80/20, temporal split)
        print("Creating train/test split...")
        interactions_df = interactions_df.sort_values('timestamp')
        split_idx = int(len(interactions_df) * 0.8)
        train_df = interactions_df[:split_idx].copy()
        test_df = interactions_df[split_idx:].copy()

        print(f"Train set: {len(train_df)} interactions")
        print(f"Test set: {len(test_df)} interactions")

        # 7. Create directory and save data
        os.makedirs('training_data', exist_ok=True)

        # Save main datasets
        train_df.to_csv('training_data/train_data.csv', index=False)
        test_df.to_csv('training_data/test_data.csv', index=False)

        # Save encoders
        with open('training_data/user_encoder.pkl', 'wb') as f:
            pickle.dump(user_encoder, f)
        with open('training_data/item_encoder.pkl', 'wb') as f:
            pickle.dump(item_encoder, f)

        # Save context encoders
        context_encoders = {
            'category': category_encoder,
            'brand': brand_encoder,
            'device': device_encoder
        }
        with open('training_data/context_encoders.pkl', 'wb') as f:
            pickle.dump(context_encoders, f)

        # Save context features info
        with open('training_data/context_features.pkl', 'wb') as f:
            pickle.dump(context_features, f)

        # Save data statistics
        stats = {
            'n_users': len(user_encoder.classes_),
            'n_items': len(item_encoder.classes_),
            'n_interactions': len(interactions_df),
            'n_train': len(train_df),
            'n_test': len(test_df),
            'context_features': context_features,
            'action_distribution': interactions_df['actionCode'].value_counts().to_dict(),
            'rating_distribution': interactions_df['rating'].value_counts().to_dict()
        }

        with open('training_data/data_stats.pkl', 'wb') as f:
            pickle.dump(stats, f)

        print("Training data extraction completed!")
        print(f"   - Users: {stats['n_users']}")
        print(f"   - Items: {stats['n_items']}")
        print(f"   - Interactions: {stats['n_interactions']}")
        print(f"   - Action distribution: {stats['action_distribution']}")

        return stats

    except Exception as e:
        print(f"Error during data extraction: {e}")
        return None
    finally:
        conn.close()

if __name__ == '__main__':
    extract_training_data()
