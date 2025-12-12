#!/usr/bin/env python3
"""
Recommendation API for real-time inference
Loads pre-trained models for e-commerce recommendations
"""

import sys
import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import mysql.connector
from datetime import datetime
import pickle

# Redirect all print statements to stderr by default for this module
_original_print = print
def print(*args, **kwargs):
    if 'file' not in kwargs:
        kwargs['file'] = sys.stderr
    _original_print(*args, **kwargs)

# Suppress TensorFlow logging and progress bars
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')
tf.keras.utils.disable_interactive_logging()
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Context manager to suppress stdout
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# Database configuration
DB_CONFIG = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'ecom'
}

# Import model classes
from model_classes import BMF, NeuMF, LNCM, ENCM


class TrainedRecommendationSystem:
    def __init__(self):
        self.db_connection = None
        self.models = {}
        self.encoders = {}
        self.data_stats = {}
        self.context_encoders = {}
        self.initialize_database()
        self.load_encoders_and_stats()
        self.load_trained_models()

    def initialize_database(self):
        try:
            self.db_connection = mysql.connector.connect(**DB_CONFIG)
        except Exception as e:
            sys.exit(1)

    def load_encoders_and_stats(self):
        """Load pre-trained encoders and data statistics"""
        try:
            # Load encoders
            with open('EcomModelTrain/training_data/user_encoder.pkl', 'rb') as f:
                self.encoders['user'] = pickle.load(f)
            with open('EcomModelTrain/training_data/item_encoder.pkl', 'rb') as f:
                self.encoders['item'] = pickle.load(f)
            with open('EcomModelTrain/training_data/context_encoders.pkl', 'rb') as f:
                self.context_encoders = pickle.load(f)
            with open('EcomModelTrain/training_data/data_stats.pkl', 'rb') as f:
                self.data_stats = pickle.load(f)

            print(f"Loaded encoders for {self.data_stats['n_users']} users, {self.data_stats['n_items']} items")
        except Exception as e:
            print(f"Error loading encoders: {e}")
            sys.exit(1)

    def load_trained_models(self):
        """Load pre-trained model weights"""
        try:
            # Initialize models with fixed dimensions (matching training)
            self.models['BMF'] = BMF(n_users=self.data_stats['n_users'], n_items=self.data_stats['n_items'], embedding_dim=50)
            self.models['NeuMF'] = NeuMF(n_users=self.data_stats['n_users'], n_items=self.data_stats['n_items'], embedding_dim=50, hidden_dims=[64, 32, 16])
            self.models['LNCM'] = LNCM(n_users=self.data_stats['n_users'], n_items=self.data_stats['n_items'], embedding_dim=50, hidden_dims=[64, 32])

            # ENCM with training model class
            from training_model_classes import ENCM as ENCMTraining
            n_contexts = [feat[1] for feat in self.data_stats['context_features']]
            # Match context_dims with actual number of classes
            context_dims = [
                len(self.context_encoders['category'].classes_),  # 8
                len(self.context_encoders['brand'].classes_),     # 6
                len(self.context_encoders['device'].classes_),    # 3
                4,  # time_of_day
                4,  # season
                4,  # gender
                24, # hour
                12, # month
                7,  # day_of_week
                2   # is_weekend
            ]
            self.models['ENCM'] = ENCMTraining(
                n_users=self.data_stats['n_users'],
                n_items=self.data_stats['n_items'],
                n_contexts=n_contexts,
                embedding_dim=50,
                context_dims=context_dims,
                hidden_dims=[64, 32]
            )

            # Build models
            self.models['BMF'].build([(None,), (None,)])
            self.models['NeuMF'].build([(None,), (None,)])
            self.models['LNCM'].build([(None,), (None,)])
            self.models['ENCM'].build([(None,), (None,), (None, 10)])

            # Build models (required before loading weights)
            self.models['BMF'].build([(None,), (None,)])
            self.models['NeuMF'].build([(None,), (None,)])
            self.models['LNCM'].build([(None,), (None,)])
            self.models['ENCM'].build([(None,), (None,), (None, 10)])  # 10 context features

            # Load weights
            self.models['BMF'].load_weights('models/bmf_model.h5')
            self.models['NeuMF'].load_weights('models/neumf_model.h5')
            self.models['LNCM'].load_weights('models/lncm_model.h5')
            self.models['ENCM'].load_weights('models/encm_model.h5')

            print("All pre-trained models loaded successfully!")

        except Exception as e:
            print(f"Error loading models: {e}")
            sys.exit(1)

    def get_user_context(self, user_id, provided_context=None):
        """Get current context for user"""
        try:
            context = provided_context or {}

            # Get user gender
            if 'gender' not in context:
                user_query = f"SELECT genderId FROM users WHERE id = {user_id}"
                user_df = pd.read_sql(user_query, self.db_connection)
                gender = user_df.iloc[0]['genderId'] if not user_df.empty else None
                gender_map = {'M': 0, 'FE': 1, 'O': 2}
                context['gender'] = gender_map.get(gender, 3)

            # Get device type
            if 'device_type' not in context:
                device_query = f"SELECT device_type FROM interactions WHERE userId = {user_id} ORDER BY timestamp DESC LIMIT 1"
                device_df = pd.read_sql(device_query, self.db_connection)
                context['device_type'] = device_df.iloc[0]['device_type'] if not device_df.empty else 'unknown'

            # Get preferred categories and brands
            if 'preferred_categories' not in context or 'preferred_brands' not in context:
                pref_query = f"""
                    SELECT p.categoryId, p.brandId
                    FROM interactions i
                    JOIN products p ON i.productId = p.id
                    WHERE i.userId = {user_id} AND i.actionCode IN ('cart', 'purchase')
                    ORDER BY i.timestamp DESC
                    LIMIT 50
                """
                pref_df = pd.read_sql(pref_query, self.db_connection)
                context['preferred_categories'] = pref_df['categoryId'].dropna().unique().tolist()
                context['preferred_brands'] = pref_df['brandId'].dropna().unique().tolist()

            # Current time context
            now = datetime.now()
            hour = now.hour
            month = now.month
            day_of_week = now.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0

            time_of_day = 0 if hour < 6 else (1 if hour < 12 else (2 if hour < 18 else 3))
            season = 0 if month <= 2 or month == 12 else (1 if month <= 5 else (2 if month <= 8 else 3))

            context['time_of_day'] = context.get('time_of_day', time_of_day)
            context['season'] = context.get('season', season)
            context['hour'] = context.get('hour', hour)
            context['month'] = context.get('month', month - 1)  # 0-based
            context['day_of_week'] = context.get('day_of_week', day_of_week)
            context['is_weekend'] = context.get('is_weekend', is_weekend)

            return context

        except Exception as e:
            return {
                'device_type': 'unknown',
                'time_of_day': 1,
                'season': 2,
                'gender': 3,
                'preferred_categories': [],
                'preferred_brands': [],
                'hour': 12,
                'month': 5,
                'day_of_week': 0,
                'is_weekend': 0
            }

    def get_recommendations(self, user_id, model_name, limit=10, provided_context=None):
        """Get recommendations for a user using specified model"""
        try:
            if model_name not in self.models:
                return {'ok': False, 'error': f'Model {model_name} not found'}

            model = self.models[model_name]

            # Get all available products
            products_df = pd.read_sql("SELECT id, name, categoryId, brandId FROM products WHERE statusId = 'S1'", self.db_connection)
            product_ids = products_df['id'].values

            # Convert to model indices
            try:
                user_id_int = int(user_id)
                if user_id_int in self.encoders['user'].classes_:
                    user_idx = self.encoders['user'].transform([user_id_int])[0]
                else:
                    user_idx = 0  # fallback

                valid_product_ids = [pid for pid in product_ids if pid in self.encoders['item'].classes_]
                if not valid_product_ids:
                    valid_product_ids = [self.encoders['item'].classes_[0]]
                item_indices = self.encoders['item'].transform(valid_product_ids)
            except Exception as e:
                return {'ok': False, 'error': f'Encoding error: {e}'}

            # Get user context
            context = self.get_user_context(user_id, provided_context)

            # Prepare input data
            n_items = len(valid_product_ids)
            user_indices = np.full(n_items, user_idx)

            if model_name == 'ENCM':
                # Get context features for ENCM
                context_features = []
                for pid in valid_product_ids:
                    product_row = products_df[products_df['id'] == pid]
                    if not product_row.empty:
                        prod = product_row.iloc[0]

                        # Encode features
                        category_id = self.context_encoders['category'].transform([prod['categoryId'] or 'unknown'])[0]
                        brand_id = self.context_encoders['brand'].transform([prod['brandId'] or 'unknown'])[0]
                        device_type_id = self.context_encoders['device'].transform([context.get('device_type', 'unknown')])[0]

                        # Time features - convert strings to integers
                        time_of_day_str = context.get('time_of_day', 'morning')
                        season_str = context.get('season', 'summer')
                        gender_str = context.get('gender', 'M')

                        # Convert string to int
                        time_of_day_map = {'night': 0, 'morning': 1, 'afternoon': 2, 'evening': 3}
                        season_map = {'winter': 0, 'spring': 1, 'summer': 2, 'autumn': 3}
                        gender_map = {'M': 0, 'FE': 1, 'O': 2}  # Add gender mapping

                        time_of_day = time_of_day_map.get(time_of_day_str, 1)  # default to morning
                        season = season_map.get(season_str, 2)  # default to summer
                        gender_id = gender_map.get(gender_str, 3)  # default to unknown (3)
                        # gender_id đã được convert ở trên
                        hour_id = context.get('hour', 12)
                        month_id = context.get('month', 5)
                        day_of_week_id = context.get('day_of_week', 0)
                        is_weekend_id = context.get('is_weekend', 0)

                        context_feature = [
                            category_id, brand_id, device_type_id, time_of_day, season, gender_id,
                            hour_id, month_id, day_of_week_id, is_weekend_id
                        ]
                        context_features.append(context_feature)
                    else:
                        context_features.append([0] * 10)

                context_features = np.array(context_features, dtype=np.int32)
                with SuppressOutput():
                    predictions = model.predict([user_indices, item_indices, context_features], batch_size=32, verbose=0)
            else:
                # For other models
                with SuppressOutput():
                    predictions = model.predict([user_indices, item_indices], batch_size=32, verbose=0)

            # Get top recommendations
            predictions_flat = predictions.flatten()
            top_indices = np.argsort(predictions_flat)[::-1][:limit]

            recommendations = []
            for idx in top_indices:
                product_id = valid_product_ids[idx]
                score = float(predictions_flat[idx])
                product_row = products_df[products_df['id'] == product_id].iloc[0]
                product_name = product_row['name']
                brand_name = product_row['brandId'] or 'Unknown Brand'

                recommendations.append({
                    'productId': int(product_id),
                    'productName': product_name,
                    'brandName': brand_name,
                    'score': score
                })

            return {
                'ok': True,
                'items': recommendations,
                'context': context,
                'model': model_name
            }
        except Exception as e:
            return {'ok': False, 'error': str(e)}


def main():
    """Main API handler"""
    try:
        # Read input from stdin
        input_data = sys.stdin.read()
        if not input_data:
            _original_print(json.dumps({'ok': False, 'error': 'No input data'}))
            return

        payload = json.loads(input_data)

        user_id = payload.get('user_id')
        limit = payload.get('limit', 10)
        model_name = payload.get('model', 'BMF')
        context = payload.get('context', {})

        if not user_id:
            _original_print(json.dumps({'ok': False, 'error': 'user_id is required'}))
            return

        # Initialize recommendation system (singleton pattern would be better in production)
        reco_system = TrainedRecommendationSystem()

        # Get recommendations
        result = reco_system.get_recommendations(user_id, model_name, limit, context)

        _original_print(json.dumps(result))

    except Exception as e:
        _original_print(json.dumps({'ok': False, 'error': str(e)}))


if __name__ == '__main__':
    main()
