#!/usr/bin/env python3
"""
Recommendation API for real-time inference
Creates and initializes models from scratch for e-commerce recommendations
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
from dotenv import load_dotenv

# Suppress TensorFlow logging and progress bars
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')  # Disable GPU to avoid warnings
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

# Database configuration from environment variables
# Note: When called from Node.js, env vars are passed through system environment
DB_CONFIG = {
    'host': os.getenv('DB_HOST', os.getenv('DBHOST', '127.0.0.1')),
    'user': os.getenv('DB_USERNAME', os.getenv('DBUSER', 'root')),
    'password': os.getenv('DB_PASSWORD', os.getenv('DBPASS', '')),
    'database': os.getenv('DB_DATABASE_NAME', os.getenv('DBNAME', 'ecom'))
}


class ENCM(tf.keras.Model):
    def __init__(self, n_users, n_items, n_contexts, embedding_dim=50, context_dims=None, hidden_dims=[64, 32]):
        super(ENCM, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # Use provided context_dims or default to uniform dimension
        if context_dims is None:
            context_dims = [10] * len(n_contexts)
        elif len(context_dims) != len(n_contexts):
            raise ValueError("context_dims must have same length as n_contexts")

        # User and item embeddings
        self.user_embedding = tf.keras.layers.Embedding(n_users, embedding_dim)
        self.item_embedding = tf.keras.layers.Embedding(n_items, embedding_dim)

        # Context embeddings with variable dimensions
        self.context_embeddings = []
        total_context_dim = 0
        for i, (n_context, context_dim) in enumerate(zip(n_contexts, context_dims)):
            self.context_embeddings.append(
                tf.keras.layers.Embedding(n_context, context_dim, name=f'context_{i}')
            )
            total_context_dim += context_dim

        # Neural layers for processing interactions
        self.hidden_layers = []
        input_dim = embedding_dim * 2 + total_context_dim

        for hidden_dim in hidden_dims:
            self.hidden_layers.append(tf.keras.layers.Dense(hidden_dim, activation='relu'))
            self.hidden_layers.append(tf.keras.layers.Dropout(0.2))
            input_dim = hidden_dim

        # Output layer
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=None):
        user_ids, item_ids, context_features = inputs

        # Get user and item embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        # Get context embeddings
        context_embs = []
        for i, context_embedding in enumerate(self.context_embeddings):
            context_embs.append(context_embedding(context_features[:, i]))

        # Concatenate all features
        if context_embs:
            all_features = tf.concat([user_emb, item_emb] + context_embs, axis=-1)
        else:
            all_features = tf.concat([user_emb, item_emb], axis=-1)

        # Pass through neural network
        output = all_features
        for layer in self.hidden_layers:
            output = layer(output, training=training)

        # Final prediction
        prediction = self.output_layer(output)

        return prediction


class LNCM(tf.keras.Model):
    def __init__(self, n_users, n_items, embedding_dim=50, hidden_dims=[64, 32]):
        super(LNCM, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # User and item embeddings
        self.user_embedding = tf.keras.layers.Embedding(n_users, embedding_dim)
        self.item_embedding = tf.keras.layers.Embedding(n_items, embedding_dim)

        # Linear combination layer
        self.linear_layer = tf.keras.layers.Dense(1, use_bias=True)

        # Neural layers
        self.hidden_layers = []
        input_dim = embedding_dim * 2

        for hidden_dim in hidden_dims:
            self.hidden_layers.append(tf.keras.layers.Dense(hidden_dim, activation='relu'))
            self.hidden_layers.append(tf.keras.layers.Dropout(0.2))
            input_dim = hidden_dim

        # Neural output layer
        self.neural_layer = tf.keras.layers.Dense(1, activation='sigmoid')

        # Combination weight
        self.alpha = self.add_weight(
            shape=(1,), initializer='uniform', trainable=True, name='alpha'
        )

    def call(self, inputs, training=None):
        user_ids, item_ids = inputs

        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        # Linear part (Matrix Factorization)
        linear_output = self.linear_layer(tf.concat([user_emb, item_emb], axis=-1))

        # Neural part
        neural_input = tf.concat([user_emb, item_emb], axis=-1)
        neural_output = neural_input
        for layer in self.hidden_layers:
            neural_output = layer(neural_output, training=training)
        neural_output = self.neural_layer(neural_output)

        # Combine linear and neural parts
        output = tf.sigmoid(self.alpha) * linear_output + (1 - tf.sigmoid(self.alpha)) * neural_output

        return output


class NeuMF(tf.keras.Model):
    def __init__(self, n_users, n_items, embedding_dim=50, hidden_dims=[64, 32, 16]):
        super(NeuMF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # GMF (Generalized Matrix Factorization) embeddings
        self.user_embedding_gmf = tf.keras.layers.Embedding(n_users, embedding_dim)
        self.item_embedding_gmf = tf.keras.layers.Embedding(n_items, embedding_dim)

        # MLP (Multi-Layer Perceptron) embeddings
        self.user_embedding_mlp = tf.keras.layers.Embedding(n_users, embedding_dim)
        self.item_embedding_mlp = tf.keras.layers.Embedding(n_items, embedding_dim)

        # MLP layers
        self.mlp_layers = []
        input_dim = embedding_dim * 2

        for hidden_dim in hidden_dims:
            self.mlp_layers.append(tf.keras.layers.Dense(hidden_dim, activation='relu'))
            self.mlp_layers.append(tf.keras.layers.Dropout(0.2))
            input_dim = hidden_dim

        # Final prediction layer (combines GMF and MLP)
        self.final_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=None):
        user_ids, item_ids = inputs

        # GMF part
        user_emb_gmf = self.user_embedding_gmf(user_ids)
        item_emb_gmf = self.item_embedding_gmf(item_ids)
        gmf_output = tf.multiply(user_emb_gmf, item_emb_gmf)

        # MLP part
        user_emb_mlp = self.user_embedding_mlp(user_ids)
        item_emb_mlp = self.item_embedding_mlp(item_ids)
        mlp_input = tf.concat([user_emb_mlp, item_emb_mlp], axis=-1)

        mlp_output = mlp_input
        for layer in self.mlp_layers:
            mlp_output = layer(mlp_output, training=training)

        # Combine GMF and MLP
        combined = tf.concat([gmf_output, mlp_output], axis=-1)

        # Final prediction
        output = self.final_layer(combined)

        return output


class BMF(tf.keras.Model):
    def __init__(self, n_users, n_items, embedding_dim=50):
        super(BMF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # User and item embeddings
        self.user_embedding = tf.keras.layers.Embedding(n_users, embedding_dim)
        self.item_embedding = tf.keras.layers.Embedding(n_items, embedding_dim)

        # User and item biases
        self.user_bias = tf.keras.layers.Embedding(n_users, 1)
        self.item_bias = tf.keras.layers.Embedding(n_items, 1)

        # Global bias
        self.global_bias = self.add_weight(
            shape=(1,), initializer='zeros', trainable=True, name='global_bias'
        )

    def call(self, inputs, training=None):
        user_ids, item_ids = inputs

        # Get embeddings and biases
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        user_b = self.user_bias(user_ids)
        item_b = self.item_bias(item_ids)

        # Compute dot product
        dot_product = tf.reduce_sum(user_emb * item_emb, axis=-1, keepdims=True)

        # Final prediction with biases
        output = tf.sigmoid(dot_product + user_b + item_b + self.global_bias)

        return output


class RecommendationSystem:
    def __init__(self):
        self.db_connection = None
        self.models = {}
        self.encoders = {}
        self.data_stats = {}
        self.initialize_database()
        self.load_data()
        self.build_models()

    def initialize_database(self):
        try:
            self.db_connection = mysql.connector.connect(**DB_CONFIG)
        except Exception as e:
            sys.exit(1)

    def load_data(self):
        """Load and preprocess data from database"""
        try:
            # Load users
            users_df = pd.read_sql("SELECT id, genderId FROM users WHERE statusId = 'S1'", self.db_connection)
            self.data_stats['n_users'] = len(users_df)
            self.encoders['user'] = LabelEncoder()
            self.encoders['user'].fit(users_df['id'].values)

            # Load products
            products_df = pd.read_sql("SELECT id, categoryId, brandId FROM products WHERE statusId = 'S1'", self.db_connection)
            self.data_stats['n_items'] = len(products_df)
            self.encoders['item'] = LabelEncoder()
            self.encoders['item'].fit(products_df['id'].values)

            # Load interactions
            interactions_df = pd.read_sql("""
                SELECT userId, productId, actionCode, timestamp, device_type
                FROM interactions
                ORDER BY timestamp DESC
            """, self.db_connection)

            # Create context encoders
            self.setup_context_encoders(products_df, interactions_df)

            self.interactions_df = interactions_df
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)

    def setup_context_encoders(self, products_df, interactions_df):
        """Setup encoders for context features"""
        # Category encoder
        categories = products_df['categoryId'].fillna('unknown').unique()
        self.encoders['category'] = LabelEncoder()
        self.encoders['category'].fit(categories)
        self.data_stats['n_categories'] = len(categories)

        # Brand encoder
        brands = products_df['brandId'].fillna('unknown').unique()
        self.encoders['brand'] = LabelEncoder()
        self.encoders['brand'].fit(brands)
        self.data_stats['n_brands'] = len(brands)

        # Device type encoder
        devices = interactions_df['device_type'].fillna('unknown').unique()
        self.encoders['device'] = LabelEncoder()
        self.encoders['device'].fit(devices)
        self.data_stats['n_devices'] = len(devices)

        # Time of day encoder (4 categories: night, morning, afternoon, evening)
        self.data_stats['n_time_of_day'] = 4

        # Season encoder (4 categories: winter, spring, summer, autumn)
        self.data_stats['n_seasons'] = 4

        # Gender encoder (assuming M, FE, O, unknown)
        self.data_stats['n_genders'] = 4

        # Hour encoder (0-23)
        self.data_stats['n_hours'] = 24

        # Month encoder (1-12)
        self.data_stats['n_months'] = 12

        # Day of week encoder (0-6: Monday-Sunday)
        self.data_stats['n_days_of_week'] = 7

        # Is weekend encoder (0-1)
        self.data_stats['n_is_weekend'] = 2


        # Price range encoder (assuming price ranges: 0-100k, 100k-500k, 500k-1M, 1M-5M, 5M+)
        price_ranges = ['low', 'medium', 'high', 'premium', 'luxury']
        self.encoders['price_range'] = LabelEncoder()
        self.encoders['price_range'].fit(price_ranges)
        self.data_stats['n_price_ranges'] = len(price_ranges)

        # Discount percentage encoder (0-100%)
        self.data_stats['n_discount_percentages'] = 101

        # Product views encoder (log scale categories)
        view_categories = ['very_low', 'low', 'medium', 'high', 'very_high']
        self.encoders['product_views'] = LabelEncoder()
        self.encoders['product_views'].fit(view_categories)
        self.data_stats['n_product_views'] = len(view_categories)

        # Rating encoder (1-5 stars, with 0.5 increments)
        self.data_stats['n_ratings'] = 9  # 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0

        # Rating count encoder (log scale categories)
        rating_count_categories = ['very_few', 'few', 'moderate', 'many', 'very_many']
        self.encoders['rating_count'] = LabelEncoder()
        self.encoders['rating_count'].fit(rating_count_categories)
        self.data_stats['n_rating_counts'] = len(rating_count_categories)

        # Preferred categories encoder (same as category)
        self.data_stats['n_preferred_categories'] = self.data_stats['n_categories']

        # Preferred brands encoder (same as brand)
        self.data_stats['n_preferred_brands'] = self.data_stats['n_brands']

    def build_models(self):
        """Build and initialize all models"""
        n_users = self.data_stats['n_users']
        n_items = self.data_stats['n_items']

        # Context dimensions for ENCM - must match order in context_features array
        n_contexts = [
            self.data_stats['n_categories'],          # 0: category
            self.data_stats['n_brands'],              # 1: brand
            self.data_stats['n_devices'],             # 2: device_type
            self.data_stats['n_time_of_day'],         # 3: time_of_day (always 4)
            self.data_stats['n_seasons'],             # 4: season (always 4)
            self.data_stats['n_genders'],             # 5: gender (always 4)
            self.data_stats['n_hours'],               # 6: hour (24)
            self.data_stats['n_months'],              # 7: month (12)
            self.data_stats['n_days_of_week'],        # 8: day_of_week (7)
            self.data_stats['n_is_weekend'],          # 9: is_weekend (2)
            self.data_stats['n_price_ranges'],        # 10: price_range (5)
            self.data_stats['n_discount_percentages'], # 11: discount_percentage (101)
            self.data_stats['n_product_views'],       # 12: product_views (5)
            self.data_stats['n_ratings'],             # 13: rating (9)
            self.data_stats['n_rating_counts'],       # 14: rating_count (5)
            self.data_stats['n_preferred_categories'], # 15: preferred_categories
            self.data_stats['n_preferred_brands']     # 16: preferred_brands
        ]

        # Context embedding dimensions - larger for important features
        context_dims = [
            15,  # 0: category - important for product categorization
            15,  # 1: brand - important for brand preferences
            8,   # 2: device_type - moderate importance
            6,   # 3: time_of_day - basic time context
            6,   # 4: season - seasonal patterns
            6,   # 5: gender - demographic factor
            20,  # 6: hour - detailed time patterns (important)
            15,  # 7: month - monthly trends (important)
            12,  # 8: day_of_week - weekly patterns (important)
            4,   # 9: is_weekend - weekend vs weekday
            12,  # 10: price_range - price sensitivity (important)
            8,   # 11: discount_percentage - discount influence
            18,  # 12: product_views - popularity indicator (important)
            15,  # 13: rating - quality indicator (important)
            12,  # 14: rating_count - reliability of rating (important)
            25,  # 15: preferred_categories - user preferences (very important)
            25   # 16: preferred_brands - user preferences (very important)
        ]

        # Build models
        self.models['BMF'] = BMF(n_users, n_items, embedding_dim=50)
        self.models['NeuMF'] = NeuMF(n_users, n_items, embedding_dim=50, hidden_dims=[64, 32, 16])
        self.models['LNCM'] = LNCM(n_users, n_items, embedding_dim=50, hidden_dims=[64, 32])
        self.models['ENCM'] = ENCM(n_users, n_items, n_contexts, embedding_dim=50, context_dims=context_dims, hidden_dims=[64, 32])

        # Compile models
        for name, model in self.models.items():
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def _get_price_range_id(self, price):
        """Convert price to price range category"""
        if price <= 100000:
            return 0  # low
        elif price <= 500000:
            return 1  # medium
        elif price <= 1000000:
            return 2  # high
        elif price <= 5000000:
            return 3  # premium
        else:
            return 4  # luxury

    def _get_product_views_id(self, views):
        """Convert product views to category"""
        if views <= 10:
            return 0  # very_low
        elif views <= 100:
            return 1  # low
        elif views <= 1000:
            return 2  # medium
        elif views <= 10000:
            return 3  # high
        else:
            return 4  # very_high

    def _get_rating_count_id(self, count):
        """Convert rating count to category"""
        if count <= 5:
            return 0  # very_few
        elif count <= 50:
            return 1  # few
        elif count <= 500:
            return 2  # moderate
        elif count <= 5000:
            return 3  # many
        else:
            return 4  # very_many

    def _get_preferred_category_id(self, preferred_categories, product_category):
        """Get encoded preferred category ID"""
        if not preferred_categories or not product_category:
            return 0
        # If product category is in preferred categories, return its encoded ID
        if product_category in preferred_categories:
            try:
                return self.encoders['category'].transform([product_category])[0]
            except:
                return 0
        return 0

    def _get_preferred_brand_id(self, preferred_brands, product_brand):
        """Get encoded preferred brand ID"""
        if not preferred_brands or not product_brand:
            return 0
        # If product brand is in preferred brands, return its encoded ID
        if product_brand in preferred_brands:
            try:
                return self.encoders['brand'].transform([product_brand])[0]
            except:
                return 0
        return 0

    def get_user_context(self, user_id, provided_context=None):
        """Get current context for user, merging with provided context"""
        try:
            # Start with provided context if available
            context = provided_context or {}

            # Get user info if not provided
            if 'gender' not in context:
                user_query = f"SELECT genderId FROM users WHERE id = {user_id}"
                user_df = pd.read_sql(user_query, self.db_connection)
                gender = user_df.iloc[0]['genderId'] if not user_df.empty else None
                gender_map = {'M': 0, 'FE': 1, 'O': 2}
                context['gender'] = gender_map.get(gender, 3)

            # Get last interaction for device type if not provided
            if 'device_type' not in context:
                device_query = f"SELECT device_type FROM interactions WHERE userId = {user_id} ORDER BY timestamp DESC LIMIT 1"
                device_df = pd.read_sql(device_query, self.db_connection)
                context['device_type'] = device_df.iloc[0]['device_type'] if not device_df.empty else 'unknown'

            # Get preferred categories and brands if not provided
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
                context['preferred_categories'] = pref_df['categoryId'].dropna().unique().tolist() if 'preferred_categories' not in context else context['preferred_categories']
                context['preferred_brands'] = pref_df['brandId'].dropna().unique().tolist() if 'preferred_brands' not in context else context['preferred_brands']

            # Current time context if not provided
            if 'time_of_day' not in context or 'season' not in context:
                now = datetime.now()
                hour = now.hour
                month = now.month
                day_of_week = now.weekday()  # 0=Monday, 6=Sunday
                is_weekend = 1 if day_of_week >= 5 else 0

                time_of_day = 0 if hour < 6 else (1 if hour < 12 else (2 if hour < 18 else 3))
                season = 0 if month <= 2 or month == 12 else (1 if month <= 5 else (2 if month <= 8 else 3))

                context['time_of_day'] = context.get('time_of_day', time_of_day)
                context['season'] = context.get('season', season)
                context['hour'] = context.get('hour', hour)
                context['month'] = context.get('month', month - 1)  # 0-based for encoding
                context['day_of_week'] = context.get('day_of_week', day_of_week)
                context['is_weekend'] = context.get('is_weekend', is_weekend)

            return context

        except Exception as e:
            return {
                'device_type': 'unknown',
                'time_of_day': 1,  # default to morning
                'season': 2,       # default to summer
                'gender': 3,       # default to unknown
                'preferred_categories': [],
                'preferred_brands': [],
                'hour': 12,
                'month': 5,        # June (0-based)
                'day_of_week': 0,  # Monday
                'is_weekend': 0    # weekday
            }

    def _sanitize_context(self, context):
        """Sanitize and convert context values to proper types"""
        if not context:
            return {}

        sanitized = {}

        # Time of day conversion
        time_of_day_map = {'night': 0, 'morning': 1, 'afternoon': 2, 'evening': 3}
        if 'time_of_day' in context:
            tod = context['time_of_day']
            if isinstance(tod, str):
                sanitized['time_of_day'] = time_of_day_map.get(tod.lower(), 1)  # default to morning
            else:
                sanitized['time_of_day'] = int(tod)

        # Season conversion
        season_map = {'winter': 0, 'spring': 1, 'summer': 2, 'autumn': 3}
        if 'season' in context:
            season = context['season']
            if isinstance(season, str):
                sanitized['season'] = season_map.get(season.lower(), 2)  # default to summer
            else:
                sanitized['season'] = int(season)

        # Convert other integer fields
        int_fields = ['gender', 'hour', 'month', 'day_of_week', 'is_weekend']
        for field in int_fields:
            if field in context:
                try:
                    sanitized[field] = int(context[field])
                except (ValueError, TypeError):
                    pass  # skip invalid values

        # Copy string/list fields as-is
        str_fields = ['device_type']
        for field in str_fields:
            if field in context:
                sanitized[field] = str(context[field])

        list_fields = ['preferred_categories', 'preferred_brands']
        for field in list_fields:
            if field in context and isinstance(context[field], list):
                sanitized[field] = context[field]

        return sanitized

    def get_recommendations(self, user_id, model_name, limit=10, provided_context=None):
        """Get recommendations for a user using specified model"""
        try:
            if model_name not in self.models:
                return {'ok': False, 'error': f'Model {model_name} not found'}

            # Sanitize provided context
            provided_context = self._sanitize_context(provided_context)

            model = self.models[model_name]

            # Get all available products
            products_df = pd.read_sql("SELECT id, name, categoryId, brandId FROM products WHERE statusId = 'S1'", self.db_connection)
            product_ids = products_df['id'].values

            # Convert to model indices
            try:
                # Check if user_id exists in fitted data, otherwise use first available
                user_id_int = int(user_id)
                if user_id_int in self.encoders['user'].classes_:
                    user_idx = self.encoders['user'].transform([user_id_int])[0]
                else:
                    # Use first user ID as fallback
                    user_idx = 0

                # Filter item_ids to only those that exist in fitted data
                valid_product_ids = [pid for pid in product_ids if pid in self.encoders['item'].classes_]
                if not valid_product_ids:
                    valid_product_ids = [self.encoders['item'].classes_[0]]  # fallback to first item
                item_indices = self.encoders['item'].transform(valid_product_ids)
            except Exception as e:
                return {'ok': False, 'error': f'Encoding error: {e}'}

            # Get user context (merge provided context with database context)
            context = self.get_user_context(user_id, provided_context)

            # Prepare input data
            n_items = len(valid_product_ids)
            user_indices = np.full(n_items, user_idx)

            if model_name == 'ENCM':
                # Get product details from products table
                product_details_query = f"""
                    SELECT id, name, categoryId, brandId, view as product_views
                    FROM products
                    WHERE id IN ({','.join(map(str, valid_product_ids))})
                """
                products_df = pd.read_sql(product_details_query, self.db_connection)

                # Get price details from productdetails table
                price_details_query = f"""
                    SELECT productId, originalPrice, discountPrice
                    FROM productdetails
                    WHERE productId IN ({','.join(map(str, valid_product_ids))})
                """
                price_details_df = pd.read_sql(price_details_query, self.db_connection)

                # Get interaction statistics for rating and rating_count
                interaction_stats_query = f"""
                    SELECT productId,
                           COUNT(*) as total_interactions,
                           COUNT(CASE WHEN actionCode = 'view' THEN 1 END) as view_count,
                           COUNT(CASE WHEN actionCode = 'cart' THEN 1 END) as cart_count,
                           COUNT(CASE WHEN actionCode = 'purchase' THEN 1 END) as purchase_count
                    FROM interactions
                    WHERE productId IN ({','.join(map(str, valid_product_ids))})
                    GROUP BY productId
                """
                interaction_stats_df = pd.read_sql(interaction_stats_query, self.db_connection)

                # Get current date/time context
                now = datetime.now()
                current_hour = now.hour
                current_month = now.month
                current_day_of_week = now.weekday()  # 0=Monday, 6=Sunday
                current_is_weekend = 1 if current_day_of_week >= 5 else 0
                current_date = now.day
                current_day_name = now.strftime('%A')

                context_features = []
                for pid in valid_product_ids:
                    # Get product data from products table
                    product_row = products_df[products_df['id'] == pid]
                    prod = product_row.iloc[0] if not product_row.empty else None

                    # Get price data from productdetails table
                    price_row = price_details_df[price_details_df['productId'] == pid]
                    price_data = price_row.iloc[0] if not price_row.empty else None

                    # Get interaction stats
                    stats_row = interaction_stats_df[interaction_stats_df['productId'] == pid]
                    stats = stats_row.iloc[0] if not stats_row.empty else None

                    if prod is not None:
                        # Encode category and brand
                        category_id = self.encoders['category'].transform([prod['categoryId'] or 'unknown'])[0] if prod['categoryId'] else 0
                        brand_id = self.encoders['brand'].transform([prod['brandId'] or 'unknown'])[0] if prod['brandId'] else 0

                        # Device type from context
                        device_type_id = self.encoders['device'].transform([context.get('device_type', 'unknown')])[0]

                        # Time context
                        time_of_day = context.get('time_of_day', current_hour // 6)
                        season = context.get('season', (current_month - 1) // 3)

                        # Gender
                        gender_id = context.get('gender', 3)

                        # Detailed time features
                        hour_id = context.get('hour', current_hour)
                        month_id = context.get('month', current_month - 1)  # 0-based
                        day_of_week_id = context.get('day_of_week', current_day_of_week)
                        is_weekend_id = context.get('is_weekend', current_is_weekend)

                        # Product attributes from database
                        # Price from productdetails (use discountPrice if available, else originalPrice)
                        price = 0
                        discount_percentage = 0
                        if price_data is not None:
                            original_price = price_data['originalPrice'] or 0
                            discount_price = price_data['discountPrice'] or original_price
                            price = discount_price
                            if original_price > 0:
                                discount_percentage = min(100, int(((original_price - discount_price) / original_price) * 100))

                        price_range_id = self._get_price_range_id(price)
                        discount_percentage_id = discount_percentage

                        # Product views from products table
                        product_views = prod['product_views'] or 0
                        product_views_id = self._get_product_views_id(product_views)

                        # Rating and rating_count from interactions (derived metrics)
                        rating = 0.0
                        rating_count = 0

                        if stats is not None:
                            total_interactions = stats['total_interactions'] or 0
                            cart_count = stats['cart_count'] or 0
                            purchase_count = stats['purchase_count'] or 0

                            # Use purchase-to-cart ratio as proxy for rating (higher ratio = higher rating)
                            if cart_count > 0:
                                purchase_ratio = purchase_count / cart_count
                                rating = min(5.0, 2.0 + (purchase_ratio * 3.0))  # Scale 2.0-5.0
                            elif purchase_count > 0:
                                rating = 4.0  # Default good rating if purchases exist

                            rating_count = purchase_count  # Use purchase count as rating count

                        rating_id = max(0, min(8, int((rating - 1) * 2)))  # Convert to 0-8 scale
                        rating_count_id = self._get_rating_count_id(rating_count)

                        # Preferred categories and brands
                        preferred_categories = context.get('preferred_categories', [])
                        preferred_category_id = self._get_preferred_category_id(preferred_categories, prod['categoryId'])

                        preferred_brands = context.get('preferred_brands', [])
                        preferred_brand_id = self._get_preferred_brand_id(preferred_brands, prod['brandId'])

                        context_feature = [
                            category_id, brand_id, device_type_id, time_of_day, season, gender_id,
                            hour_id, month_id, day_of_week_id, is_weekend_id,
                            price_range_id, discount_percentage_id, product_views_id, rating_id, rating_count_id,
                            preferred_category_id, preferred_brand_id
                        ]
                        context_features.append(context_feature)
                    else:
                        # Default values if product not found
                        context_features.append([0] * 17)

                context_features = np.array(context_features, dtype=np.int32)
                with SuppressOutput():
                    predictions = model.predict([user_indices, item_indices, context_features], batch_size=32, verbose=0)
            else:
                # For other models, just user-item pairs
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
            print(json.dumps({'ok': False, 'error': 'No input data'}))
            return

        payload = json.loads(input_data)

        user_id = payload.get('user_id')
        limit = payload.get('limit', 10)
        model_name = payload.get('model', 'BMF')
        context = payload.get('context', {})

        if not user_id:
            print(json.dumps({'ok': False, 'error': 'user_id is required'}))
            return

        # Initialize recommendation system (singleton pattern would be better in production)
        reco_system = RecommendationSystem()

        # Get recommendations with provided context
        result = reco_system.get_recommendations(user_id, model_name, limit, context)

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({'ok': False, 'error': str(e)}))


if __name__ == '__main__':
    main()