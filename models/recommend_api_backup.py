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
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.preprocessing import LabelEncoder
import mysql.connector
from datetime import datetime
from dotenv import load_dotenv

# Database configuration from environment variables
# Note: When called from Node.js, env vars are passed through system environment
DB_CONFIG = {
    'host': os.getenv('DB_HOST', os.getenv('DBHOST', '127.0.0.1')),
    'user': os.getenv('DB_USERNAME', os.getenv('DBUSER', 'root')),
    'password': os.getenv('DB_PASSWORD', os.getenv('DBPASS', '')),
    'database': os.getenv('DB_DATABASE_NAME', os.getenv('DBNAME', 'ecom'))
}


class ENCM(keras.Model):
    def __init__(self, n_users, n_items, n_contexts, embedding_dim=50, context_dim=10, hidden_dims=[64, 32]):
        super(ENCM, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # User and item embeddings
        self.user_embedding = layers.Embedding(n_users, embedding_dim)
        self.item_embedding = layers.Embedding(n_items, embedding_dim)

        # Context embeddings
        self.context_embeddings = []
        for i, n_context in enumerate(n_contexts):
            self.context_embeddings.append(
                layers.Embedding(n_context, context_dim, name=f'context_{i}')
            )

        # Neural layers for processing interactions
        self.hidden_layers = []
        input_dim = embedding_dim * 2 + len(n_contexts) * context_dim

        for hidden_dim in hidden_dims:
            self.hidden_layers.append(layers.Dense(hidden_dim, activation='relu'))
            self.hidden_layers.append(layers.Dropout(0.2))
            input_dim = hidden_dim

        # Output layer
        self.output_layer = layers.Dense(1, activation='sigmoid')

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


class LNCM(keras.Model):
    def __init__(self, n_users, n_items, embedding_dim=50, hidden_dims=[64, 32]):
        super(LNCM, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # User and item embeddings
        self.user_embedding = layers.Embedding(n_users, embedding_dim)
        self.item_embedding = layers.Embedding(n_items, embedding_dim)

        # Linear combination layer
        self.linear_layer = layers.Dense(1, use_bias=True)

        # Neural layers
        self.hidden_layers = []
        input_dim = embedding_dim * 2

        for hidden_dim in hidden_dims:
            self.hidden_layers.append(layers.Dense(hidden_dim, activation='relu'))
            self.hidden_layers.append(layers.Dropout(0.2))
            input_dim = hidden_dim

        # Neural output layer
        self.neural_layer = layers.Dense(1, activation='sigmoid')

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


class NeuMF(keras.Model):
    def __init__(self, n_users, n_items, embedding_dim=50, hidden_dims=[64, 32, 16]):
        super(NeuMF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # GMF (Generalized Matrix Factorization) embeddings
        self.user_embedding_gmf = layers.Embedding(n_users, embedding_dim)
        self.item_embedding_gmf = layers.Embedding(n_items, embedding_dim)

        # MLP (Multi-Layer Perceptron) embeddings
        self.user_embedding_mlp = layers.Embedding(n_users, embedding_dim)
        self.item_embedding_mlp = layers.Embedding(n_items, embedding_dim)

        # MLP layers
        self.mlp_layers = []
        input_dim = embedding_dim * 2

        for hidden_dim in hidden_dims:
            self.mlp_layers.append(layers.Dense(hidden_dim, activation='relu'))
            self.mlp_layers.append(layers.Dropout(0.2))
            input_dim = hidden_dim

        # Final prediction layer (combines GMF and MLP)
        self.final_layer = layers.Dense(1, activation='sigmoid')

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


class BMF(keras.Model):
    def __init__(self, n_users, n_items, embedding_dim=50):
        super(BMF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # User and item embeddings
        self.user_embedding = layers.Embedding(n_users, embedding_dim)
        self.item_embedding = layers.Embedding(n_items, embedding_dim)

        # User and item biases
        self.user_bias = layers.Embedding(n_users, 1)
        self.item_bias = layers.Embedding(n_items, 1)

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

    def build_models(self):
        """Build and initialize all models"""
        n_users = self.data_stats['n_users']
        n_items = self.data_stats['n_items']

        # Context dimensions for ENCM - must match order in context_features array
        n_contexts = [
            self.data_stats['n_categories'],  # 0: category
            self.data_stats['n_brands'],      # 1: brand
            self.data_stats['n_devices'],     # 2: device_type
            self.data_stats['n_time_of_day'], # 3: time_of_day (always 4)
            self.data_stats['n_seasons'],     # 4: season (always 4)
            self.data_stats['n_genders']      # 5: gender (always 4)
        ]

        # Build models
        self.models['BMF'] = BMF(n_users, n_items, embedding_dim=50)
        self.models['NeuMF'] = NeuMF(n_users, n_items, embedding_dim=50, hidden_dims=[64, 32, 16])
        self.models['LNCM'] = LNCM(n_users, n_items, embedding_dim=50, hidden_dims=[64, 32])
        self.models['ENCM'] = ENCM(n_users, n_items, n_contexts, embedding_dim=50, context_dim=10, hidden_dims=[64, 32])

        # Compile models
        for name, model in self.models.items():
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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
                time_of_day = 0 if hour < 6 else (1 if hour < 12 else (2 if hour < 18 else 3))
                season = 0 if month <= 2 or month == 12 else (1 if month <= 5 else (2 if month <= 8 else 3))
                context['time_of_day'] = context.get('time_of_day', time_of_day)
                context['season'] = context.get('season', season)
                context['hour'] = context.get('hour', hour)
                context['month'] = context.get('month', month)

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
                'month': 6
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
                # Clamp user_id to valid range
                clamped_user_id = max(1, min(self.data_stats['n_users'], user_id))
                user_idx = self.encoders['user'].transform([clamped_user_id])[0]

                # Clamp item_ids to valid range
                valid_product_ids = [max(1, min(self.data_stats['n_items'], pid)) for pid in product_ids]
                item_indices = self.encoders['item'].transform(valid_product_ids)
            except Exception as e:
                return {'ok': False, 'error': f'Encoding error: {e}'}

            # Get user context (merge provided context with database context)
            context = self.get_user_context(user_id, provided_context)

            # Prepare input data
            n_items = len(product_ids)
            user_indices = np.full(n_items, user_idx)

            if model_name == 'ENCM':
                # For ENCM, we need context features as indices for embedding layers
                context_features = []
                preference_boost = []
                for pid in product_ids:
                    product = products_df[products_df['id'] == pid].iloc[0]

                    # Encode all context features as indices for embedding layers
                    # Order must match n_contexts in build_models()
                    category_code = self.encoders['category'].transform([product['categoryId'] or 'unknown'])[0]
                    brand_code = self.encoders['brand'].transform([product['brandId'] or 'unknown'])[0]
                    device_code = self.encoders['device'].transform([context['device_type']])[0]

                    # Context values are already encoded as 0,1,2,3 so they're valid indices
                    time_of_day_code = max(0, min(3, int(context.get('time_of_day', 1))))  # 0-3
                    season_code = max(0, min(3, int(context.get('season', 0))))           # 0-3
                    gender_code = max(0, min(3, int(context.get('gender', 3))))           # 0-3

                    context_features.append([
                        category_code,      # 0: category
                        brand_code,         # 1: brand
                        device_code,        # 2: device_type
                        time_of_day_code,   # 3: time_of_day
                        season_code,        # 4: season
                        gender_code         # 5: gender
                    ])

                    # Calculate preference boost with context awareness
                    boost = 0.0

                    # Seasonal boost (winter: coats, long pants, etc.)
                    if context.get('season') == 0:  # winter
                        winter_items = ['ao-khoac', 'quan-dai', 'dong-ho']  # coats, long pants, watches
                        if product['categoryId'] in winter_items:
                            boost += 0.15  # strong seasonal boost

                    # Device type boost (mobile: cheaper items, desktop: variety)
                    if context.get('device_type') == 'mobile':
                        # Assume mobile users prefer cheaper/better discount items
                        # This would need price data, for now just reduce boost
                        boost -= 0.05  # slight penalty for mobile (prefer cheaper items)
                    elif context.get('device_type') == 'desktop':
                        # Desktop users can browse more, prefer variety
                        boost += 0.05  # slight boost for variety

                    # Time of day boost
                    if context.get('time_of_day') == 1:  # morning - work/study items
                        morning_items = ['ao-thun', 'quan-dai', 'giay']
                        if product['categoryId'] in morning_items:
                            boost += 0.08

                    # Preferred categories and brands
                    if context.get('preferred_categories') and product['categoryId'] in context['preferred_categories']:
                        boost += 0.12  # boost for preferred category
                    if context.get('preferred_brands') and product['brandId'] in context['preferred_brands']:
                        boost += 0.12  # boost for preferred brand

                    preference_boost.append(boost)

                context_features = np.array(context_features)
                preference_boost = np.array(preference_boost)

                # Debug: check context features validity (will be in stderr)
                import sys
                print(f"DEBUG ENCM: context_features shape: {context_features.shape}", file=sys.stderr)
                print(f"DEBUG ENCM: context_features sample: {context_features[:3]}", file=sys.stderr)
                print(f"DEBUG ENCM: n_contexts: {[self.data_stats['n_categories'], self.data_stats['n_brands'], self.data_stats['n_devices'], self.data_stats['n_time_of_day'], self.data_stats['n_seasons'], self.data_stats['n_genders']]}", file=sys.stderr)

                # Get predictions
                predictions = model.predict([user_indices, item_indices, context_features], batch_size=32)

                # Apply preference boost
                predictions = predictions + preference_boost.reshape(-1, 1)

            else:
                # For other models, just user-item pairs
                predictions = model.predict([user_indices, item_indices], batch_size=32)

                # Apply preference boost for non-ENCM models too
                preference_boost = []
                for pid in product_ids:
                    product = products_df[products_df['id'] == pid].iloc[0]
                    boost = 0.0

                    # Seasonal boost for all models
                    if context.get('season') == 0:  # winter
                        winter_items = ['ao-khoac', 'quan-dai', 'dong-ho']
                        if product['categoryId'] in winter_items:
                            boost += 0.08

                    # Device type boost
                    if context.get('device_type') == 'mobile':
                        boost -= 0.03
                    elif context.get('device_type') == 'desktop':
                        boost += 0.03

                    # Time of day boost
                    if context.get('time_of_day') == 1:  # morning
                        morning_items = ['ao-thun', 'quan-dai', 'giay']
                        if product['categoryId'] in morning_items:
                            boost += 0.05

                    # Preferred categories and brands
                    if context.get('preferred_categories') and product['categoryId'] in context['preferred_categories']:
                        boost += 0.06
                    if context.get('preferred_brands') and product['brandId'] in context['preferred_brands']:
                        boost += 0.06

                    preference_boost.append(boost)

                preference_boost = np.array(preference_boost)
                predictions = predictions + preference_boost.reshape(-1, 1)

            # Get top recommendations
            predictions_flat = predictions.flatten()
            top_indices = np.argsort(predictions_flat)[::-1][:limit]

            recommendations = []
            for idx in top_indices:
                product_id = product_ids[idx]
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