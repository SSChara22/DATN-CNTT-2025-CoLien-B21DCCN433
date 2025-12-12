#!/usr/bin/env python3
"""
Simplified Recommendation API for ENCM debugging
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

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USERNAME', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_DATABASE_NAME', 'ecom')
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

        # Neural layers
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

        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        # Get context embeddings
        context_embs = []
        for i, context_embedding in enumerate(self.context_embeddings):
            context_embs.append(context_embedding(context_features[:, i]))

        # Concatenate
        all_features = tf.concat([user_emb, item_emb] + context_embs, axis=-1)

        # Neural network
        output = all_features
        for layer in self.hidden_layers:
            output = layer(output, training=training)

        # Final prediction
        prediction = self.output_layer(output)
        return prediction


class BMF(keras.Model):
    def __init__(self, n_users, n_items, embedding_dim=50):
        super(BMF, self).__init__()
        self.user_embedding = layers.Embedding(n_users, embedding_dim)
        self.item_embedding = layers.Embedding(n_items, embedding_dim)
        self.user_bias = layers.Embedding(n_users, 1)
        self.item_bias = layers.Embedding(n_items, 1)
        self.global_bias = self.add_weight(shape=(1,), initializer='zeros', trainable=True)

    def call(self, inputs, training=None):
        user_ids, item_ids = inputs
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        user_b = self.user_bias(user_ids)
        item_b = self.item_bias(item_ids)
        dot_product = tf.reduce_sum(user_emb * item_emb, axis=-1, keepdims=True)
        return tf.sigmoid(dot_product + user_b + item_b + self.global_bias)


class NeuMF(keras.Model):
    def __init__(self, n_users, n_items, embedding_dim=50, hidden_dims=[64, 32, 16]):
        super(NeuMF, self).__init__()
        self.user_embedding_gmf = layers.Embedding(n_users, embedding_dim)
        self.item_embedding_gmf = layers.Embedding(n_items, embedding_dim)
        self.user_embedding_mlp = layers.Embedding(n_users, embedding_dim)
        self.item_embedding_mlp = layers.Embedding(n_items, embedding_dim)

        self.mlp_layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            self.mlp_layers.append(layers.Dense(hidden_dim, activation='relu'))
            self.mlp_layers.append(layers.Dropout(0.2))
            input_dim = hidden_dim

        self.final_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=None):
        user_ids, item_ids = inputs
        user_emb_gmf = self.user_embedding_gmf(user_ids)
        item_emb_gmf = self.item_embedding_gmf(item_ids)
        gmf_output = tf.multiply(user_emb_gmf, item_emb_gmf)

        user_emb_mlp = self.user_embedding_mlp(user_ids)
        item_emb_mlp = self.item_embedding_mlp(item_ids)
        mlp_input = tf.concat([user_emb_mlp, item_emb_mlp], axis=-1)

        mlp_output = mlp_input
        for layer in self.mlp_layers:
            mlp_output = layer(mlp_output, training=training)

        combined = tf.concat([gmf_output, mlp_output], axis=-1)
        return self.final_layer(combined)


class LNCM(keras.Model):
    def __init__(self, n_users, n_items, embedding_dim=50, hidden_dims=[64, 32]):
        super(LNCM, self).__init__()
        self.user_embedding = layers.Embedding(n_users, embedding_dim)
        self.item_embedding = layers.Embedding(n_items, embedding_dim)
        self.linear_layer = layers.Dense(1, use_bias=True)

        self.hidden_layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(layers.Dense(hidden_dim, activation='relu'))
            self.hidden_layers.append(layers.Dropout(0.2))
            input_dim = hidden_dim

        self.neural_layer = layers.Dense(1, activation='sigmoid')
        self.alpha = self.add_weight(shape=(1,), initializer='uniform', trainable=True)

    def call(self, inputs, training=None):
        user_ids, item_ids = inputs
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        linear_output = self.linear_layer(tf.concat([user_emb, item_emb], axis=-1))

        neural_input = tf.concat([user_emb, item_emb], axis=-1)
        neural_output = neural_input
        for layer in self.hidden_layers:
            neural_output = layer(neural_output, training=training)
        neural_output = self.neural_layer(neural_output)

        return tf.sigmoid(self.alpha) * linear_output + (1 - tf.sigmoid(self.alpha)) * neural_output


class SimpleRecommender:
    def __init__(self):
        self.db_connection = mysql.connector.connect(**DB_CONFIG)
        self.encoders = {}
        self.data_stats = {}
        self.models = {}

        # Simple initialization
        self.data_stats = {
            'n_users': 24, 'n_items': 30,
            'n_categories': 10, 'n_brands': 8, 'n_devices': 3,
            'n_time_of_day': 4, 'n_seasons': 4, 'n_genders': 4
        }

        # Create simple encoders
        self.encoders['user'] = LabelEncoder()
        self.encoders['item'] = LabelEncoder()
        self.encoders['category'] = LabelEncoder()
        self.encoders['brand'] = LabelEncoder()
        self.encoders['device'] = LabelEncoder()

        # Fit with dummy data
        self.encoders['user'].fit(range(1, 25))  # users 1-24
        self.encoders['item'].fit(range(1, 31))  # items 1-30
        self.encoders['category'].fit(['ao-thun', 'quan-dai', 'giay', 'ao-khoac', 'dong-ho', 'unknown'])
        self.encoders['brand'].fit(['nike', 'chanel', 'outerity', 'yuumy', 'icondenim', 'prada', 'unknown'])
        self.encoders['device'].fit(['desktop', 'mobile', 'unknown'])

        # Build models
        n_contexts = [self.data_stats['n_categories'], self.data_stats['n_brands'],
                     self.data_stats['n_devices'], self.data_stats['n_time_of_day'],
                     self.data_stats['n_seasons'], self.data_stats['n_genders']]

        self.models['BMF'] = BMF(self.data_stats['n_users'], self.data_stats['n_items'])
        self.models['NeuMF'] = NeuMF(self.data_stats['n_users'], self.data_stats['n_items'])
        self.models['LNCM'] = LNCM(self.data_stats['n_users'], self.data_stats['n_items'])
        self.models['ENCM'] = ENCM(self.data_stats['n_users'], self.data_stats['n_items'], n_contexts)

    def get_recommendations(self, user_id, model_name, limit=10, context=None):
        try:
            if model_name not in self.models:
                return {'ok': False, 'error': f'Model {model_name} not found'}

            model = self.models[model_name]

            # Simple product list (1-30)
            product_ids = np.arange(1, 31)

            # Convert to model indices (clamp to valid range)
            clamped_user_id = max(1, min(24, user_id))  # users 1-24
            user_indices = np.full(len(product_ids), clamped_user_id - 1)  # 0-based
            item_indices = product_ids - 1  # 0-based (items 0-29)

            if model_name == 'ENCM':
                # Create context features for ENCM
                context_features = []
                for pid in product_ids:
                    # Simple context encoding
                    category_code = 0  # default
                    brand_code = 0     # default
                    device_code = 0    # desktop
                    time_code = context.get('time_of_day', 1) if context else 1
                    season_code = context.get('season', 0) if context else 0
                    gender_code = context.get('gender', 0) if context else 0

                    context_features.append([
                        category_code, brand_code, device_code,
                        time_code, season_code, gender_code
                    ])

                context_features = np.array(context_features)
                predictions = model.predict([user_indices, item_indices, context_features], batch_size=32, verbose=0)
            else:
                # BMF and other models
                predictions = model.predict([user_indices, item_indices], batch_size=32, verbose=0)

            # Get top recommendations
            predictions_flat = predictions.flatten()
            top_indices = np.argsort(predictions_flat)[::-1][:limit]

            recommendations = []
            for idx in top_indices:
                product_id = int(product_ids[idx])
                score = float(predictions_flat[idx])
                recommendations.append({
                    'productId': product_id,
                    'productName': f'Product {product_id}',
                    'brandName': f'Brand {(product_id % 7) + 1}',
                    'score': score
                })

            return {
                'ok': True,
                'items': recommendations,
                'context': context or {},
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

        # Initialize recommender
        recommender = SimpleRecommender()
        result = recommender.get_recommendations(user_id, model_name, limit, context)
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({'ok': False, 'error': str(e)}))


if __name__ == '__main__':
    main()
