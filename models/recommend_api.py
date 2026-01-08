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
            try:
                cur = self.db_connection.cursor()
                cur.execute("SET time_zone = '+07:00'")
                cur.close()
            except Exception as tz_e:
                print(f"Warning: could not set session time_zone: {tz_e}")
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
            # Prefer training-time configs to match checkpoint shapes
            def _load_json(path):
                try:
                    with open(path, 'r') as f:
                        return json.load(f)
                except Exception:
                    return None

            # Load configs for all models
            bmf_cfg = _load_json('models/bmf_config.json')
            neumf_cfg = _load_json('models/neumf_config.json')
            lncm_cfg = _load_json('models/lncm_config.json')
            encm_cfg = _load_json('models/encm_config.json')

            # ENCM with training model class and context dims
            from training_model_classes import ENCM as ENCMTraining
            if encm_cfg:
                n_users_enc = encm_cfg.get('n_users', self.data_stats['n_users'])
                n_items_enc = encm_cfg.get('n_items', self.data_stats['n_items'])
                n_contexts = encm_cfg.get('n_contexts', [feat[1] for feat in self.data_stats['context_features']])
                context_dims = encm_cfg.get('context_dims', [feat[1] for feat in self.data_stats['context_features']])
                hidden_dims = encm_cfg.get('hidden_dims', [64, 32])
                embedding_dim = encm_cfg.get('embedding_dim', 50)
            else:
                n_users_enc = self.data_stats['n_users']
                n_items_enc = self.data_stats['n_items']
                n_contexts = [feat[1] for feat in self.data_stats['context_features']]
                context_dims = [feat[1] for feat in self.data_stats['context_features']]
                hidden_dims = [64, 32]
                embedding_dim = 50

            self.models['ENCM'] = ENCMTraining(
                n_users=n_users_enc,
                n_items=n_items_enc,
                n_contexts=n_contexts,
                embedding_dim=embedding_dim,
                context_dims=context_dims,
                hidden_dims=hidden_dims
            )

            # Try load BMF/LNCM/NeuMF/ENCM weights; set Popularity fallback if fails
            loaded_any = False

            # BMF
            # [DISABLED] 
            # try:
            #     if bmf_cfg is None:
            #         bmf_cfg = {
            #             'n_users': self.data_stats.get('n_users'),
            #             'n_items': self.data_stats.get('n_items'),
            #             'embedding_dim': 50,
            #         }
            #     self.models['BMF'] = BMF(
            #         n_users=bmf_cfg.get('n_users', self.data_stats['n_users']),
            #         n_items=bmf_cfg.get('n_items', self.data_stats['n_items']),
            #         embedding_dim=bmf_cfg.get('embedding_dim', 50),
            #     )
            #     _bmf = self.models['BMF']
            #     _bmf.build([(None,), (None,)])
            #     _bmf.load_weights('models/bmf_model.h5')
            #     loaded_any = True
            # except Exception:
            #     if 'BMF' in self.models:
            #         del self.models['BMF']

            # LNCM
            # try:
            #    if lncm_cfg is None:
            #        lncm_cfg = {
            #            'n_users': self.data_stats.get('n_users'),
            #            'n_items': self.data_stats.get('n_items'),
            #            'embedding_dim': 50,
            #            'hidden_dims': [64, 32],
            #        }
            #    self.models['LNCM'] = LNCM(
            #        n_users=lncm_cfg.get('n_users', self.data_stats['n_users']),
            #        n_items=lncm_cfg.get('n_items', self.data_stats['n_items']),
            #        embedding_dim=lncm_cfg.get('embedding_dim', 50),
            #        hidden_dims=lncm_cfg.get('hidden_dims', [64, 32]),
            #    )
            #    _lncm = self.models['LNCM']
            #    _lncm.build([(None,), (None,)])
            #    _lncm.load_weights('models/lncm_model.h5')
            #    loaded_any = True
            #except Exception:
            #    if 'LNCM' in self.models:
            #        del self.models['LNCM']

            # NeuMF
            # [DISABLED]
            # try:
            #     if neumf_cfg is None:
            #         neumf_cfg = {
            #             'n_users': self.data_stats.get('n_users'),
            #             'n_items': self.data_stats.get('n_items'),
            #             'embedding_dim': 50,
            #             'hidden_dims': [64, 32, 16],
            #         }
            #     self.models['NeuMF'] = NeuMF(
            #         n_users=neumf_cfg.get('n_users', self.data_stats['n_users']),
            #         n_items=neumf_cfg.get('n_items', self.data_stats['n_items']),
            #         embedding_dim=neumf_cfg.get('embedding_dim', 50),
            #         hidden_dims=neumf_cfg.get('hidden_dims', [64, 32, 16]),
            #     )
            #     _neumf = self.models['NeuMF']
            #     _neumf.build([(None,), (None,)])
            #     _neumf.load_weights('models/neumf_model.h5')
            #     loaded_any = True
            # except Exception:
            #     if 'NeuMF' in self.models:
            #         del self.models['NeuMF']
            try:
                model = self.models.get('ENCM')
                if model is not None:
                    model.build([(None,), (None,), (None, 10)])
                    model.load_weights('models/encm_model.h5')
                    loaded_any = True
            except Exception as me:
                print(f"Warning: ENCM initial load failed: {me}")
                # Retry with encoder-based dimensions
                try:
                    from training_model_classes import ENCM as ENCMTraining
                    n_users_alt = len(self.encoders['user'].classes_)
                    n_items_alt = len(self.encoders['item'].classes_)
                    self.models['ENCM'] = ENCMTraining(n_users=n_users_alt, n_items=n_items_alt, n_contexts=n_contexts, embedding_dim=50, context_dims=context_dims, hidden_dims=[64, 32])
                    model = self.models['ENCM']
                    model.build([(None,), (None,), (None, 10)])
                    model.load_weights('models/encm_model.h5')
                    loaded_any = True
                    print(f"✓ ENCM loaded after retry with encoder-based dimensions ({n_users_alt} users, {n_items_alt} items)")
                except Exception as me2:
                    print(f"Warning: ENCM not loaded after retry: {me2}")
                    if 'ENCM' in self.models:
                        del self.models['ENCM']

            if not loaded_any:
                print("No neural models loaded. Falling back to Popularity recommender.")
                self.models['Popularity'] = 'fallback'
            else:
                print("✓ At least one model loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.models = {'Popularity': 'fallback'}

    def get_user_context(self, user_id, provided_context=None):
        """Get current context for user"""
        try:
            context = provided_context or {}

            # Get user gender (keep as string; map later for model)
            if 'gender' not in context:
                user_query = f"SELECT genderId FROM users WHERE id = {user_id}"
                user_df = pd.read_sql(user_query, self.db_connection)
                gender = user_df.iloc[0]['genderId'] if not user_df.empty else None
                context['gender'] = gender if gender in ['M','FE','O'] else 'unknown'

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
            context['month'] = context.get('month', month - 1)
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
                # Choose best available fallback order
                for alt in ['ENCM', 'LNCM', 'NeuMF', 'BMF', 'Popularity']:
                    if alt in self.models:
                        model_name = alt
                        break
                else:
                    return {'ok': False, 'error': f'Model {model_name} not found'}

            model = self.models.get(model_name)

            # Early role check: only R2 or value 'user'
            try:
                role_q = f"""
                    SELECT u.roleId,
                           a.value AS role_value,
                           a.code AS role_code
                    FROM users u
                    LEFT JOIN allcodes a
                      ON a.type = 'ROLE' AND a.code = u.roleId
                    WHERE u.id = {int(user_id)}
                """
                role_df = pd.read_sql(role_q, self.db_connection)
                if role_df.empty:
                    return {'ok': False, 'error': 'User not found or no role assigned'}
                role_value = str(role_df.iloc[0].get('role_value') or '').lower()
                user_role_code = str(role_df.iloc[0].get('roleId') or '').upper()
                if not (user_role_code == 'R2' or role_value == 'user'):
                    return {'ok': False, 'error': 'User role not permitted for recommendations'}
            except Exception as e:
                return {'ok': False, 'error': f'Role check failed: {e}'}

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

                # Guard indices within model embeddings
                model_n_users = getattr(model, 'n_users', None)
                model_n_items = getattr(model, 'n_items', None)
                if isinstance(model_n_users, int) and user_idx >= model_n_users:
                    user_idx = 0
                if isinstance(model_n_items, int):
                    mask = item_indices < model_n_items
                    if not np.any(mask):
                        model_name = 'Popularity'
                        model = self.models.get('Popularity')
                    else:
                        item_indices = item_indices[mask]
                        valid_product_ids = list(np.array(valid_product_ids)[mask])
            except Exception as e:
                return {'ok': False, 'error': f'Encoding error: {e}'}

            # Get user context
            context = self.get_user_context(user_id, provided_context)

            # Prepare input data
            n_items = len(valid_product_ids)
            if n_items == 0:
                model_name = 'Popularity'
                model = self.models.get('Popularity')
                valid_product_ids = [int(self.encoders['item'].classes_[0])]
                n_items = 1
                item_indices = self.encoders['item'].transform(valid_product_ids)
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
            elif model_name == 'LNCM':
                # Two-input model without explicit context (LNCM active)
                with SuppressOutput():
                    predictions = model.predict([user_indices, item_indices], batch_size=32, verbose=0)
            elif model_name in ['BMF', 'NeuMF']:
                with SuppressOutput():
                    predictions = model.predict([user_indices, item_indices], batch_size=32, verbose=0)
            else:
                # Popularity fallback or explicit Popularity
                try:
                    valid_ids_tuple = tuple(int(x) for x in valid_product_ids)
                    gender_filter = context.get('gender', 'unknown')
                    base_query = f"""
                        SELECT p.id AS productId,
                               SUM(CASE WHEN i.actionCode='purchase' THEN 3 WHEN i.actionCode='cart' THEN 2 ELSE 1 END) AS pop_score
                        FROM interactions i
                        JOIN products p ON p.id = i.productId
                        LEFT JOIN users u ON u.id = i.userId
                        WHERE p.id IN {valid_ids_tuple}
                    """
                    if gender_filter in ['M','FE','O']:
                        base_query += f" AND u.genderId = '{gender_filter}'"
                    # Time-of-day filter
                    tod = context.get('time_of_day', 'morning')
                    tod_map = {'night': (0,6), 'morning': (6,12), 'afternoon': (12,18), 'evening': (18,24)}
                    if isinstance(tod, str) and tod in tod_map:
                        h0, h1 = tod_map[tod]
                        base_query += f" AND HOUR(i.timestamp) >= {h0} AND HOUR(i.timestamp) < {h1}"
                    elif isinstance(tod, int):
                        tb = {0:(0,6),1:(6,12),2:(12,18),3:(18,24)}
                        if tod in tb:
                            h0, h1 = tb[tod]
                            base_query += f" AND HOUR(i.timestamp) >= {h0} AND HOUR(i.timestamp) < {h1}"
                    # Weekend filter
                    is_weekend = int(context.get('is_weekend', 0))
                    if is_weekend == 1:
                        base_query += " AND WEEKDAY(i.timestamp) IN (5,6)"
                    else:
                        base_query += " AND WEEKDAY(i.timestamp) IN (0,1,2,3,4)"
                    base_query += " AND i.timestamp >= DATE_SUB(NOW(), INTERVAL 180 DAY) GROUP BY p.id"
                    priors_df = pd.read_sql(base_query, self.db_connection)
                    prior_scores = {int(pid): 0.0 for pid in valid_product_ids}
                    if not priors_df.empty:
                        priors_df['pop_score'] = priors_df['pop_score'].astype(float)
                        max_pop = priors_df['pop_score'].max()
                        priors_df['norm'] = priors_df['pop_score'] / max_pop if max_pop > 0 else 0.0
                        for _, r in priors_df.iterrows():
                            prior_scores[int(r['productId'])] = float(r['norm'])
                    # Score = prior + preference boosts (time-gated)
                    preferred_brands = set(context.get('preferred_brands', []) or [])
                    preferred_categories = set(context.get('preferred_categories', []) or [])
                    brand_series = products_df.set_index('id')['brandId']
                    category_series = products_df.set_index('id')['categoryId']
                    scores = []
                    for pid in valid_product_ids:
                        base = prior_scores.get(int(pid), 0.0)
                        boost = 0.0
                        b = brand_series.get(pid)
                        c = category_series.get(pid)
                        has_prefs = (len(preferred_brands) > 0) or (len(preferred_categories) > 0)
                        if b in preferred_brands and b is not None:
                            boost += 0.8 if has_prefs else 0.3
                        if c in preferred_categories and c is not None:
                            boost += 0.8 if has_prefs else 0.3
                        if (b in preferred_brands and b is not None) and (c in preferred_categories and c is not None):
                            boost += 0.2
                        scores.append(min(1.0, base + boost))
                    scores = np.array(scores, dtype=np.float32)
                    # Preferred-first rerank with time constraint
                    is_pref = np.zeros_like(scores, dtype=np.int32)
                    for i, pid in enumerate(valid_product_ids):
                        b = brand_series.get(pid)
                        c = category_series.get(pid)
                        if (b in preferred_brands and b is not None) or (c in preferred_categories and c is not None):
                            is_pref[i] = 1
                    score_order_desc = np.argsort(scores)[::-1]
                    min_prior = 0.25
                    pref_indices = [idx for idx in score_order_desc if is_pref[idx] == 1 and prior_scores.get(int(valid_product_ids[idx]), 0.0) >= min_prior]
                    nonpref_indices = [idx for idx in score_order_desc if is_pref[idx] == 0]
                    if len(pref_indices) > 0:
                        preferred_quota = min(len(pref_indices), max(int(0.5 * limit), 3))
                        need = max(limit - preferred_quota, 0)
                        top_indices = pref_indices[:preferred_quota] + nonpref_indices[:need]
                    else:
                        top_indices = score_order_desc[:limit]
                    predictions_flat = scores
                except Exception as e:
                    return {'ok': False, 'error': str(e)}

            # Compute priors for ENCM blend/padding
            try:
                valid_ids_tuple = tuple(int(x) for x in valid_product_ids)
                gender_filter = context.get('gender', 'unknown')
                base_query = f"""
                    SELECT p.id AS productId,
                           SUM(CASE WHEN i.actionCode='purchase' THEN 3 WHEN i.actionCode='cart' THEN 2 ELSE 1 END) AS pop_score
                    FROM interactions i
                    JOIN products p ON p.id = i.productId
                    LEFT JOIN users u ON u.id = i.userId
                    WHERE p.id IN {valid_ids_tuple}
                """
                # Map int time_of_day
                try:
                    hour = int(context.get('hour', 12))
                    dow = int(context.get('day_of_week', 0))
                    is_weekend = int(context.get('is_weekend', 0))
                    time_of_day = int(context.get('time_of_day', 1))
                    tod_bounds = {0:(0,6),1:(6,12),2:(12,18),3:(18,24)}
                    if time_of_day in tod_bounds:
                        h0, h1 = tod_bounds[time_of_day]
                        base_query += f" AND HOUR(i.timestamp) >= {h0} AND HOUR(i.timestamp) < {h1}"
                    if is_weekend == 1:
                        base_query += " AND WEEKDAY(i.timestamp) IN (5,6)"
                    else:
                        base_query += " AND WEEKDAY(i.timestamp) IN (0,1,2,3,4)"
                    base_query += " AND i.timestamp >= DATE_SUB(NOW(), INTERVAL 180 DAY)"
                except Exception:
                    pass
                if gender_filter in ['M','FE','O']:
                    base_query += f" AND u.genderId = '{gender_filter}'"
                base_query += " GROUP BY p.id"
                priors_df = pd.read_sql(base_query, self.db_connection)
                if priors_df.empty:
                    prior_scores = {int(pid): 0.0 for pid in valid_product_ids}
                else:
                    priors_df['pop_score'] = priors_df['pop_score'].astype(float)
                    max_pop = priors_df['pop_score'].max()
                    priors_df['norm'] = priors_df['pop_score'] / max_pop if max_pop > 0 else 0.0
                    prior_scores = {int(r['productId']): float(r['norm']) for _, r in priors_df.iterrows()}
                    for pid in valid_product_ids:
                        if int(pid) not in prior_scores:
                            prior_scores[int(pid)] = 0.0
            except Exception:
                prior_scores = {int(pid): 0.0 for pid in valid_product_ids}

            predictions_flat = predictions.flatten() if 'predictions' in locals() else predictions_flat
            # History count (cold start)
            try:
                hist_df = pd.read_sql(
                    f"""
                    SELECT COUNT(*) AS cnt
                    FROM interactions
                    WHERE userId = {int(user_id)}
                      AND actionCode IN ('cart','purchase','view')
                    """,
                    self.db_connection
                )
                history_count = int(hist_df.iloc[0]['cnt']) if not hist_df.empty else 0
            except Exception:
                history_count = 0

            cold_threshold = 10
            is_cold = history_count < cold_threshold

            if model_name == 'ENCM' and is_cold:
                # Normalize predictions 0..1
                pmin = float(np.min(predictions_flat))
                pmax = float(np.max(predictions_flat))
                pred_norm = (predictions_flat - pmin) / (pmax - pmin + 1e-8)
                priors_vec = np.array([prior_scores[int(pid)] for pid in valid_product_ids], dtype=np.float32)
                # Preference boosts gated by priors
                preferred_brands = set(context.get('preferred_brands', []) or [])
                preferred_categories = set(context.get('preferred_categories', []) or [])
                brand_boost = np.zeros_like(priors_vec)
                category_boost = np.zeros_like(priors_vec)
                brand_series = products_df.set_index('id')['brandId']
                category_series = products_df.set_index('id')['categoryId']
                for i, pid in enumerate(valid_product_ids):
                    b = brand_series.get(pid)
                    c = category_series.get(pid)
                    if b in preferred_brands and b is not None:
                        brand_boost[i] = 1.0
                    if c in preferred_categories and c is not None:
                        category_boost[i] = 1.0
                threshold = 10.0
                k = 0.5
                alpha_tmp = 1.0 / (1.0 + np.exp(-k * (history_count - threshold)))
                coldness = 1.0 - alpha_tmp
                has_prefs = (len(preferred_brands) > 0) or (len(preferred_categories) > 0)
                w_brand = 0.8 if has_prefs else 0.3
                w_cat = 0.8 if has_prefs else 0.3
                raw_boost = (w_brand * brand_boost + w_cat * category_boost)
                boost_vec = coldness * raw_boost * np.array([prior_scores[int(pid)] for pid in valid_product_ids], dtype=np.float32)
                priors_vec = np.clip(priors_vec + boost_vec, 0.0, 1.0)
                alpha = 1.0 / (1.0 + np.exp(-0.5 * (history_count - 10.0)))
                if not has_prefs:
                    alpha = max(0.2, alpha * 0.5)
                else:
                    alpha = min(alpha, 0.35)
                blended = alpha * pred_norm + (1 - alpha) * priors_vec
                time_weight = 0.5 + 0.5 * priors_vec
                blended = blended * time_weight
                predictions_flat = blended
                # Preference-first rerank with time constraint
                is_pref = np.zeros_like(predictions_flat, dtype=np.int32)
                for i, pid in enumerate(valid_product_ids):
                    b = brand_series.get(pid)
                    c = category_series.get(pid)
                    if (b in preferred_brands and b is not None) or (c in preferred_categories and c is not None):
                        is_pref[i] = 1
                order_desc = np.argsort(predictions_flat)[::-1]
                min_prior = 0.25
                priors_vec2 = np.array([prior_scores.get(int(pid), 0.0) for pid in valid_product_ids], dtype=np.float32)
                pref_indices = [idx for idx in order_desc if is_pref[idx] == 1 and priors_vec2[idx] >= min_prior]
                nonpref_indices = [idx for idx in order_desc if is_pref[idx] == 0]
                if len(pref_indices) > 0:
                    preferred_quota = min(len(pref_indices), max(int(0.5 * limit), 3))
                    need = max(limit - preferred_quota, 0)
                    top_indices = list(pref_indices[:preferred_quota]) + list(nonpref_indices[:need])
                else:
                    top_indices = order_desc[:limit]
            elif model_name == 'Popularity' or model == 'fallback':
                # top_indices computed in popularity path
                pass
            else:
                # Warm user path: lightly rerank ENCM predictions by time-consistent priors and preferences
                if model_name == 'ENCM':
                    try:
                        priors_vec = np.array([prior_scores.get(int(pid), 0.0) for pid in valid_product_ids], dtype=np.float32)
                        # Small time-aware calibration
                        calibrated = predictions_flat + 0.15 * priors_vec
                        # Small preference bonus gated by time prior
                        preferred_brands = set(context.get('preferred_brands', []) or [])
                        preferred_categories = set(context.get('preferred_categories', []) or [])
                        brand_series = products_df.set_index('id')['brandId']
                        category_series = products_df.set_index('id')['categoryId']
                        for i, pid in enumerate(valid_product_ids):
                            if priors_vec[i] >= 0.25:
                                b = brand_series.get(pid)
                                c = category_series.get(pid)
                                if (b in preferred_brands and b is not None) or (c in preferred_categories and c is not None):
                                    calibrated[i] += 0.07
                        order_desc = np.argsort(calibrated)[::-1]
                        top_indices = order_desc[:limit]
                        predictions_flat = calibrated
                    except Exception:
                        order_desc = np.argsort(predictions_flat)[::-1]
                        top_indices = order_desc[:limit]
                else:
                    order_desc = np.argsort(predictions_flat)[::-1]
                    top_indices = order_desc[:limit]

            # Build recommendations
            recommendations = []
            for idx in top_indices:
                product_id = valid_product_ids[int(idx)]
                score = float(predictions_flat[int(idx)])
                product_row = products_df[products_df['id'] == product_id].iloc[0]
                product_name = product_row['name']
                brand_name = product_row['brandId'] or 'Unknown Brand'
                recommendations.append({
                    'productId': int(product_id),
                    'productName': product_name,
                    'brandName': brand_name,
                    'score': score
                })

            # Padding to ensure limit
            if len(recommendations) < limit:
                try:
                    existing = set([r['productId'] for r in recommendations])
                    need = limit - len(recommendations)
                    extra = [pid for pid in sorted(valid_product_ids, key=lambda x: prior_scores.get(int(x), 0.0), reverse=True) if int(pid) not in existing]
                    for pid in extra[:need]:
                        product_row = products_df[products_df['id'] == pid].iloc[0]
                        recommendations.append({
                            'productId': int(pid),
                            'productName': product_row['name'],
                            'brandName': product_row['brandId'] or 'Unknown Brand',
                            'score': float(prior_scores.get(int(pid), 0.0))
                        })
                except Exception:
                    pass

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
