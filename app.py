"""
Recommendation System - Web Application
Uses trained models (ENCM, LNCM, NeuMF, BMF) to provide personalized recommendations
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import os

app = Flask(__name__)

# Global variables for models and data
models = {}
data = {}
encoders = {}

def _to_python(obj):
    """Convert NumPy/Pandas scalars to native Python types for JSON serialization."""
    import numpy as _np
    import pandas as _pd
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    if isinstance(obj, (_pd.Timestamp,)):
        return obj.isoformat()
    return obj

def load_models_and_data():
    """Load all trained models and necessary data"""
    global models, data, encoders
    
    print("Loading models and data...")
    
    # Load processed data
    try:
        data['cars_enhanced'] = pd.read_csv('processed_data/cars_enhanced.csv')
        data['items_info'] = pd.read_csv('processed_data/items_info.csv')
        data['users_info'] = pd.read_csv('processed_data/users_info.csv')

        # Load encoders
        with open('processed_data/user_encoder.pkl', 'rb') as f:
            encoders['user'] = pickle.load(f)
        with open('processed_data/item_encoder.pkl', 'rb') as f:
            encoders['item'] = pickle.load(f)
        
        # Load context information
        with open('processed_data/context_info.pkl', 'rb') as f:
            data['context_info'] = pickle.load(f)

        # Build an items catalog with metadata for UI/filters (category, brand, price, etc.)
        cars_cols = [c for c in ['iid', 'category', 'brand', 'price', 'product_id', 'rating'] if c in data['cars_enhanced'].columns]
        items_catalog = (
            data['cars_enhanced'][cars_cols]
            .drop_duplicates(subset=['iid'])
            .copy()
        )
        # Standardize category text field used by templates
        if 'category' in items_catalog.columns and 'category_text' not in items_catalog.columns:
            items_catalog['category_text'] = items_catalog['category']

        # Merge items_info with catalog metadata
        data['items_catalog'] = data['items_info'].merge(items_catalog, on='iid', how='left')

        print("✓ Data loaded successfully")
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    # Load models from weights (resilient)
    import json
    import sys
    sys.path.insert(0, 'models')
    from model_classes import ENCM, LNCM, NeuMF, BMF

    loaded_any = False
    # ENCM
    try:
        with open('models/encm_config.json', 'r') as f:
            config = json.load(f)
        models['ENCM'] = ENCM(
            n_users=config['n_users'],
            n_items=config['n_items'],
            n_contexts=config['n_contexts'],
            embedding_dim=config['embedding_dim'],
            context_dim=config['context_dim'],
            hidden_dims=config['hidden_dims']
        )
        dummy_input = [
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.zeros((1, len(config['n_contexts'])), dtype=np.int32)
        ]
        models['ENCM'](dummy_input)
        models['ENCM'].load_weights('models/encm.weights.h5')
        print("✓ ENCM loaded")
        loaded_any = True
    except Exception as e:
        print(f"Warning: ENCM not loaded: {e}")

    # LNCM
    try:
        with open('models/lncm_config.json', 'r') as f:
            config = json.load(f)
        models['LNCM'] = LNCM(
            n_users=config['n_users'],
            n_items=config['n_items'],
            embedding_dim=config['embedding_dim'],
            hidden_dims=config['hidden_dims']
        )
        dummy_input = [np.array([0], dtype=np.int32), np.array([0], dtype=np.int32)]
        models['LNCM'](dummy_input)
        models['LNCM'].load_weights('models/lncm.weights.h5')
        print("✓ LNCM loaded")
        loaded_any = True
    except Exception as e:
        print(f"Warning: LNCM not loaded: {e}")

    # NeuMF
    try:
        with open('models/neumf_config.json', 'r') as f:
            config = json.load(f)
        models['NeuMF'] = NeuMF(
            n_users=config['n_users'],
            n_items=config['n_items'],
            embedding_dim=config['embedding_dim'],
            hidden_dims=config.get('hidden_dims', [128, 64, 32])
        )
        dummy_input = [np.array([0], dtype=np.int32), np.array([0], dtype=np.int32)]
        models['NeuMF'](dummy_input)
        models['NeuMF'].load_weights('models/neumf.weights.h5')
        print("✓ NeuMF loaded")
        loaded_any = True
    except Exception as e:
        print(f"Warning: NeuMF not loaded: {e}")

    # BMF
    try:
        with open('models/bmf_config.json', 'r') as f:
            config = json.load(f)
        models['BMF'] = BMF(
            n_users=config['n_users'],
            n_items=config['n_items'],
            embedding_dim=config['embedding_dim']
        )
        dummy_input = [np.array([0], dtype=np.int32), np.array([0], dtype=np.int32)]
        models['BMF'](dummy_input)
        models['BMF'].load_weights('models/bmf.weights.h5')
        print("✓ BMF loaded")
        loaded_any = True
    except Exception as e:
        print(f"Warning: BMF not loaded: {e}")

    if not loaded_any:
        print("No neural models loaded. Falling back to Popularity recommender.")
        models['Popularity'] = 'fallback'
    else:
        print("✓ At least one model loaded successfully")
    return True


def get_context_features(context_dict):
    """Convert context dictionary to feature array"""
    features = []
    context_info = data['context_info']
    
    # Map readable context to encoded values
    for feature_name in context_info['feature_names']:
        if feature_name == 'time_of_day_encoded':
            time_text = context_dict.get('time_of_day', 'Morning')
            features.append(context_info['reverse_mappings']['time_of_day'].get(time_text, 0))
        elif feature_name == 'season_encoded':
            season_text = context_dict.get('season', 'Spring')
            features.append(context_info['reverse_mappings']['season'].get(season_text, 0))
        elif feature_name == 'device_type_encoded':
            device_text = context_dict.get('device_type', 'Mobile')
            features.append(context_info['reverse_mappings']['device_type'].get(device_text, 0))
        elif feature_name == 'category_encoded':
            category_text = context_dict.get('category', 'Sedan')
            features.append(context_info['reverse_mappings']['category'].get(category_text, 0))
        else:
            features.append(0)
    
    return np.array([features])


def get_recommendations(user_id, model_name='ENCM', context=None, top_k=10):
    """
    Get top-k recommendations for a user
    
    Args:
        user_id: User identifier
        model_name: Model to use (ENCM, LNCM, NeuMF, BMF)
        context: Dictionary of context features
        top_k: Number of recommendations
    
    Returns:
        List of recommended items with predicted ratings
    """
    # Encode user
    try:
        user_encoded = encoders['user'].transform([user_id])[0]
    except:
        return []
    
    # Get candidate items, optionally filter by category from context
    catalog_df = data.get('items_catalog', data['items_info'])
    if context and isinstance(context, dict):
        selected_category = context.get('category')
        if selected_category and 'category' in catalog_df.columns:
            filtered = catalog_df[catalog_df['category'] == selected_category]
            if not filtered.empty:
                all_items = filtered['iid'].unique()
            else:
                all_items = catalog_df['iid'].unique()
        else:
            all_items = catalog_df['iid'].unique()
    else:
        all_items = catalog_df['iid'].unique()
    all_items_encoded = encoders['item'].transform(all_items)
    
    # Prepare input
    n_items = len(all_items_encoded)
    user_ids = np.array([user_encoded] * n_items, dtype=np.int32)
    item_ids = all_items_encoded.astype(np.int32)
    
    # Get predictions
    model = models.get(model_name)
    if model is None or model_name == 'Popularity':
        # Fallback: popularity-based scoring within current candidate set
        # Merge candidate items with catalog metrics
        base = pd.DataFrame({'iid': all_items})
        # Use items_info if available for counts/avg_rating
        pop = data['items_info'][['iid', 'total_ratings', 'avg_rating']] if 'items_info' in data else pd.DataFrame()
        merged = base.merge(pop, on='iid', how='left')
        merged['total_ratings'] = merged['total_ratings'].fillna(0)
        merged['avg_rating'] = merged['avg_rating'].fillna(0.0)
        # Score
        merged['pop_score'] = merged['avg_rating'] * 0.6 + np.log1p(merged['total_ratings']) * 0.4
        merged = merged.sort_values('pop_score', ascending=False).head(int(top_k))
        recommendations = []
        source_df = data.get('items_catalog', data['items_info'])
        for _, row in merged.iterrows():
            item_id = row['iid']
            item_row = source_df[source_df['iid'] == item_id]
            item_info = item_row.iloc[0] if not item_row.empty else {}
            recommendations.append({
                'item_id': _to_python(item_id),
                'predicted_rating': float(row['pop_score']),
                'name': _to_python(item_info.get('name', f'Item {item_id}')),
                'full_name': _to_python(item_info.get('full_name', f'Item {item_id}')),
                'brand': _to_python(item_info.get('brand', 'Unknown')),
                'model': _to_python(item_info.get('model', 'Unknown')),
                'category': _to_python(item_info.get('category_text', item_info.get('category', 'Unknown'))),
                'price': _to_python(item_info.get('price', 0)),
                'year': _to_python(item_info.get('year', 0))
            })
        return recommendations
    
    if model_name == 'ENCM':
        # Context-aware model
        if context is None:
            context = {}
        context_features = get_context_features(context)
        context_features = np.repeat(context_features, n_items, axis=0)
        predictions = model.predict([user_ids, item_ids, context_features], verbose=0)
    else:
        # Non-context models (LNCM, NeuMF, BMF)
        predictions = model.predict([user_ids, item_ids], verbose=0)
    
    predictions = predictions.flatten()
    
    # Get top-k
    top_k_indices = np.argsort(predictions)[-int(top_k):][::-1]
    
    recommendations = []
    for idx in top_k_indices:
        item_id = all_items[idx]
        predicted_rating = predictions[idx]

        # Get item details from items_catalog (has category/brand/price)
        source_df = data.get('items_catalog', data['items_info'])
        item_row = source_df[source_df['iid'] == item_id]
        if item_row.empty:
            item_info = {}
        else:
            item_info = item_row.iloc[0]
        
        recommendations.append({
            'item_id': _to_python(item_id),
            'predicted_rating': float(predicted_rating),
            'name': _to_python(item_info.get('name', f"{item_info.get('brand', 'Brand')} {item_info.get('category', 'Item')}") if isinstance(item_info, dict) else (f"{item_info.get('brand', 'Brand')} {item_info.get('category', 'Item')}")),
            'full_name': _to_python(item_info.get('full_name', f"{item_info.get('brand', 'Brand')} {item_info.get('category', 'Item')}") if isinstance(item_info, dict) else (f"{item_info.get('brand', 'Brand')} {item_info.get('category', 'Item')}")),
            'brand': _to_python(item_info.get('brand', 'Unknown')),
            'model': _to_python(item_info.get('model', 'Unknown')),
            'category': _to_python(item_info.get('category_text', item_info.get('category', 'Unknown'))),
            'price': _to_python(item_info.get('price', 0)),
            'year': _to_python(item_info.get('year', 0))
        })
    
    return recommendations


@app.route('/')
def index():
    """Main page"""
    # Get list of users
    users = data['users_info']['uid'].unique()[:100]  # Limit to 100 users for demo
    # Cast to Python types for template rendering safety
    users = [ _to_python(u) for u in users ]
    
    # Get context options for UI
    context_options = {
        'time_of_day': list(data['context_info']['mappings']['time_of_day'].values()),
        'season': list(data['context_info']['mappings']['season'].values()),
        'device_type': list(data['context_info']['mappings']['device_type'].values()),
        'category': list(data['context_info']['mappings']['category'].values())
    }
    
    return render_template('index.html', 
                         users=users,
                         models=list(models.keys()),
                         context_options=context_options)


@app.route('/recommend', methods=['POST'])
def recommend():
    """API endpoint for getting recommendations"""
    try:
        user_id = request.json.get('user_id')
        model_name = request.json.get('model', 'ENCM')
        context = request.json.get('context', {})
        top_k = int(request.json.get('top_k', 10))
        
        recommendations = get_recommendations(user_id, model_name, context, top_k)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'user_id': _to_python(user_id),
            'model': _to_python(model_name)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/user_profile/<int:user_id>')
def user_profile(user_id: int):
    """Return user profile and recent history, using processed data (cars_enhanced)."""
    try:
        df = data.get('cars_enhanced')
        if df is None:
            return jsonify({'error': 'Data not loaded'}), 500

        user_df = df[df['uid'] == user_id]
        if user_df.empty:
            return jsonify({'error': 'User not found'}), 404

        profile = {
            'user_id': int(user_id),
            'total_interactions': int(len(user_df)),
            'categories': {str(k): int(v) for k, v in user_df['category'].value_counts().to_dict().items()},
            'brands': {str(k): int(v) for k, v in user_df['brand'].value_counts().to_dict().items()},
            'avg_rating': float(user_df['rating'].mean()) if 'rating' in user_df.columns else None,
            'avg_price': float(user_df['price'].mean()) if 'price' in user_df.columns else None,
            'device_usage': {str(k): int(v) for k, v in user_df['device_type'].value_counts().to_dict().items()},
            'recent_items': user_df.sort_values('timestamp', ascending=False).head(5)[
                ['category', 'brand', 'rating', 'price', 'timestamp']
            ].to_dict('records') if 'timestamp' in user_df.columns else []
        }

        return jsonify(profile)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/compare', methods=['POST'])
def compare_models():
    """Compare recommendations from all models"""
    try:
        user_id = request.json.get('user_id')
        context = request.json.get('context', {})
        top_k = int(request.json.get('top_k', 10))
        
        results = {}
        for model_name in models.keys():
            recommendations = get_recommendations(user_id, model_name, context, top_k)
            results[model_name] = recommendations
        
        return jsonify({
            'success': True,
            'results': results,
            'user_id': _to_python(user_id)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


if __name__ == '__main__':
    # Load models and data
    if load_models_and_data():
        print("\n" + "="*50)
        print("Item Recommendation System")
        print("="*50)
        print("Available models:", list(models.keys()))
        print("Number of users:", len(data['users_info']))
        print("Number of Items:", len(data['items_info']))
        print("Context features:", data['context_info']['feature_names'])
        print("="*50)
        print("\nStarting server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load models and data. Please check the files.")
