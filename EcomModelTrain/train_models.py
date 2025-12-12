"""
Train recommendation models on e-commerce data from database
Trains BMF, NeuMF, LNCM, and ENCM models and saves weights
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

# Import model classes for training
from training_model_classes import BMF, NeuMF, LNCM, ENCM

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.set_visible_devices([], 'GPU')

def load_training_data():
    """Load preprocessed training data"""
    print("Loading training data...")

    # Load data
    train_df = pd.read_csv('training_data/train_data.csv')
    test_df = pd.read_csv('training_data/test_data.csv')

    # Load encoders and stats
    with open('training_data/user_encoder.pkl', 'rb') as f:
        user_encoder = pickle.load(f)
    with open('training_data/item_encoder.pkl', 'rb') as f:
        item_encoder = pickle.load(f)
    with open('training_data/context_encoders.pkl', 'rb') as f:
        context_encoders = pickle.load(f)
    with open('training_data/data_stats.pkl', 'rb') as f:
        data_stats = pickle.load(f)

    print(f"Loaded {len(train_df)} train samples, {len(test_df)} test samples")
    print(f"Users: {data_stats['n_users']}, Items: {data_stats['n_items']}")

    return train_df, test_df, user_encoder, item_encoder, context_encoders, data_stats

def create_tf_datasets(train_df, test_df, batch_size=32):
    """Create TensorFlow datasets for training"""

    # Training dataset - return ((user_ids, item_ids), ratings)
    train_ds = tf.data.Dataset.from_tensor_slices(({
        'user_ids': train_df['user_encoded'].values.astype(np.int32),
        'item_ids': train_df['item_encoded'].values.astype(np.int32)
    }, train_df['rating'].values.astype(np.float32))).shuffle(1000).batch(batch_size)

    # Test dataset - return ((user_ids, item_ids), ratings)
    test_ds = tf.data.Dataset.from_tensor_slices(({
        'user_ids': test_df['user_encoded'].values.astype(np.int32),
        'item_ids': test_df['item_encoded'].values.astype(np.int32)
    }, test_df['rating'].values.astype(np.float32))).batch(batch_size)

    return train_ds, test_ds

def create_context_datasets(train_df, test_df, context_encoders, data_stats, batch_size=32):
    """Create TensorFlow datasets with context features for ENCM"""

    # Context features for ENCM
    context_features = [
        'category_encoded', 'brand_encoded', 'device_encoded',
        'time_of_day', 'season', 'gender_encoded',
        'hour', 'month', 'day_of_week', 'is_weekend'
    ]

    # Training context dataset
    train_context = np.column_stack([
        train_df[feat].values.astype(np.int32) for feat in context_features
    ])
    train_context_ds = tf.data.Dataset.from_tensor_slices(({
        'user_ids': train_df['user_encoded'].values.astype(np.int32),
        'item_ids': train_df['item_encoded'].values.astype(np.int32),
        'context_features': train_context
    }, train_df['rating'].values.astype(np.float32))).shuffle(1000).batch(batch_size)

    # Test context dataset
    test_context = np.column_stack([
        test_df[feat].values.astype(np.int32) for feat in context_features
    ])
    test_context_ds = tf.data.Dataset.from_tensor_slices(({
        'user_ids': test_df['user_encoded'].values.astype(np.int32),
        'item_ids': test_df['item_encoded'].values.astype(np.int32),
        'context_features': test_context
    }, test_df['rating'].values.astype(np.float32))).batch(batch_size)

    return train_context_ds, test_context_ds

def train_bmf_model(train_ds, test_ds, n_users, n_items, epochs=20, save_path='models/bmf_model.h5'):
    """Train BMF (Bayesian Matrix Factorization) model"""
    print(f"\n--- Training BMF Model ---")
    print(f"Users: {n_users}, Items: {n_items}")

    model = BMF(n_users=n_users, n_items=n_items, embedding_dim=50)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    # Train
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        callbacks=[early_stop],
        verbose=1
    )

    # Save model
    os.makedirs('models', exist_ok=True)
    model.save_weights(save_path)
    print(f"BMF model saved to {save_path}")

    # Evaluate
    test_loss, test_mae = model.evaluate(test_ds, verbose=0)
    print(".4f")
    print(".4f")

    return model, history

def train_neumf_model(train_ds, test_ds, n_users, n_items, epochs=20, save_path='models/neumf_model.h5'):
    """Train NeuMF (Neural Matrix Factorization) model"""
    print(f"\n--- Training NeuMF Model ---")
    print(f"Users: {n_users}, Items: {n_items}")

    model = NeuMF(n_users=n_users, n_items=n_items, embedding_dim=50, hidden_dims=[64, 32, 16])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    # Train
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        callbacks=[early_stop],
        verbose=1
    )

    # Save model
    model.save_weights(save_path)
    print(f"NeuMF model saved to {save_path}")

    # Evaluate
    test_loss, test_mae = model.evaluate(test_ds, verbose=0)
    print(".4f")
    print(".4f")

    return model, history

def train_lncm_model(train_ds, test_ds, n_users, n_items, epochs=20, save_path='models/lncm_model.h5'):
    """Train LNCM (Linear-Neural Collaborative Model)"""
    print(f"\n--- Training LNCM Model ---")
    print(f"Users: {n_users}, Items: {n_items}")

    model = LNCM(n_users=n_users, n_items=n_items, embedding_dim=50, hidden_dims=[64, 32])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    # Train
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        callbacks=[early_stop],
        verbose=1
    )

    # Save model
    model.save_weights(save_path)
    print(f"LNCM model saved to {save_path}")

    # Evaluate
    test_loss, test_mae = model.evaluate(test_ds, verbose=0)
    print(".4f")
    print(".4f")

    return model, history

def train_encm_model(train_context_ds, test_context_ds, n_users, n_items, n_contexts, context_encoders, data_stats, epochs=20, save_path='models/encm_model.h5'):
    """Train ENCM (Enhanced Neural Collaborative Model) with context"""
    print(f"\n--- Training ENCM Model ---")
    print(f"Users: {n_users}, Items: {n_items}")
    print(f"Context features: {len(n_contexts)}")

    # Context dimensions - match actual number of classes
    context_dims = [
        len(context_encoders['category'].classes_),  # categories
        len(context_encoders['brand'].classes_),     # brands
        len(context_encoders['device'].classes_),    # devices
        4,  # time_of_day
        4,  # season
        4,  # gender (M, FE, O, unknown -> 0, 1, 2, 3)
        24, # hour
        12, # month (0-11)
        7,  # day_of_week
        2   # is_weekend
    ]

    model = ENCM(n_users=n_users, n_items=n_items, n_contexts=n_contexts,
                embedding_dim=50, context_dims=context_dims, hidden_dims=[64, 32])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    # Train
    history = model.fit(
        train_context_ds,
        validation_data=test_context_ds,
        epochs=epochs,
        callbacks=[early_stop],
        verbose=1
    )

    # Save model
    model.save_weights(save_path)
    print(f"ENCM model saved to {save_path}")

    # Evaluate
    test_loss, test_mae = model.evaluate(test_context_ds, verbose=0)
    print(".4f")
    print(".4f")

    return model, history

def save_model_configs(data_stats, context_encoders):
    """Save model configurations for inference"""

    # BMF config
    bmf_config = {
        'n_users': data_stats['n_users'],
        'n_items': data_stats['n_items'],
        'embedding_dim': 50
    }

    # NeuMF config
    neumf_config = {
        'n_users': data_stats['n_users'],
        'n_items': data_stats['n_items'],
        'embedding_dim': 50,
        'hidden_dims': [64, 32, 16]
    }

    # LNCM config
    lncm_config = {
        'n_users': data_stats['n_users'],
        'n_items': data_stats['n_items'],
        'embedding_dim': 50,
        'hidden_dims': [64, 32]
    }

    # ENCM config - fix n_contexts to match actual context features count
    encm_config = {
        'n_users': data_stats['n_users'],
        'n_items': data_stats['n_items'],
        'n_contexts': [feat[1] for feat in data_stats['context_features']],
        'embedding_dim': 50,
        'context_dims': [
            len(context_encoders['category'].classes_),  # 8
            len(context_encoders['brand'].classes_),     # 6
            len(context_encoders['device'].classes_),    # 3
            4, 4, 4, 24, 12, 7, 2  # other dims
        ],
        'hidden_dims': [64, 32]
    }

    # Save configs
    with open('models/bmf_config.json', 'w') as f:
        import json
        json.dump(bmf_config, f, indent=2)

    with open('models/neumf_config.json', 'w') as f:
        json.dump(neumf_config, f, indent=2)

    with open('models/lncm_config.json', 'w') as f:
        json.dump(lncm_config, f, indent=2)

    with open('models/encm_config.json', 'w') as f:
        json.dump(encm_config, f, indent=2)

    print("Model configurations saved")

def main():
    """Main training function"""
    print("=" * 60)
    print("TRAINING RECOMMENDATION MODELS ON E-COMMERCE DATA")
    print("=" * 60)
    print(f"Start time: {datetime.now()}")

    # Load data
    train_df, test_df, user_encoder, item_encoder, context_encoders, data_stats = load_training_data()

    # Create datasets
    train_ds, test_ds = create_tf_datasets(train_df, test_df)
    train_context_ds, test_context_ds = create_context_datasets(
        train_df, test_df, context_encoders, data_stats
    )

    n_users = data_stats['n_users']
    n_items = data_stats['n_items']
    n_contexts = [feat[1] for feat in data_stats['context_features']]

    # Train models
    results = {}

    # Skip training if models already exist
    import os
    if not os.path.exists('models/bmf_model.h5'):
        # BMF
        bmf_model, bmf_history = train_bmf_model(train_ds, test_ds, n_users, n_items)
        results['BMF'] = {'model': bmf_model, 'history': bmf_history}
    else:
        print("BMF model already exists, skipping...")

    if not os.path.exists('models/neumf_model.h5'):
        # NeuMF
        neumf_model, neumf_history = train_neumf_model(train_ds, test_ds, n_users, n_items)
        results['NeuMF'] = {'model': neumf_model, 'history': neumf_history}
    else:
        print("NeuMF model already exists, skipping...")

    if not os.path.exists('models/lncm_model.h5'):
        # LNCM
        lncm_model, lncm_history = train_lncm_model(train_ds, test_ds, n_users, n_items)
        results['LNCM'] = {'model': lncm_model, 'history': lncm_history}
    else:
        print("LNCM model already exists, skipping...")

    # ENCM - always train as we fixed the month issue
    encm_model, encm_history = train_encm_model(
        train_context_ds, test_context_ds, n_users, n_items,
        n_contexts, context_encoders, data_stats
    )
    results['ENCM'] = {'model': encm_model, 'history': encm_history}

    # Save model configurations
    save_model_configs(data_stats, context_encoders)

    # Save training results
    results_summary = {
        'training_date': str(datetime.now()),
        'data_stats': data_stats,
        'models_trained': list(results.keys()),
        'model_configs_saved': True
    }

    with open('training_data/training_results.pkl', 'wb') as f:
        pickle.dump(results_summary, f)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"End time: {datetime.now()}")
    print("All 4 models trained and saved:")
    print("- BMF (Bayesian Matrix Factorization)")
    print("- NeuMF (Neural Matrix Factorization)")
    print("- LNCM (Linear-Neural Collaborative Model)")
    print("- ENCM (Enhanced Neural Collaborative Model)")
    print("\nModels are ready for real-time inference!")

if __name__ == '__main__':
    main()
