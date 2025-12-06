#!/usr/bin/env python3
import sys
import os
import json
import pickle
import numpy as np
import tensorflow as tf
import pymysql

# Ensure we can import model classes from models/model_classes.py
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
try:
    from model_classes import ENCM, LNCM, NeuMF, BMF
except Exception as e:
    print(json.dumps({"ok": False, "error": f"import_error: {e}"}))
    sys.exit(0)

MODELS_DIR = ROOT

# DB config (read from env) with compatibility to Node .env naming
_DB_HOST = os.getenv('DB_HOST') or os.getenv('DB_HOST', None)
DB_HOST = _DB_HOST if _DB_HOST else '127.0.0.1'
_DB_PORT = os.getenv('DB_PORT') or os.getenv('DB_PORT', None)
DB_PORT = int(_DB_PORT) if _DB_PORT else 3306
DB_USER = (os.getenv('DB_USER') or os.getenv('DB_USERNAME') or 'root')
DB_PASS = (os.getenv('DB_PASS') or os.getenv('DB_PASSWORD') or '')
DB_NAME = (os.getenv('DB_NAME') or os.getenv('DB_DATABASE_NAME') or 'ecom')

# Lazy singletons
_cached = {
    'encoders': None,          # {'user': dict(original_id->idx), 'item': dict(original_id->idx)}
    'context_info': None,      # {'reverse_mappings': {...}, 'feature_names': [...]} built from DB
    'idx_to_pid': None,        # {item_idx: productId} for active items
    'models': {},
    'configs': {}
}

def _load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def db_connect():
    return pymysql.connect(host=DB_HOST, port=DB_PORT, user=DB_USER,
                           password=DB_PASS, database=DB_NAME,
                           charset='utf8mb4', autocommit=True)

def load_resources():
    if _cached['encoders'] is None or _cached['idx_to_pid'] is None or _cached['context_info'] is None:
        conn = db_connect()
        try:
            user_map = {}
            item_map = {}
            idx_to_pid = {}
            # user encoder
            with conn.cursor() as cur:
                cur.execute("SELECT original_id, idx FROM rec_user_encoder")
                for oid, idx in cur.fetchall():
                    user_map[str(oid)] = int(idx)
            # item encoder (full mapping)
            with conn.cursor() as cur:
                cur.execute("SELECT original_id, idx FROM rec_item_encoder")
                for oid, idx in cur.fetchall():
                    item_map[str(oid)] = int(idx)
            # active candidates (statusId='S1') and idx->productId map
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT e.idx, e.original_id
                    FROM rec_item_encoder e
                    JOIN products p ON p.id = e.original_id
                    WHERE p.statusId = 'S1'
                """)
                for idx, oid in cur.fetchall():
                    idx_to_pid[int(idx)] = int(oid)
            # Fallback: if no active candidates found, try all mapped items (no status filter)
            if not idx_to_pid:
                with conn.cursor() as cur:
                    cur.execute("SELECT idx, original_id FROM rec_item_encoder")
                    for idx, oid in cur.fetchall():
                        idx_to_pid[int(idx)] = int(oid)
            # Last resort: derive popular items from Product.view and keep those that exist in item_map
            if not idx_to_pid:
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM products WHERE statusId='S1' ORDER BY view DESC LIMIT 500")
                    rows = cur.fetchall()
                    for (pid,) in rows:
                        # find encoder index for this product if present
                        if str(pid) in item_map:
                            idx_to_pid[int(item_map[str(pid)])] = int(pid)
            # context mappings
            reverse_mappings = {}
            feature_names = []
            with conn.cursor() as cur:
                cur.execute("SELECT feature_name, original_value, idx FROM rec_context_mapping")
                rows = cur.fetchall()
                for feat, val, idx in rows:
                    reverse_mappings.setdefault(feat, {})[str(val)] = int(idx)
            with conn.cursor() as cur:
                cur.execute("SELECT feature_name, position FROM rec_context_meta ORDER BY position ASC")
                feature_names = [r[0] for r in cur.fetchall()]
            _cached['encoders'] = {'user': user_map, 'item': item_map}
            _cached['idx_to_pid'] = idx_to_pid
            _cached['context_info'] = {'reverse_mappings': reverse_mappings, 'feature_names': feature_names}
        finally:
            conn.close()

def _decode_item_index(idx):
    # Decode using DB-provided idx_to_pid for active candidates
    m = _cached.get('idx_to_pid') or {}
    return int(m.get(int(idx), idx))

def _encode_user(user_id, user_map):
    try:
        return int(user_map.get(str(user_id))) if str(user_id) in user_map else None
    except Exception:
        return None

def _candidate_indices_from_db():
    # Use cached idx_to_pid keys as candidates (active items only)
    keys = sorted((_cached.get('idx_to_pid') or {}).keys())
    if not keys:
        return np.arange(0, dtype=np.int32)
    return np.array(keys, dtype=np.int32)

def _build_model(model_name, configs):
    if model_name in _cached['models']:
        return _cached['models'][model_name]
    import json as _json
    cfg_path = os.path.join(MODELS_DIR, f"{model_name.lower()}_config.json")
    with open(cfg_path, 'r') as f:
        cfg = _json.load(f)
    _cached['configs'][model_name] = cfg
    if model_name == 'ENCM':
        m = ENCM(n_users=cfg['n_users'], n_items=cfg['n_items'], n_contexts=cfg['n_contexts'],
                 embedding_dim=cfg['embedding_dim'], context_dim=cfg['context_dim'], hidden_dims=cfg['hidden_dims'])
        # build
        dummy_u = np.array([0], dtype=np.int32)
        dummy_i = np.array([0], dtype=np.int32)
        dummy_ctx = np.zeros((1, len(cfg['n_contexts'])), dtype=np.int32)
        m([dummy_u, dummy_i, dummy_ctx])
        m.load_weights(os.path.join(MODELS_DIR, 'encm.weights.h5'), skip_mismatch=True)
    elif model_name == 'LNCM':
        m = LNCM(n_users=cfg['n_users'], n_items=cfg['n_items'], embedding_dim=cfg['embedding_dim'], hidden_dims=cfg['hidden_dims'])
        m([np.array([0], dtype=np.int32), np.array([0], dtype=np.int32)])
        m.load_weights(os.path.join(MODELS_DIR, 'lncm.weights.h5'), skip_mismatch=True)
    elif model_name == 'NeuMF':
        m = NeuMF(n_users=cfg['n_users'], n_items=cfg['n_items'], embedding_dim=cfg['embedding_dim'], hidden_dims=cfg.get('hidden_dims', [64,32,16]))
        m([np.array([0], dtype=np.int32), np.array([0], dtype=np.int32)])
        m.load_weights(os.path.join(MODELS_DIR, 'neumf.weights.h5'), skip_mismatch=True)
    elif model_name == 'BMF':
        m = BMF(n_users=cfg['n_users'], n_items=cfg['n_items'], embedding_dim=cfg['embedding_dim'])
        m([np.array([0], dtype=np.int32), np.array([0], dtype=np.int32)])
        m.load_weights(os.path.join(MODELS_DIR, 'bmf.weights.h5'), skip_mismatch=True)
    else:
        raise ValueError(f"unsupported model {model_name}")
    _cached['models'][model_name] = m
    return m

def _context_to_features(context, context_info):
    if context_info is None:
        # no context used
        return None
    rev = context_info.get('reverse_mappings') or {}
    feature_names = context_info.get('feature_names') or []
    vals = []
    for name in feature_names:
        if name == 'time_of_day_encoded':
            vals.append(rev.get('time_of_day', {}).get(context.get('time_of_day','Morning'), 0))
        elif name == 'season_encoded':
            vals.append(rev.get('season', {}).get(context.get('season','Spring'), 0))
        elif name == 'device_type_encoded':
            vals.append(rev.get('device_type', {}).get(context.get('device_type','Mobile'), 0))
        elif name == 'category_encoded':
            vals.append(rev.get('category', {}).get(context.get('category','Sedan'), 0))
        else:
            vals.append(0)
    return np.array([vals], dtype=np.int32)

def handle_request(req):
    load_resources()
    enc_user = _cached['encoders']['user']
    user_id = req.get('user_id')
    limit = int(req.get('limit', 10))
    model_label = (req.get('model') or 'ENCM').strip()
    # Normalize model name
    model_name = {'encm':'ENCM','lncm':'LNCM','bmf':'BMF','neumf':'NeuMF'}.get(model_label.lower(), model_label)
    user_idx = _encode_user(str(user_id), enc_user)
    # Fallback unseen user to index 0 to avoid empty results
    if user_idx is None:
        user_idx = 0
    item_indices = _candidate_indices_from_db()
    if item_indices.size == 0:
        return {"ok": True, "model": model_name, "items": []}
    user_ids = np.full_like(item_indices, np.int32(user_idx))

    # Build model
    try:
        m = _build_model(model_name, _cached['configs'])
    except Exception as e:
        return {"ok": False, "error": f"load_model: {e}"}

    # Predict
    try:
        # Ensure candidate indices fit model's item embedding size
        cfg = _cached['configs'].get(model_name) or {}
        max_items = int(cfg.get('n_items', 0))
        if max_items > 0:
            item_indices = item_indices[item_indices < max_items]
            if item_indices.size == 0:
                return {"ok": True, "model": model_name, "items": []}
            user_ids = np.full_like(item_indices, np.int32(user_idx))
        if model_name == 'ENCM':
            ctx = req.get('context') or {}
            ctx_info = _cached['context_info']
            ctx_feats = _context_to_features(ctx, ctx_info)
            if ctx_feats is None:
                # Fallback zero context features matching expected dims
                cfg_encm = _cached['configs'].get('ENCM') or {}
                n = len(cfg_encm.get('n_contexts', []))
                ctx_feats = np.zeros((1, n), dtype=np.int32)
            ctx_feats = np.repeat(ctx_feats, item_indices.shape[0], axis=0)
            scores = m.predict([user_ids, item_indices, ctx_feats], verbose=0).reshape(-1)
        else:
            scores = m.predict([user_ids, item_indices], verbose=0).reshape(-1)
        # Top-K
        top_idx = np.argsort(-scores)[:limit]
        items = []
        for pos in top_idx.tolist():
            model_idx = int(item_indices[pos])
            items.append({
                'productId': _decode_item_index(model_idx),
                'score': float(scores[pos])
            })
        return {"ok": True, "model": model_name, "items": items}
    except Exception as e:
        return {"ok": False, "error": f"predict_error: {e}"}

if __name__ == '__main__':
    try:
        req = json.loads(sys.stdin.read() or '{}')
    except Exception:
        print(json.dumps({"ok": False, "error": "invalid_json"}))
        sys.exit(0)
    resp = handle_request(req)
    print(json.dumps(resp))
