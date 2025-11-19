import os
import sys
import json
import pickle
import numpy as np
import tensorflow as tf
from models.model_classes import BMF, NeuMF, LNCM

# Usage: python recommend.py <userId> <limit>

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def topn(scores, n):
    idx = np.argsort(-scores)[:n]
    return idx, scores[idx]

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"models": []}))
        return
    user_id = str(sys.argv[1])
    limit = int(sys.argv[2])

    root = os.path.dirname(os.path.abspath(__file__))
    proc_dir = os.path.join(root, 'processed_data')
    models_dir = os.path.join(root, 'models')

    user_encoder_path = os.path.join(proc_dir, 'user_encoder.pkl')
    item_encoder_path = os.path.join(proc_dir, 'item_encoder.pkl')

    if not (os.path.exists(user_encoder_path) and os.path.exists(item_encoder_path)):
        print(json.dumps({"models": []}))
        return

    user_encoder = load_pickle(user_encoder_path)
    item_encoder = load_pickle(item_encoder_path)

    # Build index maps
    user_to_index = {}
    item_to_index = {}
    # Accept both sklearn LabelEncoder and dict-like
    if hasattr(user_encoder, 'transform') and hasattr(user_encoder, 'classes_'):
        user_to_index = {str(u): i for i, u in enumerate(user_encoder.classes_)}
    elif isinstance(user_encoder, dict):
        user_to_index = {str(k): int(v) for k, v in user_encoder.items()}

    if hasattr(item_encoder, 'transform') and hasattr(item_encoder, 'classes_'):
        item_to_index = {str(it): i for i, it in enumerate(item_encoder.classes_)}
    elif isinstance(item_encoder, dict):
        item_to_index = {str(k): int(v) for k, v in item_encoder.items()}

    if str(user_id) not in user_to_index:
        print(json.dumps({"models": []}))
        return

    n_users = max(user_to_index.values()) + 1
    n_items = max(item_to_index.values()) + 1

    # candidate items are all items
    cand_item_indices = np.arange(n_items, dtype=np.int32)
    user_index = np.int32(user_to_index[str(user_id)])
    user_indices = np.full_like(cand_item_indices, user_index)

    outputs = []

    # helper to decode item index -> original id label
    def decode_item(idx: int) -> int:
        if hasattr(item_encoder, 'classes_'):
            try:
                return int(item_encoder.classes_[idx])
            except Exception:
                return int(idx)
        if isinstance(item_encoder, dict):
            # inverse mapping
            inv = getattr(decode_item, '_inv', None)
            if inv is None:
                inv = {v: k for k, v in item_encoder.items()}
                setattr(decode_item, '_inv', inv)
            try:
                return int(inv.get(idx, idx))
            except Exception:
                # if key is string digits
                val = inv.get(idx, idx)
                try:
                    return int(val)
                except Exception:
                    return int(idx)
        return int(idx)

    # BMF
    try:
        bmf = BMF(n_users=n_users, n_items=n_items, embedding_dim=50)
        bmf.build([(None,), (None,)])
        bmf.load_weights(os.path.join(models_dir, 'bmf.weights.h5'))
        bmf_scores = bmf([user_indices, cand_item_indices], training=False).numpy().reshape(-1)
        top_idx, top_scores = topn(bmf_scores, limit)
        outputs.append({
            "name": "BMF",
            "items": [{"productId": decode_item(int(idx)), "score": float(score)} for idx, score in zip(top_idx.tolist(), top_scores.tolist())],
            "metrics": {}
        })
    except Exception:
        pass

    # NeuMF
    try:
        neumf = NeuMF(n_users=n_users, n_items=n_items, embedding_dim=50)
        neumf.build([(None,), (None,)])
        neumf.load_weights(os.path.join(models_dir, 'neumf.weights.h5'))
        neu_scores = neumf([user_indices, cand_item_indices], training=False).numpy().reshape(-1)
        top_idx, top_scores = topn(neu_scores, limit)
        outputs.append({
            "name": "NeuMF",
            "items": [{"productId": decode_item(int(idx)), "score": float(score)} for idx, score in zip(top_idx.tolist(), top_scores.tolist())],
            "metrics": {}
        })
    except Exception:
        pass

    # LNCM
    try:
        lncm = LNCM(n_users=n_users, n_items=n_items, embedding_dim=50)
        lncm.build([(None,), (None,)])
        lncm.load_weights(os.path.join(models_dir, 'lncm.weights.h5'))
        ln_scores = lncm([user_indices, cand_item_indices], training=False).numpy().reshape(-1)
        top_idx, top_scores = topn(ln_scores, limit)
        outputs.append({
            "name": "LNCM",
            "items": [{"productId": decode_item(int(idx)), "score": float(score)} for idx, score in zip(top_idx.tolist(), top_scores.tolist())],
            "metrics": {}
        })
    except Exception:
        pass

    print(json.dumps({"models": outputs}))

if __name__ == '__main__':
    main()
