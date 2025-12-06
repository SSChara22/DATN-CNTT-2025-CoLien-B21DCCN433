#!/usr/bin/env python3
"""
Migrate training encoders/mappings from processed_data/*.pkl into MySQL tables
so realtime serving can read mappings from DB instead of .pkl files.

Creates tables if absent:
- rec_user_encoder(original_id VARCHAR(191) PRIMARY KEY, idx INT NOT NULL)
- rec_item_encoder(original_id VARCHAR(191) PRIMARY KEY, idx INT NOT NULL)
- rec_context_mapping(feature_name VARCHAR(64), original_value VARCHAR(191), idx INT NOT NULL,
                      PRIMARY KEY(feature_name, original_value))
- rec_context_meta(feature_name VARCHAR(64) PRIMARY KEY, position INT NOT NULL)

Env variables (or edit defaults below):
- DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME

Usage:
  python tools/migrate_encoders.py
"""
import os
import sys
import pickle
from typing import Dict, Any

try:
    import pymysql
except ImportError:
    print("Please install pymysql: pip install pymysql")
    sys.exit(1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(ROOT, 'processed_data')

USER_ENCODER_PKL = os.path.join(PROC_DIR, 'user_encoder.pkl')
ITEM_ENCODER_PKL = os.path.join(PROC_DIR, 'item_encoder.pkl')
CONTEXT_INFO_PKL = os.path.join(PROC_DIR, 'context_info.pkl')

DB_HOST = os.getenv('DB_HOST', '127.0.0.1')
DB_PORT = int(os.getenv('DB_PORT', '3306'))
DB_USER = os.getenv('DB_USER', 'root')
DB_PASS = os.getenv('DB_PASS', '')
DB_NAME = os.getenv('DB_NAME', 'ecom')


def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def to_mapping_from_encoder(enc) -> Dict[str, int]:
    """Support both sklearn LabelEncoder and dict mapping."""
    if hasattr(enc, 'classes_'):
        # classes_ could be array of strings; cast to str for safety
        return {str(v): int(i) for i, v in enumerate(enc.classes_)}
    if isinstance(enc, dict):
        # keys: original id (str/int), values: continuous indices
        out = {}
        for k, v in enc.items():
            try:
                out[str(k)] = int(v)
            except Exception:
                continue
        return out
    raise ValueError('Unsupported encoder type: ' + type(enc).__name__)


def connect_db():
    conn = pymysql.connect(host=DB_HOST, port=DB_PORT, user=DB_USER,
                           password=DB_PASS, database=DB_NAME,
                           charset='utf8mb4', autocommit=True)
    return conn


def ensure_tables(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rec_user_encoder (
              original_id VARCHAR(191) NOT NULL PRIMARY KEY,
              idx INT NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rec_item_encoder (
              original_id VARCHAR(191) NOT NULL PRIMARY KEY,
              idx INT NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rec_context_mapping (
              feature_name VARCHAR(64) NOT NULL,
              original_value VARCHAR(191) NOT NULL,
              idx INT NOT NULL,
              PRIMARY KEY (feature_name, original_value)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rec_context_meta (
              feature_name VARCHAR(64) NOT NULL PRIMARY KEY,
              position INT NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        )


def upsert_user_item_maps(conn, user_map: Dict[str, int], item_map: Dict[str, int]):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM rec_user_encoder")
        cur.execute("DELETE FROM rec_item_encoder")
        if user_map:
            cur.executemany(
                "INSERT INTO rec_user_encoder (original_id, idx) VALUES (%s, %s)",
                [(k, v) for k, v in user_map.items()]
            )
        if item_map:
            cur.executemany(
                "INSERT INTO rec_item_encoder (original_id, idx) VALUES (%s, %s)",
                [(k, v) for k, v in item_map.items()]
            )


def upsert_context_map(conn, context_info: Dict[str, Any]):
    # context_info expected keys: 'reverse_mappings' => { feature: { text: idx } }, 'feature_names' [...] (optional)
    rev = (context_info or {}).get('reverse_mappings') or {}
    rows = []
    for feat, mapping in rev.items():
        for original_value, idx in (mapping or {}).items():
            try:
                rows.append((feat, str(original_value), int(idx)))
            except Exception:
                continue
    with conn.cursor() as cur:
        # mapping values
        cur.execute("DELETE FROM rec_context_mapping")
        if rows:
            cur.executemany(
                "INSERT INTO rec_context_mapping (feature_name, original_value, idx) VALUES (%s, %s, %s)",
                rows
            )
        # feature order (position)
        order_list = (context_info or {}).get('feature_names') or []
        cur.execute("DELETE FROM rec_context_meta")
        if order_list:
            cur.executemany(
                "INSERT INTO rec_context_meta (feature_name, position) VALUES (%s, %s)",
                [(name, i) for i, name in enumerate(order_list)]
            )


def main():
    # Validate inputs
    missing = [p for p in [USER_ENCODER_PKL, ITEM_ENCODER_PKL, CONTEXT_INFO_PKL] if not os.path.exists(p)]
    if missing:
        print("Missing files:\n - " + "\n - ".join(missing))
        sys.exit(2)

    user_enc = load_pickle(USER_ENCODER_PKL)
    item_enc = load_pickle(ITEM_ENCODER_PKL)
    ctx_info = load_pickle(CONTEXT_INFO_PKL)

    user_map = to_mapping_from_encoder(user_enc)
    item_map = to_mapping_from_encoder(item_enc)

    conn = connect_db()
    try:
        ensure_tables(conn)
        upsert_user_item_maps(conn, user_map, item_map)
        upsert_context_map(conn, ctx_info)
    finally:
        conn.close()

    print("OK: migrated user/item/context mappings into DB")


if __name__ == '__main__':
    main()
