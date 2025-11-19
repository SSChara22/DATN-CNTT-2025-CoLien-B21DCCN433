# file: seed_images_from_unsplash.py

import re
import time
import requests
import mysql.connector
from mysql.connector import Error

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "ecom",
}

HTTP_TIMEOUT = 15

# ============================
# FIXED UNSPLASH IMAGE MAP
# ============================
CATEGORY_IMAGE = {
    "t-shirt": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab",
    "bag": "https://images.unsplash.com/photo-1526170375885-4d8ecf77b99f",
    "sunglasses": "https://images.unsplash.com/photo-1511497584788-876760111969",
    "shirt": "https://images.unsplash.com/photo-1520975661595-6453be3f7070",
    "hat": "https://images.unsplash.com/photo-1503341455253-b2e723bb3dbb",
    "jeans": "https://images.unsplash.com/photo-1495121605193-b116b5b09a6b",
    "cardigan": "https://images.unsplash.com/photo-1530099486328-e021101a3362",
    "shoes": "https://images.unsplash.com/photo-1520256862855-398228c41684",
    "cargo-pants": "https://images.unsplash.com/photo-1539109136881-3be0616acf4e",
    "jacket": "https://images.unsplash.com/photo-1521577352947-9bb58764b69a",
    "hoodie": "https://images.unsplash.com/photo-1520975918318-3bbf2c5f3c56",
    "shorts": "https://images.unsplash.com/photo-1490481651871-ab68de25d43d",
    "skirt": "https://images.unsplash.com/photo-1539008835657-9e8e9680c956",
    "dress": "https://images.unsplash.com/photo-1520962918287-7448c2878f65",
    "sneakers": "https://images.unsplash.com/photo-1528701800489-20be7c57a89f",
    "blouse": "https://images.unsplash.com/photo-1521577352947-9bb58764b69a",
    "polo-shirt": "https://images.unsplash.com/photo-1530092285049-1c42085fd395",
    "boots": "https://images.unsplash.com/photo-1520256862855-398228c41684",
    "coat": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab",
    "sweater": "https://images.unsplash.com/photo-1522098543979-ffc7f79d4075",
    "default": "https://images.unsplash.com/photo-1512436991641-6745cdb1723f"
}

# ============================

def slugify(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"\s+", "-", text)
    text = text.replace("'", "")
    return text

def get_rows_to_update(cursor, limit=None, force_all=False):
    sql = """
    SELECT
      pi.id AS image_id,
      pd.id AS productdetail_id,
      COALESCE(ac.value, p.categoryId) AS category_value
    FROM productimages pi
    JOIN productdetails pd ON pd.id = pi.productdetailId
    JOIN products p ON p.id = pd.productId
    LEFT JOIN allcodes ac ON ac.type='CATEGORY' AND ac.code = p.categoryId
    WHERE %s
    """
    cond = "pi.image IS NULL" if not force_all else "1=1"
    sql = sql % cond

    if limit:
        sql += " LIMIT %s"
        cursor.execute(sql, (limit,))
    else:
        cursor.execute(sql)

    return cursor.fetchall()

def download_image_bytes(url: str) -> bytes | None:
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT)
        if r.status_code == 200 and r.content:
            return r.content
        return None
    except requests.RequestException:
        return None

def main(force_all=False, batch_limit=None, sleep_between=0.2):
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        conn.autocommit = False
        cursor = conn.cursor(dictionary=True)

        rows = get_rows_to_update(cursor, limit=batch_limit, force_all=force_all)
        print(f"Found {len(rows)} rows to update.")

        if not rows:
            return

        update_sql = """
            UPDATE productimages
            SET image = %s, caption = NULL, updatedAt = NOW()
            WHERE id = %s
        """

        updated = 0

        for row in rows:
            raw_category = row["category_value"] or "default"
            slug = slugify(raw_category)

            # lấy URL đúng trong map
            img_url = CATEGORY_IMAGE.get(slug, CATEGORY_IMAGE["default"])

            img_bytes = download_image_bytes(img_url)

            if img_bytes:
                cursor.execute(update_sql, (img_bytes, row["image_id"]))
                updated += 1
                print(f"[OK] {slug}")
            else:
                print(f"[FAIL] {slug} (URL lỗi)")

            if sleep_between:
                time.sleep(sleep_between)

        conn.commit()
        print(f"Updated {updated} images.")

    except Error as e:
        if conn:
            conn.rollback()
        print("Error:", e)
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main(force_all=False, batch_limit=None, sleep_between=0.15)
