# PHÂN TÍCH LUỒNG XỬ LÝ HỆ THỐNG GỢI Ý SẢN PHẨM

## 1. TỔNG QUAN KIẾN TRÚC

Hệ thống gợi ý sản phẩm được xây dựng theo kiến trúc **MVC (Model-View-Controller)** kết hợp với **Service Layer** và **ML Service** độc lập. Hệ thống sử dụng **hybrid approach** kết hợp giữa Machine Learning models và heuristic scoring để đảm bảo tính ổn định và hiệu quả.

### Các thành phần chính:
- **Frontend**: React.js - Giao diện người dùng
- **Backend API**: Node.js (Express) - Xử lý request/response
- **ML Service**: Python (TensorFlow/Keras) - Inference các mô hình ML
- **Database**: MySQL - Lưu trữ dữ liệu và cache

---

## 2. LUỒNG XỬ LÝ CHÍNH

### 2.1. Luồng khởi tạo gợi ý (Initialize Recommendations)

**Endpoint**: `POST /api/recommend/init?limit=10`

#### Bước 1: Frontend gửi request
```javascript
// React Component
fetch('/api/recommend/init?limit=10', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`
  }
})
```

#### Bước 2: Route Layer nhận request
```javascript
// ecomAPI/src/route/web.js
router.post('/api/recommend/init', 
  middlewareControllers.verifyTokenUser, 
  recommendationController.initForCurrentUser
)
```

#### Bước 3: Middleware xác thực JWT
- Kiểm tra token hợp lệ
- Lấy thông tin user từ token
- Gắn `req.user` với thông tin user đã xác thực

#### Bước 4: Controller xử lý request
```javascript
// recommendationController.js
let initForCurrentUser = async (req, res) => {
  const userId = req.user.id;  // Lấy từ JWT token
  const limit = +(req.query.limit || 10);
  await recommendationService.initForUser(userId, limit);
  return res.status(200).json({ errCode: 0, message: 'initialized' });
}
```

#### Bước 5: Service Layer - Xóa cache cũ
```javascript
// recommendationService.js - initForUser()
await ensureTables();  // Đảm bảo bảng recommendations và model_runs tồn tại
await db.Recommendation.destroy({ where: { userId } });  // Xóa cache cũ
await db.ModelRun.destroy({ where: { userId } });
```

#### Bước 6: Service Layer - Tính toán gợi ý mới
```javascript
const { bestModel, top, modelRuns } = await computeRecommendationsForUser(userId, limit);
```

**Chi tiết hàm `computeRecommendationsForUser()`:**

##### 6.1. Thử gọi Python ML Service (Ưu tiên)
```javascript
if (pythonInvoker) {
  // Chuẩn bị context
  const nowCtx = deriveContext(new Date());  // Thời gian hiện tại
  const lastInter = await db.Interaction.findOne({ 
    where: { userId }, 
    order: [['timestamp','DESC']] 
  });
  
  const ctxPayload = {
    time_of_day: nowCtx.time_of_day,      // morning, afternoon, evening, night
    season: nowCtx.season,                // spring, summer, autumn, winter
    device_type: lastInter?.device_type || 'Mobile',
    category: null
  };
  
  // Chạy 4 models song song (parallel)
  const modelNames = ['ENCM','LNCM','NeuMF','BMF'];
  const parallel = await Promise.all(
    modelNames.map(async (name) => {
      const resp = await pythonInvoker.runPythonInference({
        user_id: userId,
        limit: k,
        model: name,
        context: ctxPayload
      });
      return { name, resp };
    })
  );
}
```

##### 6.2. Python Invoker gọi ML Service
```javascript
// pythonInvoker.js
function runPythonInference(payload, { timeoutMs = 15000 } = {}) {
  const ps = spawn(PYTHON, ['-u', SCRIPT], { cwd: ROOT });
  ps.stdin.write(JSON.stringify(payload));
  // Đọc kết quả từ stdout
}
```

##### 6.3. Python ML Service xử lý
```python
# recommend_api.py
def handle_request(req):
    # 1. Load encoders và models (lazy loading, cache)
    load_resources()
    
    # 2. Encode user_id thành user_idx
    user_idx = _encode_user(user_id, enc_user)
    
    # 3. Lấy danh sách candidate items (chỉ sản phẩm active - statusId='S1')
    item_indices = _candidate_indices_from_db()
    
    # 4. Build model (load từ disk nếu chưa có trong cache)
    m = _build_model(model_name, _cached['configs'])
    
    # 5. Predict scores cho tất cả candidates
    if model_name == 'ENCM':
        # ENCM cần context features
        ctx_feats = _context_to_features(context, context_info)
        scores = m.predict([user_ids, item_indices, ctx_feats])
    else:
        # LNCM, NeuMF, BMF chỉ cần user và item
        scores = m.predict([user_ids, item_indices])
    
    # 6. Top-K selection
    top_idx = np.argsort(-scores)[:limit]
    
    # 7. Decode item_idx về productId
    items = []
    for pos in top_idx:
        model_idx = int(item_indices[pos])
        items.append({
            'productId': _decode_item_index(model_idx),
            'score': float(scores[pos])
        })
    
    return {"ok": True, "model": model_name, "items": items}
```

##### 6.4. Đánh giá và chọn model tốt nhất
```javascript
// Tính Precision@10 và MAP@10 cho mỗi model
const gtPurch = await db.Interaction.findAll({ 
  where: { userId, actionCode: 'purchase' } 
});
const gtSet = new Set(gtPurch.map(r => r.productId));

for (const { name, resp } of parallel) {
  let hits = 0;
  let sumPrec = 0;
  for (let i = 0; i < Math.min(k, recs.length); i++) {
    const rel = gtSet.has(recs[i].productId) ? 1 : 0;
    if (rel) { 
      hits += 1; 
      sumPrec += hits / (i + 1); 
    }
  }
  const precision10 = hits / k;
  const map10 = sumPrec / denom;
  
  modelRuns.push({ 
    modelName: name, 
    recommendations: recs, 
    metrics: { precision10, map10 } 
  });
}

// Sắp xếp theo MAP@10, sau đó Precision@10
modelRuns.sort((a,b) => 
  (b.metrics.map10 - a.metrics.map10) || 
  (b.metrics.precision10 - a.metrics.precision10)
);

const best = modelRuns[0];  // Model tốt nhất
```

##### 6.5. Fallback về Heuristic Scoring (nếu Python fail)
```javascript
// Nếu Python service không khả dụng hoặc lỗi
// Sử dụng heuristic scoring dựa trên:
// - Rating (r): 0-1 normalized
// - Rating count (rc): số lượng đánh giá
// - Product views (pv): lượt xem
// - Discount percentage (disc): % giảm giá
// - Purchase intent (iw): từ interaction history

const variants = [
  { name: 'bmf',   weights: { r:0.20, rc:0.10, pv:0.20, disc:0.35, iw:0.15 } },
  { name: 'encm',  weights: { r:0.30, rc:0.10, pv:0.15, disc:0.10, iw:0.35 } },
  { name: 'lncm',  weights: { r:0.55, rc:0.20, pv:0.10, disc:0.10, iw:0.05 } },
  { name: 'neumf', weights: { r:0.40, rc:0.15, pv:0.15, disc:0.10, iw:0.20 } },
];

// Tính điểm cho mỗi variant
for (const v of variants) {
  const rescored = rescore(base, v.weights);
  const topV = await backfillTop(rescored);
  modelRuns.push({ modelName: v.name, recommendations: topV, metrics: {...} });
}

// Chọn best model dựa trên intentHigh, aligned, avgScore
modelRuns.sort((a,b) => 
  (b.metrics.intentHigh - a.metrics.intentHigh) || 
  (b.metrics.aligned - a.metrics.aligned) || 
  (b.metrics.avgScore - a.metrics.avgScore)
);
```

#### Bước 7: Lưu cache vào database
```javascript
// Lưu recommendations
for (const item of top) {
  await db.Recommendation.create({ 
    userId, 
    productId: item.productId, 
    modelName: bestModel, 
    score: item.score || 0 
  });
}

// Lưu metrics của tất cả models
for (const r of modelRuns) {
  await db.ModelRun.create({ 
    userId, 
    modelName: r.modelName, 
    metricsJson: JSON.stringify(r.metrics),
    recommendationsJson: JSON.stringify(r.recommendations)
  });
}
```

#### Bước 8: Trả response về Frontend
```javascript
return res.status(200).json({ 
  errCode: 0, 
  message: 'initialized' 
});
```

---

### 2.2. Luồng lấy danh sách gợi ý (Get Recommendations)

**Endpoint**: `GET /api/recommend/list?limit=10`

#### Bước 1-4: Tương tự như luồng khởi tạo
- Frontend gửi request
- Route nhận request
- Middleware xác thực
- Controller xử lý

#### Bước 5: Service Layer - Lấy từ cache
```javascript
// recommendationController.js
let listForCurrentUser = async (req, res) => {
  const userId = req.user.id;
  const limit = +(req.query.limit || 10);
  const recs = await recommendationService.getCachedForUser(userId, limit);
  
  // Hydrate products (lấy thông tin chi tiết sản phẩm)
  const result = [];
  for (const r of recs) {
    const product = await db.Product.findOne({ where: { id: r.productId } });
    if (product) {
      result.push({ 
        product, 
        score: r.score, 
        modelName: r.modelName 
      });
    }
  }
  
  return res.status(200).json({ errCode: 0, data: result });
}
```

```javascript
// recommendationService.js
async function getCachedForUser(userId, limit=10) {
  await ensureTables();
  const rows = await db.Recommendation.findAll({ 
    where: { userId }, 
    order: [['score', 'DESC']], 
    limit, 
    raw: true 
  });
  return rows;
}
```

#### Bước 6: Trả response về Frontend
```json
{
  "errCode": 0,
  "data": [
    {
      "product": {
        "id": 1,
        "name": "Sản phẩm A",
        "view": 100,
        ...
      },
      "score": 0.95,
      "modelName": "ENCM"
    },
    ...
  ]
}
```

---

## 3. CHI TIẾT CÁC THÀNH PHẦN XỬ LÝ

### 3.1. Xây dựng User-Product Features

Hàm `buildUserProductFeatures()` thu thập và tính toán các đặc trưng cho mỗi sản phẩm:

```javascript
async function buildUserProductFeatures(userId) {
  // 1. Lấy tất cả sản phẩm active (statusId='S1')
  const products = await db.Product.findAll({ where: { statusId: 'S1' } });
  
  // 2. Lấy ProductDetails (giá, discount)
  const details = await db.ProductDetail.findAll({ 
    where: { productId: { [Op.in]: productIds } } 
  });
  
  // 3. Lấy Comments (rating, rating_count)
  const comments = await db.Comment.findAll({ 
    where: { productId: { [Op.in]: productIds } } 
  });
  
  // 4. Lấy Interactions của user (view, cart, purchase)
  const interactions = await db.Interaction.findAll({ 
    where: { userId, productId: { [Op.in]: productIds } },
    order: [['timestamp','DESC']]
  });
  
  // 5. Tính toán Context
  const nowCtx = deriveContext(new Date());
  // - time_of_day: morning, afternoon, evening, night
  // - season: spring, summer, autumn, winter
  // - day_of_week: 1-7
  // - is_weekend: 0/1
  
  // 6. Tính toán Purchase Intent
  const purchase_intent = intentFromInteractions(actions);
  // - high: có purchase
  // - medium: có cart
  // - low: chỉ có view
  
  // 7. Tính điểm heuristic
  const score = scoreSample({ 
    rating, rating_count, product_views, 
    discount_percentage, intent_weight 
  });
  
  return results;  // Sorted by score DESC
}
```

### 3.2. Derive Context từ Timestamp

```javascript
function deriveContext(ts) {
  const m = moment(ts);
  const hour = m.hour();
  const month = m.month() + 1;
  const dow = m.isoWeekday();
  const isWeekend = dow >= 6 ? 1 : 0;
  const season = month <= 2 || month === 12 ? "winter" 
                : month <= 5 ? "spring" 
                : month <= 8 ? "summer" 
                : "autumn";
  const time_of_day = hour < 6 ? "night" 
                     : hour < 12 ? "morning" 
                     : hour < 18 ? "afternoon" 
                     : "evening";
  return { 
    hour, month, day_of_week: dow, is_weekend: isWeekend, 
    season, time_of_day, day_name: m.format("dddd"), 
    date: m.format("YYYY-MM-DD") 
  };
}
```

### 3.3. Heuristic Scoring

```javascript
function scoreSample({ rating, rating_count, product_views, discount_percentage, intent_weight }) {
  // Normalize các giá trị về 0-1
  const r = Math.max(0, Math.min(1, (rating - 1) / 4));  // 1-5 -> 0-1
  const rc = Math.min(1, (rating_count || 0) / 50);      // Max 50 reviews
  const pv = Math.min(1, (product_views || 0) / 50);     // Max 50 views
  const disc = Math.min(1, Math.max(0, (discount_percentage || 0) / 70));  // Max 70%
  const iw = intent_weight;  // 0.0, 0.5, 1.0
  
  // Weighted sum (default weights)
  return 0.45 * r + 0.15 * rc + 0.15 * pv + 0.15 * disc + 0.10 * iw;
}
```

### 3.4. Python ML Service - Load Resources

```python
def load_resources():
    # 1. Load user encoder từ database
    # rec_user_encoder: original_id -> idx
    user_map = {}
    with conn.cursor() as cur:
        cur.execute("SELECT original_id, idx FROM rec_user_encoder")
        for oid, idx in cur.fetchall():
            user_map[str(oid)] = int(idx)
    
    # 2. Load item encoder từ database
    # rec_item_encoder: original_id -> idx
    item_map = {}
    with conn.cursor() as cur:
        cur.execute("SELECT original_id, idx FROM rec_item_encoder")
        for oid, idx in cur.fetchall():
            item_map[str(oid)] = int(idx)
    
    # 3. Lấy active candidates (statusId='S1')
    idx_to_pid = {}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT e.idx, e.original_id
            FROM rec_item_encoder e
            JOIN products p ON p.id = e.original_id
            WHERE p.statusId = 'S1'
        """)
        for idx, oid in cur.fetchall():
            idx_to_pid[int(idx)] = int(oid)
    
    # 4. Load context mappings (cho ENCM)
    reverse_mappings = {}
    with conn.cursor() as cur:
        cur.execute("SELECT feature_name, original_value, idx FROM rec_context_mapping")
        for feat, val, idx in cur.fetchall():
            reverse_mappings.setdefault(feat, {})[str(val)] = int(idx)
    
    # Cache vào _cached
    _cached['encoders'] = {'user': user_map, 'item': item_map}
    _cached['idx_to_pid'] = idx_to_pid
    _cached['context_info'] = {'reverse_mappings': reverse_mappings, ...}
```

### 3.5. Python ML Service - Model Inference

```python
def _build_model(model_name, configs):
    # Lazy loading: chỉ load khi cần
    if model_name in _cached['models']:
        return _cached['models'][model_name]
    
    # Load config từ JSON
    cfg_path = os.path.join(MODELS_DIR, f"{model_name.lower()}_config.json")
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    # Build model theo architecture
    if model_name == 'ENCM':
        m = ENCM(
            n_users=cfg['n_users'],
            n_items=cfg['n_items'],
            n_contexts=cfg['n_contexts'],
            embedding_dim=cfg['embedding_dim'],
            context_dim=cfg['context_dim'],
            hidden_dims=cfg['hidden_dims']
        )
        # Build và load weights
        m([dummy_u, dummy_i, dummy_ctx])
        m.load_weights('encm.weights.h5')
    elif model_name == 'LNCM':
        m = LNCM(...)
        m.load_weights('lncm.weights.h5')
    # ... tương tự cho NeuMF, BMF
    
    _cached['models'][model_name] = m
    return m
```

---

## 4. CƠ CHẾ FALLBACK VÀ XỬ LÝ LỖI

### 4.1. Fallback Strategy

Hệ thống sử dụng **multi-tier fallback**:

1. **Tier 1**: Python ML Service (4 models song song)
   - Nếu thành công → Chọn model tốt nhất (MAP@10 cao nhất)
   - Nếu thất bại → Chuyển sang Tier 2

2. **Tier 2**: Heuristic Scoring (4 variants)
   - Tính điểm dựa trên rating, views, discount, intent
   - Chọn variant tốt nhất (intentHigh, aligned, avgScore)

3. **Tier 3**: Backfill với sản phẩm cùng category/brand
   - Nếu không đủ 10 items → Lấy thêm từ category/brand user đã tương tác

4. **Tier 4**: Popular items
   - Nếu vẫn không đủ → Lấy sản phẩm phổ biến nhất (view cao)

### 4.2. Xử lý lỗi

```javascript
// Python Invoker timeout
const timer = setTimeout(() => {
  ps.kill('SIGKILL');
  resolve({ ok: false, error: 'timeout' });
}, timeoutMs);  // Default 15s

// Python service error
try {
  const resp = await pythonInvoker.runPythonInference({...});
} catch (e) {
  return { name, resp: { ok: false, error: e?.message } };
}

// Fallback tự động
if (!modelRuns.length || !best) {
  // Sử dụng heuristic scoring
}
```

---

## 5. CƠ CHẾ CACHE

### 5.1. Cache Structure

**Bảng `recommendations`**:
- `userId`: ID người dùng
- `productId`: ID sản phẩm được gợi ý
- `modelName`: Tên model tạo ra gợi ý này
- `score`: Điểm số dự đoán
- `createdAt`, `updatedAt`: Timestamp

**Bảng `model_runs`**:
- `userId`: ID người dùng
- `modelName`: Tên model
- `metricsJson`: JSON chứa Precision@10, MAP@10, ...
- `recommendationsJson`: JSON chứa danh sách recommendations
- `createdAt`, `updatedAt`: Timestamp

### 5.2. Cache Lifecycle

1. **Khởi tạo**: `POST /api/recommend/init`
   - Xóa cache cũ
   - Tính toán mới
   - Lưu vào database

2. **Đọc**: `GET /api/recommend/list`
   - Đọc từ cache (không tính toán lại)

3. **Xóa**: `DELETE /api/recommend/clear`
   - Xóa cache khi user logout hoặc cần refresh

---

## 6. TỐI ƯU HÓA HIỆU NĂNG

### 6.1. Parallel Processing

```javascript
// Chạy 4 models song song thay vì tuần tự
const parallel = await Promise.all(
  modelNames.map(async (name) => {
    return await pythonInvoker.runPythonInference({...});
  })
);
```

### 6.2. Lazy Loading

```python
# Python service chỉ load model khi cần
if model_name in _cached['models']:
    return _cached['models'][model_name]  # Đã cache
else:
    m = _build_model(model_name, ...)     # Load lần đầu
    _cached['models'][model_name] = m     # Cache lại
```

### 6.3. Database Indexing

- Index trên `recommendations(userId, score)` để query nhanh
- Index trên `interactions(userId, productId, timestamp)` để filter nhanh

### 6.4. Candidate Filtering

```python
# Chỉ predict cho active items (statusId='S1')
# Giảm số lượng candidates từ hàng nghìn xuống vài trăm
idx_to_pid = {}  # Chỉ chứa active items
```

---

## 7. ĐIỂM ĐẶC BIỆT CỦA HỆ THỐNG

### 7.1. Context-Aware Recommendations

- **ENCM model** sử dụng context features:
  - `time_of_day`: morning, afternoon, evening, night
  - `season`: spring, summer, autumn, winter
  - `device_type`: Mobile, Desktop, Tablet
  - `category`: Danh mục sản phẩm

### 7.2. Multi-Model Ensemble

- Hệ thống chạy 4 models đồng thời:
  - **ENCM**: Embedding-based Neural Context Model
  - **LNCM**: Linear Neural Combination Model
  - **NeuMF**: Neural Matrix Factorization
  - **BMF**: Bias Matrix Factorization

- Tự động chọn model tốt nhất dựa trên metrics (MAP@10, Precision@10)

### 7.3. Hybrid Approach

- Kết hợp ML models và heuristic rules
- Đảm bảo luôn có recommendations ngay cả khi ML service fail

### 7.4. Real-time Evaluation

- Tính Precision@10 và MAP@10 ngay trong quá trình inference
- So sánh với ground truth (purchase history) để chọn model tốt nhất

---

## 8. KẾT LUẬN

Hệ thống gợi ý sản phẩm được thiết kế với các đặc điểm:

1. **Kiến trúc phân tầng rõ ràng**: MVC + Service Layer + ML Service
2. **Tính ổn định cao**: Multi-tier fallback mechanism
3. **Hiệu năng tốt**: Parallel processing, lazy loading, caching
4. **Linh hoạt**: Hỗ trợ nhiều models, context-aware, real-time evaluation
5. **Dễ mở rộng**: Có thể thêm models mới, điều chỉnh weights, thêm context features

Luồng xử lý được tối ưu để đảm bảo:
- **Response time nhanh**: Cache, parallel processing
- **Độ chính xác cao**: Multi-model ensemble, real-time evaluation
- **Tính sẵn sàng**: Fallback mechanism đảm bảo luôn có kết quả






