# TÓM TẮT KIẾN TRÚC VÀ CÁC LỚP THAM GIA HỆ THỐNG GỢI Ý

## 1. KIẾN TRÚC PHẦN MỀM

Hệ thống gợi ý sản phẩm được xây dựng theo **kiến trúc phân tầng (Layered Architecture)** kết hợp nhiều design patterns:

### 1.1. MVC Pattern (Model-View-Controller)
- **Controller**: `recommendationController.js` - Xử lý HTTP requests/responses
- **Model**: Sequelize ORM models (`Product`, `Interaction`, `Comment`, `Recommendation`, `ModelRun`)
- **View**: React.js components - Hiển thị giao diện người dùng

### 1.2. Service Layer Pattern
- **Service**: `recommendationService.js` - Chứa business logic chính
- Tách biệt business logic khỏi controller
- Dễ dàng test và maintain

### 1.3. Bridge Pattern
- **Python Invoker**: `pythonInvoker.js` - Kết nối Node.js và Python
- Cho phép giao tiếp giữa 2 ngôn ngữ khác nhau

### 1.4. Repository Pattern
- **Sequelize ORM**: Đóng vai trò repository
- Tách biệt data access logic khỏi business logic
- Cung cấp abstraction layer cho database

### 1.5. Strategy Pattern
- **4 ML Models**: ENCM, LNCM, NeuMF, BMF
- Mỗi model là một strategy khác nhau
- Chọn strategy tốt nhất dựa trên metrics (MAP@10, Precision@10)

### 1.6. Lazy Loading Pattern
- Python models chỉ load khi cần thiết
- Cache models sau khi load lần đầu
- Giảm memory usage và startup time

---

## 2. CÁC LỚP THAM GIA (CLASSES/COMPONENTS)

### 2.1. PRESENTATION LAYER (Frontend)

#### React Components
- **RecommendationView**: Component hiển thị danh sách gợi ý
- **ProductCard**: Component hiển thị thông tin sản phẩm
- **RecommendationButton**: Button để khởi tạo/lấy gợi ý

**Trách nhiệm**:
- Gửi HTTP requests đến Backend API
- Hiển thị kết quả gợi ý cho người dùng
- Xử lý user interactions (click, view, cart, purchase)

---

### 2.2. ROUTING LAYER (Express.js)

#### Express Router
- **File**: `ecomAPI/src/route/web.js`
- **Routes**:
  - `POST /api/recommend/init` - Khởi tạo gợi ý
  - `GET /api/recommend/list` - Lấy danh sách gợi ý
  - `DELETE /api/recommend/clear` - Xóa cache
  - `GET /internal/recommendations` - Dashboard

**Trách nhiệm**:
- Định tuyến HTTP requests đến controller tương ứng
- Áp dụng middleware (authentication, CORS)

---

### 2.3. MIDDLEWARE LAYER

#### JWT Middleware
- **File**: `ecomAPI/src/middlewares/jwtVerify.js`
- **Functions**:
  - `verifyTokenUser(req, res, next)` - Xác thực user
  - `verifyTokenAdmin(req, res, next)` - Xác thực admin

**Trách nhiệm**:
- Verify JWT token
- Extract user info từ token
- Attach `req.user` với thông tin user đã xác thực
- Reject request nếu token invalid

---

### 2.4. CONTROLLER LAYER (MVC Pattern)

#### RecommendationController
- **File**: `ecomAPI/src/controllers/recommendationController.js`
- **Methods**:
  - `initForCurrentUser(req, res)` - Khởi tạo gợi ý cho user hiện tại
  - `listForCurrentUser(req, res)` - Lấy danh sách gợi ý đã cache
  - `clearForCurrentUser(req, res)` - Xóa cache gợi ý
  - `dashboardPage(req, res)` - Hiển thị dashboard

**Trách nhiệm**:
- Nhận HTTP requests từ route
- Extract parameters (userId, limit) từ request
- Gọi service layer để xử lý business logic
- Format và trả response về client

---

### 2.5. SERVICE LAYER (Business Logic)

#### RecommendationService
- **File**: `ecomAPI/src/services/recommendationService.js`
- **Main Functions**:
  - `initForUser(userId, limit)` - Khởi tạo và tính toán gợi ý mới
  - `getCachedForUser(userId, limit)` - Lấy gợi ý từ cache
  - `clearForUser(userId)` - Xóa cache
  - `computeRecommendationsForUser(userId, limit)` - Tính toán gợi ý (core logic)
  - `buildUserProductFeatures(userId)` - Xây dựng features cho heuristic scoring

**Helper Functions**:
- `deriveContext(timestamp)` - Tính toán context từ timestamp
- `intentFromInteractions(actions)` - Xác định purchase intent
- `scoreSample({rating, rating_count, ...})` - Tính heuristic score
- `pickBestModel()` - Chọn model tốt nhất từ performance CSV

**Trách nhiệm**:
- Orchestration: Điều phối các thành phần khác
- Decision making: Quyết định sử dụng ML models hay heuristic
- Cache management: Quản lý cache recommendations
- Model evaluation: Đánh giá và chọn model tốt nhất

---

### 2.6. ML SERVICE LAYER (Python Bridge)

#### Python Invoker
- **File**: `ecomAPI/src/services/pythonInvoker.js`
- **Function**: `runPythonInference(payload, options)`

**Trách nhiệm**:
- Spawn Python process
- Gửi JSON payload qua stdin
- Nhận kết quả từ stdout
- Handle timeout (default 15s)
- Xử lý errors

#### Python ML Service
- **File**: `models/recommend_api.py`
- **Main Functions**:
  - `handle_request(req)` - Xử lý inference request
  - `load_resources()` - Load encoders và context mappings từ DB
  - `_build_model(model_name, configs)` - Build và load model weights
  - `_encode_user(user_id)` - Encode user ID thành index
  - `_candidate_indices_from_db()` - Lấy danh sách candidate items
  - `_context_to_features(context)` - Convert context thành features array
  - `_decode_item_index(idx)` - Decode item index về product ID

**Trách nhiệm**:
- Load encoders (user, item, context) từ database
- Load model weights từ file .h5
- Encode user và items
- Predict scores cho tất cả candidates
- Top-K selection
- Decode kết quả về product IDs

#### Model Classes (Keras/TensorFlow)
- **File**: `models/model_classes.py`
- **Classes**:
  - `ENCM` - Embedding-based Neural Context Model (context-aware)
  - `LNCM` - Linear Neural Combination Model
  - `NeuMF` - Neural Matrix Factorization
  - `BMF` - Bias Matrix Factorization

**Trách nhiệm**:
- Định nghĩa architecture của neural networks
- Forward pass để tính prediction scores
- Load/save weights

---

### 2.7. DATA ACCESS LAYER (Repository Pattern)

#### Sequelize ORM Models
- **Files**: `ecomAPI/src/models/*.js`
- **Models**:
  - `User` - Người dùng
  - `Product` - Sản phẩm
  - `ProductDetail` - Chi tiết sản phẩm (giá, size, ...)
  - `Interaction` - Tương tác (view, cart, purchase)
  - `Comment` - Đánh giá/comment
  - `Recommendation` - Cache gợi ý
  - `ModelRun` - Metrics và recommendations của mỗi model

**Trách nhiệm**:
- Abstract database operations
- Provide query interface
- Handle relationships (associations)
- Data validation

---

### 2.8. DATABASE LAYER

#### MySQL Database
- **Tables**:
  - `users` - Thông tin người dùng
  - `products` - Sản phẩm
  - `product_details` - Chi tiết sản phẩm
  - `interactions` - Lịch sử tương tác
  - `comments` - Đánh giá
  - `recommendations` - Cache gợi ý
  - `model_runs` - Metrics của models
  - `rec_user_encoder` - User encoder mapping
  - `rec_item_encoder` - Item encoder mapping
  - `rec_context_mapping` - Context feature mappings
  - `rec_context_meta` - Context metadata

**Trách nhiệm**:
- Lưu trữ dữ liệu persistent
- Cung cấp data cho ML models
- Cache recommendations
- Store metrics và evaluation results

---

## 3. LUỒNG TƯƠNG TÁC GIỮA CÁC LỚP

### 3.1. Luồng khởi tạo gợi ý

```
User (Frontend)
  ↓ HTTP POST
Express Router
  ↓ verifyTokenUser()
JWT Middleware
  ↓ next()
RecommendationController.initForCurrentUser()
  ↓ initForUser()
RecommendationService
  ↓ computeRecommendationsForUser()
  ├─→ Python Invoker
  │     ↓ spawn Python
  │   Python ML Service
  │     ↓ load_resources()
  │   Database (encoders)
  │     ↓ _build_model()
  │   Model Classes (ENCM/LNCM/NeuMF/BMF)
  │     ↓ predict()
  │   Top-K selection
  │     ↑
  └─→ Evaluate & Select best model
      ↓ Save cache
  Sequelize Models
    ↓ INSERT
  MySQL Database
```

### 3.2. Luồng lấy danh sách gợi ý

```
User (Frontend)
  ↓ HTTP GET
Express Router
  ↓ verifyTokenUser()
JWT Middleware
  ↓ next()
RecommendationController.listForCurrentUser()
  ↓ getCachedForUser()
RecommendationService
  ↓ Recommendation.findAll()
Sequelize Models
  ↓ SELECT
MySQL Database (recommendations table)
  ↑
  ↓ Hydrate products
Product.findOne() (loop)
  ↓
Response với product details
```

---

## 4. DESIGN PATTERNS ĐƯỢC SỬ DỤNG

### 4.1. MVC Pattern
- **Model**: Sequelize ORM models
- **View**: React components
- **Controller**: recommendationController.js

### 4.2. Service Layer Pattern
- Tách business logic khỏi controller
- RecommendationService chứa toàn bộ logic xử lý

### 4.3. Bridge Pattern
- Python Invoker kết nối Node.js và Python
- Cho phép giao tiếp giữa 2 ngôn ngữ

### 4.4. Repository Pattern
- Sequelize ORM đóng vai trò repository
- Tách biệt data access logic

### 4.5. Strategy Pattern
- 4 ML models là các strategies khác nhau
- Chọn strategy tốt nhất dựa trên metrics

### 4.6. Singleton Pattern
- Python service sử dụng `_cached` dictionary
- Models chỉ load một lần và cache lại

### 4.7. Factory Pattern
- `_build_model()` tạo model instances
- Dựa trên model_name để tạo model tương ứng

### 4.8. Template Method Pattern
- `computeRecommendationsForUser()` định nghĩa template
- Các bước cụ thể được implement trong helper functions

---

## 5. TÓM TẮT

Hệ thống gợi ý sản phẩm sử dụng **kiến trúc phân tầng** với các pattern:

1. **Presentation Layer**: React.js components
2. **Routing Layer**: Express Router
3. **Middleware Layer**: JWT Authentication
4. **Controller Layer**: MVC Controllers
5. **Service Layer**: Business Logic
6. **ML Service Layer**: Python ML Models
7. **Data Access Layer**: Sequelize ORM
8. **Database Layer**: MySQL

**Đặc điểm**:
- Tách biệt rõ ràng giữa các layers
- Dễ dàng test và maintain
- Hỗ trợ multiple strategies (ML models)
- Có cơ chế fallback (heuristic scoring)
- Tối ưu hiệu năng (parallel processing, caching, lazy loading)






