# PROMPT ĐỂ TẠO SƠ ĐỒ HOẠT ĐỘNG HỆ THỐNG GỢI Ý

## PROMPT CHO GPT:

Bạn hãy tạo sơ đồ hoạt động (flowchart) chi tiết cho hệ thống gợi ý sản phẩm thương mại điện tử theo ngữ cảnh với các yêu cầu sau:

### MÔ TẢ HỆ THỐNG:

**Kiến trúc 3 tầng:**
1. **Frontend (React)**: Giao diện người dùng
2. **Backend API (Node.js)**: Xử lý business logic và gọi ML models
3. **ML Models (Python)**: 4 mô hình deep learning (ENCM, LNCM, NeuMF, BMF)

### LUỒNG HOẠT ĐỘNG CHI TIẾT:

**Bước 1: User Request**
- User gửi request lấy gợi ý sản phẩm từ Frontend
- Request gồm: userId, limit (số lượng gợi ý)

**Bước 2: Backend nhận request**
- API Node.js nhận request tại endpoint `/api/recommendations/list` hoặc `/api/recommendations/init`
- Kiểm tra cache trong database (bảng `recommendations`)

**Bước 3: Kiểm tra Cache**
- Nếu có cache và còn hợp lệ → Trả về kết quả từ cache
- Nếu không có cache hoặc cần refresh → Tiến hành tính toán mới

**Bước 4: Thu thập dữ liệu người dùng**
- Lấy lịch sử tương tác: view, cart, purchase
- Lấy thông tin sản phẩm đã tương tác (category, brand)
- Thu thập context: time_of_day, season, device_type
- Lấy ground truth: danh sách sản phẩm đã mua (để đánh giá)

**Bước 5: Gọi Python ML Models (Song song)**
- Chạy 4 mô hình cùng lúc (parallel):
  - **ENCM**: Mô hình có context (time, season, device, category)
  - **LNCM**: Kết hợp linear + neural
  - **NeuMF**: Neural Matrix Factorization
  - **BMF**: Bias Matrix Factorization
- Mỗi mô hình:
  - Encode user_id và item_ids
  - Load model weights từ file .h5
  - Dự đoán score cho tất cả sản phẩm active
  - Trả về top-K sản phẩm có score cao nhất

**Bước 6: Đánh giá và chọn mô hình tốt nhất**
- Tính metrics cho mỗi mô hình:
  - **Precision@10**: Tỷ lệ sản phẩm gợi ý đúng trong top 10
  - **MAP@10**: Mean Average Precision tại top 10
- Sắp xếp mô hình theo MAP@10 (ưu tiên), sau đó Precision@10
- Chọn mô hình có điểm cao nhất

**Bước 7: Fallback (nếu Python không khả dụng)**
- Sử dụng heuristic scoring từ database
- Tính điểm dựa trên:
  - Rating (45%): Điểm sao trung bình
  - Rating count (15%): Số lượng đánh giá
  - Product views (15%): Lượt xem sản phẩm
  - Discount (15%): Phần trăm giảm giá
  - Purchase intent (10%): Ý định mua (high/medium/low)
- Mỗi mô hình có trọng số khác nhau cho các yếu tố trên
- Backfill: Nếu không đủ sản phẩm, thêm sản phẩm cùng category/brand hoặc popular items

**Bước 8: Lưu Cache**
- Lưu kết quả vào bảng `recommendations` (userId, productId, modelName, score)
- Lưu metrics vào bảng `model_runs` để theo dõi

**Bước 9: Trả về kết quả**
- Hydrate thông tin sản phẩm đầy đủ (tên, giá, hình ảnh, ...)
- Trả về JSON cho Frontend
- Frontend hiển thị danh sách sản phẩm được gợi ý

### CÁC YẾU TỐ CẦN THỂ HIỆN TRONG SƠ ĐỒ:

1. **Các thành phần chính**: Frontend, Backend API, Database, Python ML Service
2. **4 mô hình ML**: ENCM, LNCM, NeuMF, BMF (chạy song song)
3. **Dữ liệu đầu vào**: userId, interactions, context, products
4. **Quy trình**: Request → Cache Check → Data Collection → ML Inference → Evaluation → Selection → Cache → Response
5. **Fallback mechanism**: Khi Python không khả dụng
6. **Metrics**: Precision@10, MAP@10
7. **Các bảng database**: recommendations, model_runs, interactions, products

### YÊU CẦU FORMAT:

- Sử dụng **Mermaid flowchart syntax**
- Sơ đồ phải rõ ràng, dễ đọc
- Có các nhánh điều kiện (if/else)
- Thể hiện được luồng song song (parallel processing)
- Có chú thích cho các bước quan trọng
- Màu sắc phân biệt các thành phần (Frontend, Backend, ML, Database)

### OUTPUT MONG MUỐN:

Tạo sơ đồ Mermaid flowchart hoàn chỉnh, có thể copy-paste vào Markdown hoặc Mermaid editor để render.

---

## PROMPT NGẮN GỌN (Nếu cần):

Tạo sơ đồ hoạt động (Mermaid flowchart) cho hệ thống gợi ý sản phẩm e-commerce với kiến trúc:
- Frontend React → Backend Node.js → Python ML Models (ENCM, LNCM, NeuMF, BMF)
- Luồng: User Request → Check Cache → Collect Data → Run 4 ML Models (parallel) → Evaluate (MAP@10, Precision@10) → Select Best Model → Save Cache → Return Results
- Có fallback heuristic scoring nếu Python không khả dụng
- Sử dụng các yếu tố: interactions (view/cart/purchase), ratings, views, discounts, context (time/season/device)







