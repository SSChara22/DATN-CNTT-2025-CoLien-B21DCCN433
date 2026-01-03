# Biểu đồ lớp phân tích - Hệ thống E-commerce với gợi ý theo ngữ cảnh
## (Analysis Class Diagram - Domain Model)

```mermaid
classDiagram
    %% Core Domain Entities
    class User {
        +Integer id {PK}
        +String email {unique, required}
        +String password {required}
        +String firstName
        +String lastName
        +String phonenumber
        +String address
        +Date dob
        +Boolean isActiveEmail
        +login(email, password) Boolean
        +updateProfile(data) void
        +getRecommendations(limit) Product[]
        +trackInteraction(productId, action) void
    }
    
    class Product {
        +Integer id {PK}
        +String name {required}
        +Text contentHTML
        +Integer view
        +String madeby
        +String material
        +getDetails() ProductDetail[]
        +getImages() ProductImage[]
        +getRating() Float
        +getComments() Comment[]
        +incrementView() void
    }
    
    class Order {
        +Integer id {PK}
        +Float totalPrice {required}
        +Date orderDate
        +String deliveryAddress
        +String paymentMethod
        +String paymentStatus
        +calculateTotal() Float
        +addProduct(productId, quantity) void
        +updateStatus(status) void
        +processPayment() Boolean
    }
    
    class Recommendation {
        +Integer id {PK}
        +Float score {required}
        +String modelName {required}
        +Text details
        +Date createdAt
        +getProduct() Product
        +getUser() User
        +isValid() Boolean
    }

    %% Product Domain
    class ProductDetail {
        +Integer id {PK}
        +Float originalPrice {required}
        +Float discountPrice
        +Integer quantity
        +Text description
        +getDiscountPercentage() Float
        +isAvailable() Boolean
    }
    
    class ProductImage {
        +Integer id {PK}
        +String imageUrl {required}
        +Boolean isPrimary
        +Integer displayOrder
    }
    
    class ProductDetailSize {
        +Integer id {PK}
        +String sizeName {required}
        +Integer quantity
        +isInStock() Boolean
    }
    
    class Category {
        +String code {PK}
        +String value {required}
        +String type {required}
        +getProducts() Product[]
    }
    
    class Brand {
        +String code {PK}
        +String value {required}
        +String type {required}
        +getProducts() Product[]
    }

    %% Order Domain
    class OrderDetail {
        +Integer id {PK}
        +Integer quantity {required}
        +Float unitPrice {required}
        +Float subtotal
        +calculateSubtotal() Float
    }
    
    class OrderProduct {
        +Integer id {PK}
        +Integer quantity {required}
        +Float price {required}
        +calculateTotal() Float
    }
    
    class AddressUser {
        +Integer id {PK}
        +String fullName {required}
        +String phone {required}
        +String address {required}
        +Boolean isDefault
        +setAsDefault() void
    }
    
    class TypeShip {
        +Integer id {PK}
        +String name {required}
        +Float price {required}
        +Text description
        +calculateShippingFee(order) Float
    }

    %% Interaction & Recommendation Domain
    class Interaction {
        +Integer id {PK}
        +String actionCode {required}
        +String device_type
        +Date timestamp {required}
        +getActionType() String
        +isPurchase() Boolean
        +isView() Boolean
        +isCart() Boolean
        +getPurchaseIntent() String
    }
    
    class Comment {
        +Integer id {PK}
        +Integer star {1-5}
        +Text content
        +Date createdAt
        +calculateAverageRating() Float
        +reply(content) Comment
    }
    
    class ModelRun {
        +Integer id {PK}
        +String modelName {required}
        +Text metricsJson
        +Text recommendationsJson
        +Date createdAt
        +getMetrics() Object
        +getPrecisionAt10() Float
        +getMAPAt10() Float
    }
    
    class Context {
        +String time_of_day
        +String season
        +String device_type
        +String category
        +deriveFromTimestamp(timestamp) Context
        +encode() Array~Integer~
    }

    %% Shopping Domain
    class ShopCart {
        +Integer id {PK}
        +Integer quantity {required}
        +Date createdAt
        +addItem(productId, quantity) void
        +removeItem(productId) void
        +updateQuantity(productId, quantity) void
        +calculateTotal() Float
        +checkout() Order
    }
    
    class Voucher {
        +Integer id {PK}
        +String code {unique, required}
        +String discountType
        +Float discountValue {required}
        +Float minPurchase
        +Float maxDiscount
        +Date startDate
        +Date endDate
        +Integer quantity
        +isValid() Boolean
        +isApplicable(order) Boolean
        +calculateDiscount(order) Float
    }
    
    class VoucherUsed {
        +Integer id {PK}
        +Date usedDate
        +markAsUsed() void
    }

    %% Supply Chain Domain
    class Receipt {
        +Integer id {PK}
        +Date receiptDate {required}
        +Float totalAmount {required}
        +String status
        +addProduct(productId, quantity, price) void
        +calculateTotal() Float
        +confirm() void
    }
    
    class ReceiptDetail {
        +Integer id {PK}
        +Integer quantity {required}
        +Float unitPrice {required}
        +Float subtotal
        +calculateSubtotal() Float
    }
    
    class Supplier {
        +Integer id {PK}
        +String name {required}
        +String contact
        +String address
        +getReceipts() Receipt[]
    }

    %% Content Domain
    class Blog {
        +Integer id {PK}
        +String title {required}
        +Text contentHTML
        +Text contentMarkdown
        +Blob image
        +Boolean isFeature
        +getComments() Comment[]
    }
    
    class Banner {
        +Integer id {PK}
        +Blob image {required}
        +String link
        +Integer displayOrder
        +Boolean isActive
        +activate() void
        +deactivate() void
    }

    %% Messaging Domain
    class RoomMessage {
        +Integer id {PK}
        +String roomName
        +Date createdAt
        +getMessages() Message[]
    }
    
    class Message {
        +Integer id {PK}
        +Text content {required}
        +Integer senderId {required}
        +Date createdAt
        +send() void
        +reply(content) Message
    }

    %% ML Recommendation Domain
    class RecommendationModel {
        <<abstract>>
        +Integer n_users
        +Integer n_items
        +Integer embedding_dim
        +predict(user_id, item_ids, context) Float[]
        +evaluate(test_data) Metrics
    }
    
    class ENCM {
        +List~Integer~ n_contexts
        +Integer context_dim
        +usesContext() Boolean
        +predict(user_id, item_ids, context_features) Float[]
    }
    
    class LNCM {
        +List~Integer~ hidden_dims
        +Float alpha
        +combineLinearAndNeural() Float
        +predict(user_id, item_ids) Float[]
    }
    
    class NeuMF {
        +List~Integer~ hidden_dims
        +combineGMFAndMLP() Float
        +predict(user_id, item_ids) Float[]
    }
    
    class BMF {
        +Float user_bias
        +Float item_bias
        +Float global_bias
        +predict(user_id, item_ids) Float[]
    }
    
    class RecommendationEngine {
        -Map~String,RecommendationModel~ models
        -Map~String,Encoder~ encoders
        +generateRecommendations(userId, limit, context) Recommendation[]
        +evaluateModels(userId) ModelRun[]
        +selectBestModel(modelRuns) String
        +loadModels() void
        +encodeUser(userId) Integer
        +encodeItems(productIds) Integer[]
    }
    
    class Encoder {
        +Integer original_id
        +Integer idx
        +encode(original_id) Integer
        +decode(idx) Integer
    }

    %% Relationships
    User "1" --> "0..*" Order : places
    User "1" --> "0..*" ShopCart : has
    User "1" --> "0..*" Recommendation : receives
    User "1" --> "0..*" Interaction : performs
    User "1" --> "0..*" Comment : writes
    User "1" --> "0..*" AddressUser : has
    User "1" --> "0..*" ModelRun : has
    User "1" --> "0..*" VoucherUsed : uses
    User "1" --> "0..*" Message : sends

    Product "1" --> "0..*" ProductDetail : has
    Product "1" --> "0..*" ProductImage : has
    Product "1" --> "0..*" ProductDetailSize : has
    Product "1" --> "0..*" Recommendation : recommended_in
    Product "1" --> "0..*" Interaction : interacted_with
    Product "1" --> "0..*" Comment : has
    Product "1" --> "0..*" ShopCart : in
    Product "1" --> "0..*" OrderDetail : in
    Product "1" --> "1" Category : belongs_to
    Product "1" --> "1" Brand : belongs_to

    Order "1" --> "0..*" OrderDetail : contains
    Order "1" --> "0..*" OrderProduct : contains
    Order "1" --> "1" User : belongs_to
    Order "1" --> "0..1" AddressUser : ships_to
    Order "1" --> "0..1" TypeShip : uses
    Order "1" --> "0..1" Voucher : applies

    Recommendation "1" --> "1" User : for
    Recommendation "1" --> "1" Product : recommends
    Recommendation "1" --> "0..1" ModelRun : generated_by

    Interaction "1" --> "1" User : performed_by
    Interaction "1" --> "1" Product : on

    Comment "1" --> "1" User : written_by
    Comment "1" --> "1" Product : on
    Comment "1" --> "0..1" Comment : replies_to

    ModelRun "1" --> "1" User : for
    ModelRun "1" --> "0..*" Recommendation : generates

    ShopCart "1" --> "1" User : belongs_to
    ShopCart "1" --> "1" Product : contains

    Voucher "1" --> "0..*" VoucherUsed : used_in
    VoucherUsed "1" --> "1" User : used_by
    VoucherUsed "1" --> "1" Voucher : uses

    Receipt "1" --> "0..*" ReceiptDetail : contains
    Receipt "1" --> "1" Supplier : from

    Blog "1" --> "0..*" Comment : has

    Message "1" --> "1" RoomMessage : in
    RoomMessage "1" --> "0..*" Message : contains

    RecommendationModel <|-- ENCM
    RecommendationModel <|-- LNCM
    RecommendationModel <|-- NeuMF
    RecommendationModel <|-- BMF

    RecommendationEngine "1" --> "1..*" RecommendationModel : uses
    RecommendationEngine "1" --> "1..*" Encoder : uses
    RecommendationEngine --> User : generates_for
    RecommendationEngine --> Product : recommends
    RecommendationEngine --> Context : uses
```

## Business Rules (Quy tắc nghiệp vụ)

### User
- User phải verify email trước khi có quyền truy cập đầy đủ
- Phân quyền dựa trên role (Admin/User)
- Password phải được hash

### Product
- Product phải có ít nhất một ProductDetail
- Chỉ sản phẩm active (statusId='S1') mới được gợi ý
- View count tăng mỗi khi có lượt xem

### Order
- Order total = tổng các OrderDetail
- Trạng thái: Pending → Confirmed → Shipping → Delivered
- Payment phải hoàn tất trước khi shipping

### Recommendation
- Recommendations được cache để tối ưu performance
- Score range: 0.0 - 1.0
- Top-K selection dựa trên score
- Model selection dựa trên MAP@10

### Interaction
- Actions: view, cart, purchase
- Purchase intent: high (purchase) > medium (cart) > low (view)
- Device type được track cho context

### RecommendationEngine
- 4 models chạy song song
- Best model được chọn bởi MAP@10
- Fallback về heuristic nếu ML fails
- Context-aware recommendations (ENCM)






