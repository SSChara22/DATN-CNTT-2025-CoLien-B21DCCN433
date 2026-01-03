# Sơ đồ lớp thiết kế - Hệ thống E-commerce với gợi ý theo ngữ cảnh
## (Design Class Diagram - MVC + Service Layer + ML Service)

```mermaid
classDiagram
    %% Presentation Layer
    class ReactFrontend {
        +Component[] components
        +Service[] services
        +getRecommendations(userId, limit)
        +displayProducts(products)
        +trackInteraction(userId, productId, action)
    }
    
    class RecommendationComponent {
        +Integer userId
        +Integer limit
        +Product[] recommendations
        +loadRecommendations()
        +render()
    }
    
    class InteractionService {
        +trackView(userId, productId)
        +trackCart(userId, productId)
        +trackPurchase(userId, productId)
        +detectDevice() String
    }

    %% Route Layer
    class ExpressRouter {
        -Map~String,Handler~ routes
        +registerRoute(method, path, handler)
        +handleRequest(req, res)
        +use(middleware)
    }
    
    class WebRoutes {
        -ExpressRouter router
        +initRoutes(app)
        +setupUserRoutes()
        +setupProductRoutes()
        +setupRecommendationRoutes()
    }

    %% Middleware Layer
    class JWTVerifyMiddleware {
        -String secretString
        +verifyTokenUser(req, res, next)
        +verifyTokenAdmin(req, res, next)
        -extractToken(req) String
        -validateToken(token) Payload
    }

    %% Controller Layer
    class BaseController {
        <<abstract>>
        +handleRequest(req, res)
        +handleError(error, res)
        +sendResponse(data, res)
    }
    
    class RecommendationController {
        -RecommendationService recommendationService
        +initForCurrentUser(req, res)
        +listForCurrentUser(req, res)
        +dashboardPage(req, res)
        +clearForCurrentUser(req, res)
    }
    
    class UserController {
        -UserService userService
        +handleLogin(req, res)
        +handleCreateNewUser(req, res)
        +handleUpdateUser(req, res)
        +getAllUser(req, res)
    }
    
    class ProductController {
        -ProductService productService
        +createNewProduct(req, res)
        +getAllProductAdmin(req, res)
        +getProductRecommend(req, res)
    }
    
    class InteractionController {
        -InteractionService interactionService
        +logInteractionController(req, res)
        +getUserInteractionsController(req, res)
    }

    %% Service Layer
    class BaseService {
        <<abstract>>
        +validateInput(data) Boolean
        +handleError(error) Error
    }
    
    class RecommendationService {
        -PythonInvoker pythonInvoker
        -String ROOT
        -String PERF_CSV
        +initForUser(userId, limit) Promise~Object~
        +getCachedForUser(userId, limit) Promise~Recommendation[]~
        +clearForUser(userId) Promise~Boolean~
        -computeRecommendationsForUser(userId, limit) Promise~Object~
        -buildUserProductFeatures(userId) Promise~Object[]~
        -pickBestModel() String
        -deriveContext(timestamp) Object
    }
    
    class PythonInvoker {
        -String ROOT
        -String PYTHON
        -String SCRIPT
        +runPythonInference(payload, options) Promise~Object~
        -spawnPythonProcess() Process
        -parseResponse(output) Object
    }
    
    class UserService {
        +handleLogin(data) Promise~Object~
        +handleCreateNewUser(data) Promise~Object~
        +getAllUser() Promise~User[]~
        -checkUserEmail(email) Promise~Boolean~
        -hashPassword(password) String
    }
    
    class ProductService {
        +createNewProduct(data) Promise~Object~
        +getAllProductAdmin() Promise~Product[]~
        +getProductRecommend(data) Promise~Product[]~
    }
    
    class InteractionService {
        +logInteraction(userId, productId, actionCode, device) Promise~Interaction~
        +getUserInteractions(userId, filterActionCode) Promise~Interaction[]~
    }

    %% Model Layer
    class SequelizeModel {
        <<abstract>>
        +Integer id {primaryKey}
        +Date createdAt
        +Date updatedAt
        +findAll(options) Promise~Model[]~
        +findOne(options) Promise~Model~
        +create(data) Promise~Model~
        +update(data, options) Promise~Model~
        +destroy(options) Promise~Integer~
    }
    
    class Recommendation {
        +Integer userId {FK}
        +Integer productId {FK}
        +String modelName
        +Float score
        +Text details
    }
    
    class ModelRun {
        +Integer userId {FK}
        +String modelName
        +Text metricsJson
        +Text recommendationsJson
    }
    
    class User {
        +String email {unique}
        +String password
        +String firstName
        +String lastName
        +String roleId {FK}
        +String statusId {FK}
    }
    
    class Product {
        +String name
        +String categoryId {FK}
        +String brandId {FK}
        +String statusId {FK}
        +Integer view
    }
    
    class Interaction {
        +Integer userId {FK}
        +Integer productId {FK}
        +String actionCode {FK}
        +String device_type
        +Date timestamp
    }
    
    class Comment {
        +Integer userId {FK}
        +Integer productId {FK}
        +Integer star
        +Text content
    }
    
    class Order {
        +Integer userId {FK}
        +String statusId {FK}
        +Float totalPrice
    }

    %% Database
    class SequelizeConnection {
        -Sequelize sequelize
        -Object db
        +authenticate() Promise~void~
        +sync(options) Promise~void~
        +loadModels() void
    }
    
    class MySQL {
        +table recommendations
        +table model_runs
        +table users
        +table products
        +table interactions
        +table comments
        +table orders
    }

    %% Python ML Service
    class TensorFlowModel {
        <<abstract>>
        +Integer n_users
        +Integer n_items
        +Integer embedding_dim
        +call(inputs, training) Tensor
        +load_weights(path) void
        +predict(inputs) Tensor
    }
    
    class ENCM {
        +List~Integer~ n_contexts
        +Integer context_dim
        +List~Integer~ hidden_dims
        +Embedding user_embedding
        +Embedding item_embedding
        +List~Embedding~ context_embeddings
        +List~Dense~ hidden_layers
        +Dense output_layer
        +call(user_ids, item_ids, context_features) Tensor
    }
    
    class LNCM {
        +List~Integer~ hidden_dims
        +Embedding user_embedding
        +Embedding item_embedding
        +Dense linear_layer
        +List~Dense~ hidden_layers
        +Dense neural_layer
        +Weight alpha
        +call(user_ids, item_ids) Tensor
    }
    
    class NeuMF {
        +List~Integer~ hidden_dims
        +Embedding user_embedding_gmf
        +Embedding item_embedding_gmf
        +Embedding user_embedding_mlp
        +Embedding item_embedding_mlp
        +List~Dense~ mlp_layers
        +Dense final_layer
        +call(user_ids, item_ids) Tensor
    }
    
    class BMF {
        +Embedding user_embedding
        +Embedding item_embedding
        +Embedding user_bias
        +Embedding item_bias
        +Weight global_bias
        +call(user_ids, item_ids) Tensor
    }
    
    class RecommendAPI {
        -String DB_HOST
        -Integer DB_PORT
        -String DB_USER
        -String DB_PASS
        -String DB_NAME
        -Dict _cached {Singleton}
        +handle_request(req) Dict
        -load_resources() void
        -db_connect() Connection
        -_build_model(model_name, configs) TensorFlowModel {Factory}
        -_encode_user(user_id, user_map) Integer
        -_decode_item_index(idx) Integer
    }
    
    class ModelFactory {
        <<Factory>>
        +createModel(model_name, config) TensorFlowModel
        +loadConfig(model_name) Dict
        +loadWeights(model, model_name) void
    }

    %% Relationships
    ReactFrontend --> RecommendationComponent : contains
    ReactFrontend --> InteractionService : uses
    
    ExpressRouter --> WebRoutes : uses
    WebRoutes --> RecommendationController : routes to
    WebRoutes --> UserController : routes to
    WebRoutes --> ProductController : routes to
    WebRoutes --> InteractionController : routes to
    
    ExpressRouter --> JWTVerifyMiddleware : uses
    JWTVerifyMiddleware --> User : queries
    
    BaseController <|-- RecommendationController
    BaseController <|-- UserController
    BaseController <|-- ProductController
    BaseController <|-- InteractionController
    
    RecommendationController --> RecommendationService : uses
    UserController --> UserService : uses
    ProductController --> ProductService : uses
    InteractionController --> InteractionService : uses
    
    BaseService <|-- RecommendationService
    BaseService <|-- UserService
    BaseService <|-- ProductService
    BaseService <|-- InteractionService
    
    RecommendationService --> PythonInvoker : uses
    RecommendationService --> Recommendation : creates/queries
    RecommendationService --> ModelRun : creates/queries
    RecommendationService --> Product : queries
    RecommendationService --> User : queries
    RecommendationService --> Interaction : queries
    
    PythonInvoker --> RecommendAPI : spawns process
    
    UserService --> User : queries/creates
    ProductService --> Product : queries/creates
    InteractionService --> Interaction : creates/queries
    
    SequelizeModel <|-- Recommendation
    SequelizeModel <|-- ModelRun
    SequelizeModel <|-- User
    SequelizeModel <|-- Product
    SequelizeModel <|-- Interaction
    SequelizeModel <|-- Comment
    SequelizeModel <|-- Order
    
    Recommendation "0..*" --> "1" User : belongsTo
    Recommendation "0..*" --> "1" Product : belongsTo
    ModelRun "0..*" --> "1" User : belongsTo
    Interaction "0..*" --> "1" User : belongsTo
    Interaction "0..*" --> "1" Product : belongsTo
    Comment "0..*" --> "1" User : belongsTo
    Comment "0..*" --> "1" Product : belongsTo
    Order "0..*" --> "1" User : belongsTo
    
    User "1" --> "0..*" Recommendation : hasMany
    User "1" --> "0..*" ModelRun : hasMany
    User "1" --> "0..*" Interaction : hasMany
    User "1" --> "0..*" Order : hasMany
    Product "1" --> "0..*" Recommendation : hasMany
    Product "1" --> "0..*" Interaction : hasMany
    Product "1" --> "0..*" Comment : hasMany
    
    SequelizeConnection --> MySQL : connects to
    SequelizeModel --> SequelizeConnection : uses
    
    TensorFlowModel <|-- ENCM
    TensorFlowModel <|-- LNCM
    TensorFlowModel <|-- NeuMF
    TensorFlowModel <|-- BMF
    
    RecommendAPI --> ModelFactory : uses
    ModelFactory --> ENCM : creates
    ModelFactory --> LNCM : creates
    ModelFactory --> NeuMF : creates
    ModelFactory --> BMF : creates
    
    RecommendAPI --> MySQL : queries encoders
    RecommendAPI --> ENCM : uses
    RecommendAPI --> LNCM : uses
    RecommendAPI --> NeuMF : uses
    RecommendAPI --> BMF : uses
```

## Design Patterns được sử dụng:

### 1. **MVC Pattern (Model-View-Controller)**
- **Model**: SequelizeModel và các subclasses
- **View**: ReactFrontend components
- **Controller**: RecommendationController, UserController, etc.

### 2. **Service Layer Pattern**
- Tách biệt business logic khỏi controllers
- BaseService làm abstract base class

### 3. **Singleton Pattern**
- `RecommendAPI._cached`: Lưu trữ models và encoders để tránh reload

### 4. **Factory Pattern**
- `ModelFactory`: Tạo các ML models dựa trên tên model
- `RecommendAPI._build_model()`: Factory method

### 5. **Strategy Pattern**
- `RecommendationService.pickBestModel()`: Chọn model tốt nhất dựa trên metrics

### 6. **Template Method Pattern**
- `BaseController`: Định nghĩa cấu trúc chung cho request handling

### 7. **Repository Pattern**
- SequelizeModel: Abstract data access layer
- Các models cụ thể implement repository methods

### 8. **Bridge Pattern**
- `PythonInvoker`: Bridge giữa Node.js và Python service

## Kiến trúc phân lớp:

1. **Presentation Layer**: ReactFrontend, Components
2. **Route Layer**: ExpressRouter, WebRoutes
3. **Middleware Layer**: JWTVerifyMiddleware
4. **Controller Layer**: Các controllers xử lý HTTP requests
5. **Service Layer**: Business logic và orchestration
6. **Model Layer**: Data access và domain models
7. **Database Layer**: MySQL persistence
8. **ML Service Layer**: Python ML models và inference






