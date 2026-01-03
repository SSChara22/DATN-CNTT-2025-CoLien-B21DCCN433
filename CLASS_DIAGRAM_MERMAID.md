# Biểu đồ lớp - Hệ thống gợi ý sản phẩm E-commerce
## (Class Diagram - MVC Architecture)

```mermaid
classDiagram
    %% Presentation Layer
    class FrontendComponent {
        +Integer userId
        +Integer limit
        +getRecommendations()
        +displayRecommendations()
        +trackInteraction()
    }

    %% Route Layer
    class ExpressRouter {
        +Map routes
        +registerRoute(path, handler)
        +handleRequest(req, res)
    }

    %% Middleware Layer
    class JWTVerifyMiddleware {
        +String secretString
        +verifyTokenUser(req, res, next)
        +verifyTokenAdmin(req, res, next)
        +extractToken(req)
        +validateUser(user)
    }

    %% Controller Layer
    class RecommendationController {
        +initForCurrentUser(req, res)
        +listForCurrentUser(req, res)
        +dashboardPage(req, res)
        +clearForCurrentUser(req, res)
    }

    %% Service Layer
    class RecommendationService {
        -PythonInvoker pythonInvoker
        -String ROOT
        -String PERF_CSV
        +initForUser(userId, limit)
        +getCachedForUser(userId, limit)
        +clearForUser(userId)
        -computeRecommendationsForUser(userId, limit)
        -buildUserProductFeatures(userId)
        -ensureTables()
        -pickBestModel()
        -deriveContext(timestamp)
        -priceRange(price)
        -intentFromInteractions(actions)
        -scoreSample(params)
    }

    class PythonInvoker {
        -String ROOT
        -String PYTHON
        -String SCRIPT
        +runPythonInference(payload, options)
        -spawnPythonProcess()
        -parseResponse(output)
    }

    %% Model Layer (Sequelize ORM)
    class SequelizeModel {
        <<abstract>>
        +Integer id
        +Date createdAt
        +Date updatedAt
        +findAll(options)
        +findOne(options)
        +create(data)
        +update(data)
        +destroy(options)
    }

    class Recommendation {
        +Integer userId
        +Integer productId
        +String modelName
        +Float score
        +Text details
        +belongsTo(User)
        +belongsTo(Product)
    }

    class ModelRun {
        +Integer userId
        +String modelName
        +Text metricsJson
        +Text recommendationsJson
        +belongsTo(User)
    }

    class Product {
        +String name
        +String categoryId
        +String brandId
        +String statusId
        +Integer view
        +hasMany(Recommendation)
        +hasMany(Interaction)
        +hasMany(Comment)
    }

    class User {
        +String email
        +String password
        +String roleId
        +String firstName
        +String lastName
        +hasMany(Recommendation)
        +hasMany(ModelRun)
        +hasMany(Interaction)
    }

    class Interaction {
        +Integer userId
        +Integer productId
        +String actionCode
        +String device_type
        +Date timestamp
        +belongsTo(User)
        +belongsTo(Product)
    }

    class Comment {
        +Integer userId
        +Integer productId
        +Integer star
        +Text content
        +belongsTo(User)
        +belongsTo(Product)
    }

    %% Python ML Service
    class TensorFlowModel {
        <<abstract>>
        +Integer n_users
        +Integer n_items
        +Integer embedding_dim
        +call(inputs, training)
        +load_weights(path)
        +predict(inputs)
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
        +call(user_ids, item_ids, context_features)
    }

    class LNCM {
        +List~Integer~ hidden_dims
        +Embedding user_embedding
        +Embedding item_embedding
        +Dense linear_layer
        +List~Dense~ hidden_layers
        +Dense neural_layer
        +Weight alpha
        +call(user_ids, item_ids)
    }

    class NeuMF {
        +List~Integer~ hidden_dims
        +Embedding user_embedding_gmf
        +Embedding item_embedding_gmf
        +Embedding user_embedding_mlp
        +Embedding item_embedding_mlp
        +List~Dense~ mlp_layers
        +Dense final_layer
        +call(user_ids, item_ids)
    }

    class BMF {
        +Embedding user_embedding
        +Embedding item_embedding
        +Embedding user_bias
        +Embedding item_bias
        +Weight global_bias
        +call(user_ids, item_ids)
    }

    class RecommendAPI {
        -String DB_HOST
        -Integer DB_PORT
        -String DB_USER
        -String DB_PASS
        -String DB_NAME
        -Dict _cached
        +handle_request(req)
        -load_resources()
        -db_connect()
        -_build_model(model_name, configs)
        -_encode_user(user_id, user_map)
        -_decode_item_index(idx)
        -_candidate_indices_from_db()
        -_context_to_features(context, context_info)
    }

    %% Database
    class MySQL {
        +table recommendations
        +table model_runs
        +table products
        +table users
        +table interactions
        +table comments
        +table rec_user_encoder
        +table rec_item_encoder
        +table rec_context_mapping
    }

    %% Relationships
    FrontendComponent --> ExpressRouter : HTTP Request
    ExpressRouter --> JWTVerifyMiddleware : uses
    JWTVerifyMiddleware --> User : queries
    ExpressRouter --> RecommendationController : routes to

    RecommendationController --> RecommendationService : uses
    RecommendationController --> Product : queries
    RecommendationController --> Recommendation : queries

    RecommendationService --> PythonInvoker : uses
    RecommendationService --> Recommendation : creates/queries
    RecommendationService --> ModelRun : creates/queries
    RecommendationService --> Product : queries
    RecommendationService --> User : queries
    RecommendationService --> Interaction : queries
    RecommendationService --> Comment : queries
    RecommendationService --> MySQL : queries

    PythonInvoker --> RecommendAPI : spawns process

    RecommendAPI --> MySQL : queries encoders
    RecommendAPI --> ENCM : uses
    RecommendAPI --> LNCM : uses
    RecommendAPI --> NeuMF : uses
    RecommendAPI --> BMF : uses

    Recommendation --> User : belongsTo
    Recommendation --> Product : belongsTo
    ModelRun --> User : belongsTo
    Interaction --> User : belongsTo
    Interaction --> Product : belongsTo
    Comment --> User : belongsTo
    Comment --> Product : belongsTo

    User --> Recommendation : hasMany
    User --> ModelRun : hasMany
    User --> Interaction : hasMany
    Product --> Recommendation : hasMany
    Product --> Interaction : hasMany
    Product --> Comment : hasMany

    SequelizeModel <|-- Recommendation
    SequelizeModel <|-- ModelRun
    SequelizeModel <|-- Product
    SequelizeModel <|-- User
    SequelizeModel <|-- Interaction
    SequelizeModel <|-- Comment

    TensorFlowModel <|-- ENCM
    TensorFlowModel <|-- LNCM
    TensorFlowModel <|-- NeuMF
    TensorFlowModel <|-- BMF
```

## Ghi chú

### RecommendationService
- **Business Logic Layer**: Xử lý logic tính toán gợi ý, quản lý cache, đánh giá models, fallback heuristic scoring

### PythonInvoker
- **Bridge Pattern**: Spawn Python process, giao tiếp qua stdin/stdout, JSON serialization

### ENCM (Embedding-based Neural Context Model)
- **Context-aware Model**: Sử dụng time, season, device, category
- **Architecture**: Embedding-based neural network

### LNCM (Linear Neural Combination Model)
- **Linear + Neural Combination**: Matrix Factorization + MLP
- **Learnable alpha weight**: Tự động học trọng số kết hợp

### NeuMF (Neural Matrix Factorization)
- **Two-tower Architecture**: GMF (Generalized MF) + MLP
- **Hybrid approach**: Kết hợp factorization và deep learning

### BMF (Bias Matrix Factorization)
- **Simple but effective**: User/Item embeddings + biases
- **Fast inference**: Nhẹ và nhanh






