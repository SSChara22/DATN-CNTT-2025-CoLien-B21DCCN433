# Tài Liệu API - Hệ Thống E-Commerce

## Format Response Chung

Tất cả API đều trả về format JSON với cấu trúc:
```json
{
  "errCode": 0,           // 0 = thành công, khác 0 = lỗi
  "errMessage": "ok",     // Thông báo lỗi/thành công
  "data": {},            // Dữ liệu trả về (nếu có)
  "count": 0,            // Tổng số bản ghi (cho API phân trang)
  "message": "",         // Thông điệp (một số API)
  "accessToken": "",     // Token JWT (API login)
  "user": {},            // Thông tin user (API login)
  "link": ""             // Link redirect (payment APIs)
}
```

---

## 1. API USER

### 1.1. POST `/api/create-new-user`
**Input (Body):**
```json
{
  "email": "user@example.com",      // Bắt buộc
  "password": "password123",
  "firstName": "John",
  "lastName": "Doe",                 // Bắt buộc
  "address": "123 Main St",
  "roleId": "R1",
  "genderId": "G1",
  "phonenumber": "0123456789",
  "avatar": "base64_image_string",
  "dob": "1990-01-01"
}
```

**Output:**
```json
// Thành công
{
  "errCode": 0,
  "message": "OK"
}

// Lỗi - Email đã tồn tại
{
  "errCode": 1,
  "errMessage": "Your email is already in used, Plz try another email!"
}

// Lỗi - Thiếu tham số
{
  "errCode": 2,
  "errMessage": "Missing required parameters !"
}
```

---

### 1.2. PUT `/api/update-user` (User Token Required)
**Input (Body):**
```json
{
  "id": 1,                           // Bắt buộc
  "firstName": "John",
  "lastName": "Doe",
  "address": "123 Main St",
  "roleId": "R1",
  "genderId": "G1",                  // Bắt buộc
  "phonenumber": "0123456789",
  "dob": "1990-01-01",
  "image": "base64_image_string"     // Tùy chọn
}
```

**Output:**
```json
// Thành công
{
  "errCode": 0,
  "errMessage": "Update the user succeeds!"
}

// Lỗi - User không tồn tại
{
  "errCode": 1,
  "errMessage": "User not found!"
}

// Lỗi - Thiếu tham số
{
  "errCode": 2,
  "errMessage": "Missing required parameters"
}
```

---

### 1.3. DELETE `/api/delete-user` (Admin Token Required)
**Input (Body):**
```json
{
  "id": 1
}
```

**Output:**
```json
// Thành công
{
  "errCode": 0,
  "message": "The user is deleted"
}

// Lỗi - User không tồn tại
{
  "errCode": 2,
  "errMessage": "The user isn't exist"
}
```

---

### 1.4. POST `/api/login`
**Input (Body):**
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**Output:**
```json
// Thành công
{
  "errCode": 0,
  "errMessage": "Ok",
  "user": {
    "id": 1,
    "email": "user@example.com",
    "roleId": "R1",
    "firstName": "John",
    "lastName": "Doe"
  },
  "accessToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}

// Lỗi - Email không tồn tại
{
  "errCode": 1,
  "errMessage": "Your's email isn't exist in your system. plz try other email"
}

// Lỗi - User không tồn tại
{
  "errCode": 2,
  "errMessage": "User not found!"
}

// Lỗi - Sai mật khẩu
{
  "errCode": 3,
  "errMessage": "Wrong password"
}

// Lỗi - Thiếu tham số
{
  "errCode": 4,
  "errMessage": "Missing required parameters!"
}
```

---

### 1.5. POST `/api/changepassword` (User Token Required)
**Input (Body):**
```json
{
  "id": 1,
  "oldpassword": "oldpass123",
  "password": "newpass123"
}
```

**Output:**
```json
// Thành công
{
  "errCode": 0,
  "errMessage": "ok"
}

// Lỗi - Mật khẩu cũ không chính xác
{
  "errCode": 2,
  "errMessage": "Mật khẩu cũ không chính xác"
}
```

---

### 1.6. GET `/api/get-all-user` (Admin Token Required)
**Input (Query):**
```
?limit=10&offset=0&keyword=0123
```

**Output:**
```json
{
  "errCode": 0,
  "data": [
    {
      "id": 1,
      "email": "user@example.com",
      "firstName": "John",
      "lastName": "Doe",
      "phonenumber": "0123456789",
      "roleData": {
        "value": "Admin",
        "code": "R1"
      },
      "genderData": {
        "value": "Nam",
        "code": "G1"
      }
    }
  ],
  "count": 100
}
```

---

### 1.7. GET `/api/get-detail-user-by-id`
**Input (Query):**
```
?id=1
```

**Output:**
```json
{
  "errCode": 0,
  "data": {
    "id": 1,
    "email": "user@example.com",
    "firstName": "John",
    "lastName": "Doe",
    "address": "123 Main St",
    "phonenumber": "0123456789",
    "image": "binary_image_data",
    "dob": "1990-01-01",
    "roleData": {
      "value": "Admin",
      "code": "R1"
    },
    "genderData": {
      "value": "Nam",
      "code": "G1"
    }
  }
}
```

---

### 1.8. GET `/api/get-detail-user-by-email`
**Input (Query):**
```
?email=user@example.com
```

**Output:**
```json
{
  "errCode": 0,
  "data": {
    "password": "hashed_password"
  }
}
```

---

### 1.9. POST `/api/send-verify-email` (User Token Required)
**Input (Body):**
```json
{
  "id": 1
}
```

**Output:**
```json
{
  "errCode": 0,
  "errMessage": "ok"
}
```

---

### 1.10. POST `/api/verify-email` (User Token Required)
**Input (Body):**
```json
{
  "id": 1,
  "token": "uuid-token-string"
}
```

**Output:**
```json
// Thành công
{
  "errCode": 0,
  "errMessage": "ok"
}

// Lỗi - User không tồn tại
{
  "errCode": 2,
  "errMessage": "User not found!"
}
```

---

### 1.11. POST `/api/send-forgotpassword-email`
**Input (Body):**
```json
{
  "email": "user@example.com"
}
```

**Output:**
```json
// Thành công
{
  "errCode": 0,
  "errMessage": "ok"
}

// Lỗi - Email không tồn tại
{
  "errCode": 2,
  "errMessage": "Your's email isn't exist in your system. plz try other email"
}
```

---

### 1.12. POST `/api/forgotpassword-email`
**Input (Body):**
```json
{
  "id": 1,
  "token": "uuid-token-string",
  "password": "newpassword123"
}
```

**Output:**
```json
{
  "errCode": 0,
  "errMessage": "ok"
}
```

---

### 1.13. GET `/api/check-phonenumber-email`
**Input (Query):**
```
?phonenumber=0123456789&email=user@example.com
```

**Output:**
```json
// Hợp lệ
{
  "isCheck": false,
  "errMessage": "Hợp lệ"
}

// Số điện thoại đã tồn tại
{
  "isCheck": true,
  "errMessage": "Số điện thoại đã tồn tại"
}

// Email đã tồn tại
{
  "isCheck": true,
  "errMessage": "Email đã tồn tại"
}
```

---

## 2. API ALLCODE

### 2.1. POST `/api/create-new-all-code` (Admin Token Required)
**Input (Body):**
```json
{
  "type": "GENDER",
  "value": "Nam",
  "code": "G1"
}
```

**Output:**
```json
// Thành công
{
  "errCode": 0,
  "errMessage": "ok"
}

// Lỗi - Code đã tồn tại
{
  "errCode": 2,
  "errMessage": "Mã code đã tồn tại !"
}
```

---

### 2.2. GET `/api/get-all-code`
**Input (Query):**
```
?type=GENDER
```

**Output:**
```json
{
  "errCode": 0,
  "data": [
    {
      "id": 1,
      "type": "GENDER",
      "value": "Nam",
      "code": "G1"
    }
  ]
}
```

---

### 2.3. GET `/api/get-list-allcode`
**Input (Query):**
```
?type=GENDER&limit=10&offset=0&keyword=Nam
```

**Output:**
```json
{
  "errCode": 0,
  "data": [
    {
      "id": 1,
      "type": "GENDER",
      "value": "Nam",
      "code": "G1"
    }
  ],
  "count": 2
}
```

---

### 2.4. GET `/api/get-all-category-blog`
**Input (Query):**
```
?type=BLOG_CATEGORY
```

**Output:**
```json
{
  "errCode": 0,
  "data": [
    {
      "id": 1,
      "type": "BLOG_CATEGORY",
      "value": "Thời trang",
      "code": "BC1",
      "countPost": 5
    }
  ]
}
```

---

## 3. API PRODUCT

### 3.1. POST `/api/create-new-product` (Admin Token Required)
**Input (Body):**
```json
{
  "name": "Áo sơ mi nam",
  "contentHTML": "<p>Mô tả HTML</p>",
  "contentMarkdown": "Mô tả Markdown",
  "categoryId": "C1",
  "brandId": "B1",
  "madeby": "Việt Nam",
  "material": "Cotton",
  "nameDetail": "Áo sơ mi trắng size M",
  "description": "Mô tả chi tiết",
  "originalPrice": 500000,
  "discountPrice": 400000,
  "image": "base64_image_string",
  "width": 50,
  "height": 70,
  "sizeId": "S1",
  "weight": 200
}
```

**Output:**
```json
{
  "errCode": 0,
  "errMessage": "ok"
}
```

---

### 3.2. GET `/api/get-all-product-user`
**Input (Query):**
```
?limit=10&offset=0&categoryId=C1&brandId=B1&sortName=true&sortPrice=true&keyword=áo
```

**Output:**
```json
{
  "errCode": 0,
  "data": [
    {
      "id": 1,
      "name": "Áo sơ mi nam",
      "categoryId": "C1",
      "brandId": "B1",
      "price": 400000,
      "categoryData": {
        "value": "Áo",
        "code": "C1"
      },
      "brandData": {
        "value": "Nike",
        "code": "B1"
      },
      "productDetail": [
        {
          "id": 1,
          "nameDetail": "Áo sơ mi trắng size M",
          "originalPrice": 500000,
          "discountPrice": 400000,
          "productImage": [
            {
              "id": 1,
              "image": "binary_image_data"
            }
          ],
          "productDetailSize": [
            {
              "id": 1,
              "sizeId": "S1",
              "stock": 10
            }
          ]
        }
      ]
    }
  ],
  "count": 50
}
```

---

### 3.3. GET `/api/get-detail-product-by-id`
**Input (Query):**
```
?id=1
```

**Output:**
```json
{
  "errCode": 0,
  "data": {
    "id": 1,
    "name": "Áo sơ mi nam",
    "contentHTML": "<p>Mô tả</p>",
    "contentMarkdown": "Mô tả",
    "categoryId": "C1",
    "brandId": "B1",
    "categoryData": {
      "value": "Áo",
      "code": "C1"
    },
    "productDetail": [
      {
        "id": 1,
        "nameDetail": "Áo sơ mi trắng size M",
        "productImage": [
          {
            "id": 1,
            "image": "binary_image_data"
          }
        ],
        "productDetailSize": [
          {
            "id": 1,
            "sizeId": "S1",
            "stock": 10,
            "sizeData": {
              "value": "M",
              "code": "S1"
            }
          }
        ]
      }
    ]
  }
}
```

---

### 3.4. GET `/api/get-product-feature`
**Input (Query):**
```
?limit=10
```

**Output:**
```json
{
  "errCode": 0,
  "data": [
    {
      "id": 1,
      "name": "Áo sơ mi nam",
      "price": 400000
    }
  ]
}
```

---

## 4. API SHOPCART

### 4.1. POST `/api/add-shopcart` (User Token Required)
**Input (Body):**
```json
{
  "userId": 1,
  "productdetailsizeId": 1,
  "quantity": 2,
  "type": "UPDATE_QUANTITY"  // Tùy chọn: "UPDATE_QUANTITY" hoặc không có
}
```

**Output:**
```json
// Thành công
{
  "errCode": 0,
  "errMessage": "ok"
}

// Lỗi - Hết hàng
{
  "errCode": 2,
  "errMessage": "Chỉ còn 5 sản phẩm",
  "quantity": 5
}
```

---

### 4.2. GET `/api/get-all-shopcart-by-userId` (User Token Required)
**Input (Query):**
```
?id=1
```

**Output:**
```json
{
  "errCode": 0,
  "data": [
    {
      "id": 1,
      "userId": 1,
      "productdetailsizeId": 1,
      "quantity": 2,
      "productdetailsizeData": {
        "id": 1,
        "sizeData": {
          "value": "M",
          "code": "S1"
        }
      },
      "productDetail": {
        "id": 1,
        "nameDetail": "Áo sơ mi trắng size M",
        "originalPrice": 500000,
        "discountPrice": 400000
      },
      "productDetailImage": [
        {
          "id": 1,
          "image": "binary_image_data"
        }
      ],
      "productData": {
        "id": 1,
        "name": "Áo sơ mi nam"
      }
    }
  ]
}
```

---

### 4.3. DELETE `/api/delete-item-shopcart` (User Token Required)
**Input (Body):**
```json
{
  "id": 1
}
```

**Output:**
```json
{
  "errCode": 0,
  "errMessage": "ok"
}
```

---

## 5. API ORDER

### 5.1. POST `/api/create-new-order` (User Token Required)
**Input (Body):**
```json
{
  "userId": 1,
  "addressUserId": 1,
  "typeShipId": 1,
  "isPaymentOnlien": 0,
  "voucherId": 1,  // Tùy chọn
  "note": "Giao hàng nhanh",
  "arrDataShopCart": [
    {
      "productId": 1,  // productDetailSizeId
      "quantity": 2,
      "price": 400000
    }
  ]
}
```

**Output:**
```json
{
  "errCode": 0,
  "errMessage": "ok"
}
```

---

### 5.2. GET `/api/get-all-order`
**Input (Query):**
```
?limit=10&offset=0&statusId=S3
```

**Output:**
```json
{
  "errCode": 0,
  "data": [
    {
      "id": 1,
      "addressUserId": 1,
      "statusId": "S3",
      "typeShipId": 1,
      "voucherId": 1,
      "createdAt": "2025-01-01T00:00:00.000Z",
      "typeShipData": {
        "id": 1,
        "name": "Giao hàng nhanh"
      },
      "voucherData": {
        "id": 1,
        "codeVoucher": "DISCOUNT10"
      },
      "statusOrderData": {
        "value": "Đang xử lý",
        "code": "S3"
      },
      "userData": {
        "id": 1,
        "email": "user@example.com"
      },
      "addressUser": {
        "id": 1,
        "shipName": "John Doe",
        "shipAdress": "123 Main St"
      }
    }
  ],
  "count": 50
}
```

---

### 5.3. GET `/api/get-detail-order`
**Input (Query):**
```
?id=1
```

**Output:**
```json
{
  "errCode": 0,
  "data": {
    "id": 1,
    "addressUserId": 1,
    "statusId": "S3",
    "image": "binary_image_data",
    "orderDetail": [
      {
        "id": 1,
        "productId": 1,
        "quantity": 2,
        "price": 400000,
        "productDetailSize": {
          "id": 1,
          "sizeData": {
            "value": "M",
            "code": "S1"
          }
        },
        "productDetail": {
          "id": 1,
          "nameDetail": "Áo sơ mi trắng size M"
        },
        "product": {
          "id": 1,
          "name": "Áo sơ mi nam"
        },
        "productImage": [
          {
            "id": 1,
            "image": "binary_image_data"
          }
        ]
      }
    ],
    "addressUser": {
      "id": 1,
      "shipName": "John Doe",
      "shipAdress": "123 Main St",
      "shipEmail": "user@example.com",
      "shipPhonenumber": "0123456789"
    },
    "userData": {
      "id": 1,
      "email": "user@example.com"
    }
  }
}
```

---

### 5.4. POST `/api/payment-order` (User Token Required)
**Input (Body):**
```json
{
  "result": [
    {
      "productId": 1,
      "quantity": 2,
      "realPrice": 400000
    }
  ],
  "total": 800000
}
```

**Output:**
```json
// Thành công
{
  "errCode": 0,
  "errMessage": "ok",
  "link": "https://www.sandbox.paypal.com/checkoutnow?token=..."
}

// Lỗi
{
  "errCode": -1,
  "errMessage": "error_message"
}
```

---

### 5.5. POST `/api/payment-order-vnpay` (User Token Required)
**Input (Body):**
```json
{
  "amount": 800000,
  "bankCode": "NCB",
  "orderDescription": "Thanh toan don hang",
  "orderType": "other",
  "language": "vn"
}
```

**Output:**
```json
{
  "errCode": 200,
  "link": "https://sandbox.vnpayment.vn/paymentv2/vpcpay.html?vnp_Amount=..."
}
```

---

## 6. API ADDRESS USER

### 6.1. POST `/api/create-new-address-user` (User Token Required)
**Input (Body):**
```json
{
  "userId": 1,
  "shipName": "John Doe",
  "shipAdress": "123 Main St",
  "shipEmail": "user@example.com",
  "shipPhonenumber": "0123456789"
}
```

**Output:**
```json
{
  "errCode": 0,
  "errMessage": "ok"
}
```

---

### 6.2. GET `/api/get-all-address-user` (User Token Required)
**Input (Query):**
```
?userId=1
```

**Output:**
```json
{
  "errCode": 0,
  "data": [
    {
      "id": 1,
      "userId": 1,
      "shipName": "John Doe",
      "shipAdress": "123 Main St",
      "shipEmail": "user@example.com",
      "shipPhonenumber": "0123456789",
      "createdAt": "2025-01-01T00:00:00.000Z",
      "updatedAt": "2025-01-01T00:00:00.000Z"
    }
  ]
}
```

---

## 7. API BANNER

### 7.1. POST `/api/create-new-banner` (Admin Token Required)
**Input (Body):**
```json
{
  "name": "Banner giảm giá",
  "description": "Mô tả banner",
  "image": "base64_image_string"
}
```

**Output:**
```json
{
  "errCode": 0,
  "errMessage": "ok"
}
```

---

### 7.2. GET `/api/get-all-banner`
**Input (Query):**
```
?limit=10&offset=0&keyword=giảm giá
```

**Output:**
```json
{
  "errCode": 0,
  "data": [
    {
      "id": 1,
      "name": "Banner giảm giá",
      "description": "Mô tả banner",
      "image": "binary_image_data",
      "statusId": "S1"
    }
  ],
  "count": 5
}
```

---

## 8. API BLOG

### 8.1. POST `/api/create-new-blog` (Admin Token Required)
**Input (Body):**
```json
{
  "title": "Bài viết mới",
  "contentHTML": "<p>Nội dung HTML</p>",
  "contentMarkdown": "Nội dung Markdown",
  "image": "base64_image_string",
  "subjectId": "BC1",
  "userId": 1,
  "shortdescription": "Mô tả ngắn"
}
```

**Output:**
```json
{
  "errCode": 0,
  "errMessage": "ok"
}
```

---

### 8.2. GET `/api/get-all-blog`
**Input (Query):**
```
?limit=10&offset=0&subjectId=BC1&keyword=bài viết
```

**Output:**
```json
{
  "errCode": 0,
  "data": [
    {
      "id": 1,
      "title": "Bài viết mới",
      "shortdescription": "Mô tả ngắn",
      "image": "binary_image_data",
      "view": 100,
      "subjectData": {
        "value": "Thời trang",
        "code": "BC1"
      },
      "userData": {
        "id": 1,
        "firstName": "John",
        "lastName": "Doe"
      },
      "commentData": [
        {
          "id": 1,
          "content": "Bình luận"
        }
      ]
    }
  ],
  "count": 20
}
```

---

## 9. API VOUCHER

### 9.1. POST `/api/create-new-voucher` (Admin Token Required)
**Input (Body):**
```json
{
  "fromDate": "2025-01-01",
  "toDate": "2025-12-31",
  "typeVoucherId": 1,
  "amount": 100,
  "codeVoucher": "DISCOUNT10"
}
```

**Output:**
```json
{
  "errCode": 0,
  "errMessage": "ok"
}
```

---

### 9.2. GET `/api/get-all-voucher`
**Input (Query):**
```
?limit=10&offset=0
```

**Output:**
```json
{
  "errCode": 0,
  "data": [
    {
      "id": 1,
      "fromDate": "2025-01-01",
      "toDate": "2025-12-31",
      "typeVoucherId": 1,
      "amount": 100,
      "codeVoucher": "DISCOUNT10",
      "usedAmount": 50,
      "typeVoucherOfVoucherData": {
        "id": 1,
        "typeVoucherData": {
          "value": "Giảm giá",
          "code": "TV1"
        }
      }
    }
  ],
  "count": 10
}
```

---

### 9.3. POST `/api/save-user-voucher` (User Token Required)
**Input (Body):**
```json
{
  "voucherId": 1,
  "userId": 1
}
```

**Output:**
```json
// Thành công
{
  "errCode": 0,
  "errMessage": "ok"
}

// Lỗi - Đã lưu voucher
{
  "errCode": 2,
  "errMessage": "Đã lưu voucher này trong kho!"
}
```

---

## 10. API REVIEW (Sản phẩm)

### 10.1. POST `/api/create-new-review` (User Token Required)
**Input (Body):**
```json
{
  "productId": 1,
  "userId": 1,
  "content": "Sản phẩm rất tốt",
  "star": 5,
  "image": "base64_image_string"  // Tùy chọn
}
```

**Output:**
```json
{
  "errCode": 0,
  "errMessage": "ok"
}
```

---

### 10.2. GET `/api/get-all-review-by-productId`
**Input (Query):**
```
?id=1
```

**Output:**
```json
{
  "errCode": 0,
  "data": [
    {
      "id": 1,
      "productId": 1,
      "userId": 1,
      "content": "Sản phẩm rất tốt",
      "star": 5,
      "image": "binary_image_data",
      "childComment": [
        {
          "id": 2,
          "content": "Cảm ơn bạn đã đánh giá"
        }
      ],
      "user": {
        "id": 1,
        "firstName": "John",
        "lastName": "Doe",
        "image": "binary_image_data"
      }
    }
  ]
}
```

---

## 11. API COMMENT (Blog)

### 11.1. POST `/api/create-new-comment` (User Token Required)
**Input (Body):**
```json
{
  "blogId": 1,
  "userId": 1,
  "content": "Bình luận hay",
  "image": "base64_image_string"  // Tùy chọn
}
```

**Output:**
```json
{
  "errCode": 0,
  "errMessage": "ok"
}
```

---

## 12. API RECOMMENDATION

### 12.1. POST `/api/recommend/init` (User Token Required)
**Input (Query):**
```
?limit=10
```

**Output:**
```json
{
  "errCode": 0,
  "message": "initialized"
}
```

---

### 12.2. GET `/api/recommend/list` (User Token Required)
**Input (Query):**
```
?limit=10
```

**Output:**
```json
{
  "errCode": 0,
  "data": [
    {
      "product": {
        "id": 1,
        "name": "Áo sơ mi nam",
        "categoryId": "C1",
        "brandId": "B1"
      },
      "score": 0.85,
      "modelName": "LNCM"
    }
  ]
}
```

---

### 12.3. DELETE `/api/recommend/clear` (User Token Required)
**Output:**
```json
{
  "errCode": 0,
  "message": "cleared"
}
```

---

## 13. API INTERACTION

### 13.1. POST `/api/interaction`
**Input (Body):**
```json
{
  "userId": 1,
  "productId": 1,
  "actionCode": "view",
  "device": "mobile"
}
```

**Output:**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "userId": 1,
    "productId": 1,
    "actionCode": "view",
    "device_type": "mobile",
    "timestamp": "2025-01-01T00:00:00.000Z"
  }
}
```

---

### 13.2. GET `/api/interaction/user/:userId`
**Input (Params):**
```
userId: 1
```

**Input (Query):**
```
?action=purchase
```

**Output:**
```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "userId": 1,
      "productId": 1,
      "actionCode": "purchase",
      "device_type": "mobile",
      "timestamp": "2025-01-01T00:00:00.000Z",
      "productData": {
        "id": 1,
        "name": "Áo sơ mi nam"
      },
      "actionData": {
        "code": "purchase",
        "value": "Mua hàng"
      }
    }
  ]
}
```

---

## 14. API TYPESHIP

### 14.1. POST `/api/create-new-typeship` (Admin Token Required)
**Input (Body):**
```json
{
  "type": "Giao hàng nhanh",
  "price": 30000
}
```

**Output:**
```json
{
  "errCode": 0,
  "errMessage": "ok"
}
```

---

### 14.2. GET `/api/get-all-typeship`
**Input (Query):**
```
?limit=10&offset=0&keyword=nhanh
```

**Output:**
```json
{
  "errCode": 0,
  "data": [
    {
      "id": 1,
      "type": "Giao hàng nhanh",
      "price": 30000
    }
  ],
  "count": 5
}
```

---

### 14.3. GET `/api/get-detail-typeship`
**Input (Query):**
```
?id=1
```

**Output:**
```json
{
  "errCode": 0,
  "data": {
    "id": 1,
    "type": "Giao hàng nhanh",
    "price": 30000
  }
}
```

---

## 15. API SUPPLIER

### 15.1. POST `/api/create-new-supplier` (Admin Token Required)
**Input (Body):**
```json
{
  "name": "Nhà cung cấp ABC",
  "address": "123 Đường XYZ",
  "phonenumber": "0123456789",
  "email": "supplier@example.com"
}
```

**Output:**
```json
{
  "errCode": 0,
  "errMessage": "ok"
}
```

---

### 15.2. GET `/api/get-all-supplier`
**Input (Query):**
```
?limit=10&offset=0&keyword=ABC
```

**Output:**
```json
{
  "errCode": 0,
  "data": [
    {
      "id": 1,
      "name": "Nhà cung cấp ABC",
      "address": "123 Đường XYZ",
      "phonenumber": "0123456789",
      "email": "supplier@example.com"
    }
  ],
  "count": 10
}
```

---

### 15.3. GET `/api/get-detail-supplier`
**Input (Query):**
```
?id=1
```

**Output:**
```json
{
  "errCode": 0,
  "data": {
    "id": 1,
    "name": "Nhà cung cấp ABC",
    "address": "123 Đường XYZ",
    "phonenumber": "0123456789",
    "email": "supplier@example.com"
  }
}
```

---

## 16. API RECEIPT (Nhập hàng)

### 16.1. POST `/api/create-new-receipt` (Admin Token Required)
**Input (Body):**
```json
{
  "userId": 1,
  "supplierId": 1,
  "productDetailSizeId": 1,
  "quantity": 100,
  "price": 200000
}
```

**Output:**
```json
{
  "errCode": 0,
  "errMessage": "ok"
}
```

---

### 16.2. POST `/api/create-new-detail-receipt` (Admin Token Required)
**Input (Body):**
```json
{
  "receiptId": 1,
  "productDetailSizeId": 1,
  "quantity": 50,
  "price": 200000
}
```

**Output:**
```json
{
  "errCode": 0,
  "errMessage": "ok"
}
```

---

### 16.3. GET `/api/get-all-receipt`
**Input (Query):**
```
?limit=10&offset=0
```

**Output:**
```json
{
  "errCode": 0,
  "data": [
    {
      "id": 1,
      "userId": 1,
      "supplierId": 1,
      "createdAt": "2025-01-01T00:00:00.000Z",
      "userData": {
        "id": 1,
        "firstName": "John",
        "lastName": "Doe"
      },
      "supplierData": {
        "id": 1,
        "name": "Nhà cung cấp ABC"
      }
    }
  ],
  "count": 20
}
```

---

### 16.4. GET `/api/get-detail-receipt`
**Input (Query):**
```
?id=1
```

**Output:**
```json
{
  "errCode": 0,
  "data": {
    "id": 1,
    "userId": 1,
    "supplierId": 1,
    "receiptDetail": [
      {
        "id": 1,
        "receiptId": 1,
        "productDetailSizeId": 1,
        "quantity": 100,
        "price": 200000,
        "productDetailSizeData": {
          "id": 1,
          "sizeData": {
            "value": "M",
            "code": "S1"
          }
        },
        "productDetailData": {
          "id": 1,
          "nameDetail": "Áo sơ mi trắng size M"
        },
        "productData": {
          "id": 1,
          "name": "Áo sơ mi nam"
        }
      }
    ]
  }
}
```

---

## 17. API STATISTIC (Admin Token Required)

### 17.1. GET `/api/get-count-card-statistic`
**Input (Query):**
```
(Không có tham số)
```

**Output:**
```json
{
  "errCode": 0,
  "data": {
    "countUser": 100,
    "countProduct": 500,
    "countReview": 250,
    "countOrder": 300
  }
}
```

---

### 17.2. GET `/api/get-count-status-order`
**Input (Query):**
```
?oneDate=2025-01-01&twoDate=2025-01-31&type=day
```

**Output:**
```json
{
  "errCode": 0,
  "data": {
    "arrayLable": ["Đang xử lý", "Đang giao", "Đã giao", "Đã hủy"],
    "arrayValue": [10, 5, 20, 2]
  }
}
```

---

### 17.3. GET `/api/get-statistic-by-month`
**Input (Query):**
```
?year=2025
```

**Output:**
```json
{
  "errCode": 0,
  "data": {
    "arrayMonthLable": ["Th 1", "Th 2", "Th 3", ...],
    "arrayMonthValue": [1000000, 1500000, 2000000, ...]
  }
}
```

---

### 17.4. GET `/api/get-statistic-by-day`
**Input (Query):**
```
?month=1&year=2025
```

**Output:**
```json
{
  "errCode": 0,
  "data": {
    "arrayDayLable": [1, 2, 3, ..., "Today", ...],
    "arrayDayValue": [50000, 75000, 100000, ...]
  }
}
```

---

### 17.5. GET `/api/get-statistic-profit`
**Input (Query):**
```
?oneDate=2025-01-01&twoDate=2025-01-31&type=day
```

**Output:**
```json
{
  "errCode": 0,
  "data": [
    {
      "id": 1,
      "totalpriceProduct": 800000,
      "importPrice": 400000,
      "profitPrice": 400000,
      "orderDetail": [...]
    }
  ]
}
```

---

### 17.6. GET `/api/get-statistic-overturn`
**Input (Query):**
```
?oneDate=2025-01-01&twoDate=2025-01-31&type=day
```

**Output:**
```json
{
  "errCode": 0,
  "data": [
    {
      "id": 1,
      "totalpriceProduct": 800000,
      "orderDetail": [...]
    }
  ]
}
```

---

### 17.7. GET `/api/get-statistic-stock-product`
**Input (Query):**
```
?limit=10&offset=0
```

**Output:**
```json
{
  "errCode": 0,
  "data": [
    {
      "id": 1,
      "productdetailId": 1,
      "stock": 50,
      "sizeData": {
        "value": "M",
        "code": "S1"
      },
      "productDetaildData": {
        "id": 1,
        "nameDetail": "Áo sơ mi trắng size M"
      },
      "productdData": {
        "id": 1,
        "name": "Áo sơ mi nam",
        "categoryData": {
          "value": "Áo",
          "code": "C1"
        },
        "brandData": {
          "value": "Nike",
          "code": "B1"
        }
      }
    }
  ],
  "count": 100
}
```

---

## 18. API MESSAGE (Chat)

### 18.1. POST `/api/create-new-room` (User Token Required)
**Input (Body):**
```json
{
  "userId1": 1
}
```

**Output:**
```json
// Thành công
{
  "errCode": 0,
  "errMessage": "ok"
}

// Lỗi - Đã có phòng
{
  "errCode": 2,
  "errMessage": "Da Co Phong"
}
```

---

### 18.2. POST `/api/sendMessage` (User Token Required)
**Input (Body):**
```json
{
  "userId": 1,
  "roomId": 1,
  "text": "Xin chào"
}
```

**Output:**
```json
{
  "errCode": 0,
  "errMessage": "ok"
}
```

---

### 18.3. GET `/api/loadMessage` (User Token Required)
**Input (Query):**
```
?roomId=1&userId=1
```

**Output:**
```json
{
  "errCode": 0,
  "data": [
    {
      "id": 1,
      "roomId": 1,
      "userId": 1,
      "text": "Xin chào",
      "unRead": false,
      "createdAt": "2025-01-01T00:00:00.000Z",
      "userData": {
        "id": 1,
        "firstName": "John",
        "lastName": "Doe",
        "image": "binary_image_data"
      }
    }
  ]
}
```

---

### 18.4. GET `/api/listRoomOfUser` (User Token Required)
**Input (Query):**
```
?userId=1
```

**Output:**
```json
{
  "errCode": 0,
  "data": [
    {
      "id": 1,
      "userOne": 1,
      "userTwo": 2,
      "userOneData": {
        "id": 1,
        "firstName": "John",
        "lastName": "Doe",
        "image": "binary_image_data"
      },
      "userTwoData": {
        "id": 2,
        "firstName": "Admin",
        "lastName": "User",
        "image": "binary_image_data"
      },
      "messageData": [
        {
          "id": 1,
          "text": "Xin chào",
          "userId": 1
        }
      ]
    }
  ]
}
```

---

### 18.5. GET `/api/listRoomOfAdmin` (Admin Token Required)
**Output:**
```json
{
  "errCode": 0,
  "data": [
    {
      "id": 1,
      "userOne": 1,
      "userTwo": 2,
      "userOneData": {
        "id": 1,
        "firstName": "John",
        "lastName": "Doe",
        "image": "binary_image_data"
      },
      "userTwoData": {
        "id": 2,
        "firstName": "Admin",
        "lastName": "User",
        "image": "binary_image_data"
      },
      "messageData": [...]
    }
  ]
}
```

---

## Lưu ý chung

1. **Status Code**: Tất cả API trả về HTTP 200, kiểm tra `errCode` để xác định thành công/lỗi
2. **Token**: Các API yêu cầu token sẽ kiểm tra qua header `Authorization: Bearer <token>`
3. **Image**: Các trường image trả về dạng binary string (đã decode từ base64)
4. **Pagination**: Các API phân trang sử dụng `limit` và `offset` trong query string
5. **Error Handling**: Khi có lỗi server, tất cả API đều trả về:
   ```json
   {
     "errCode": -1,
     "errMessage": "Error from server"
   }
   ```

