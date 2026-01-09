import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  getDetailOrder,
  updateStatusOrderService,
  updateStatusOrderShipperService,
  updateImageOrderService,
} from "../../services/userService";
import { toast } from "react-toastify";
import CommonUtils from "../../utils/CommonUtils";
import ShopCartItem from "../../component/ShopCart/ShopCartItem";

function DetailShipperOrder() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [DataOrder, setDataOrder] = useState({});
  const [user, setUser] = useState(null);
  let price = 0;
  const [priceShip, setpriceShip] = useState(0);
  const [image, setImage] = useState("");

  useEffect(() => {
    const userData = JSON.parse(localStorage.getItem("userData"));
    setUser(userData);
    loadDataOrder();
  }, []);

  let loadDataOrder = () => {
    if (id) {
      let fetchOrder = async () => {
        let order = await getDetailOrder(id);
        if (order && order.errCode == 0) {
          setDataOrder(order.data);
          setpriceShip(order.data.typeShipData.price);
        }
      };
      fetchOrder();
    }
  };

  let totalPriceDiscount = (price, discount) => {
    try {
      if (discount.typeVoucherOfVoucherData.typeVoucher === "percent") {
        if (
          (price * discount.typeVoucherOfVoucherData.value) / 100 >
          discount.typeVoucherOfVoucherData.maxValue
        ) {
          return price - discount.typeVoucherOfVoucherData.maxValue;
        } else {
          return (
            price - (price * discount.typeVoucherOfVoucherData.value) / 100
          );
        }
      } else {
        return price - discount.typeVoucherOfVoucherData.maxValue;
      }
    } catch (error) {}
  };

  let handleConfirmOrder = async () => {
    let res = await updateStatusOrderShipperService({
      id: DataOrder.id,
      statusId: "S5", // Đang giao hàng
      shipperId: user?.id, // Gán shipper vào đơn hàng
    });
    if (res && res.errCode == 0) {
      toast.success("Xác nhận nhận đơn thành công");
      loadDataOrder();
    } else {
      toast.error(res.errMessage || "Có lỗi xảy ra");
    }
  };

  const handleOnChangeImage = async (e) => {
    let file = e.target.files[0];
    if (file) {
      let base64 = await CommonUtils.getBase64(file);
      setImage(base64);
    }
  };

  let handleCompleteOrder = async () => {
    if (!image) {
      toast.error("Vui lòng upload ảnh xác nhận giao hàng");
      return;
    }
    const resImage = await updateImageOrderService({ id: DataOrder.id, image });
    if (!resImage || resImage.errCode !== 0) {
      toast.error("Upload ảnh giao hàng thất bại");
      return;
    }
    toast.success("Đã gửi bằng chứng giao hàng. Chờ admin xác nhận hoàn tất.");
    navigate("/shipper");
  };

  return (
    <div className="container-fluid px-4">
      <h1 className="mt-4">Chi tiết đơn hàng #{id}</h1>

      <div className="card mb-4">
        <div className="card-header">
          <i className="fas fa-info-circle me-1" />
          Thông tin đơn hàng
        </div>
        <div className="card-body">
          {DataOrder && DataOrder.addressUser && (
            <div className="mb-3">
              <h5>Địa chỉ nhận hàng:</h5>
              <p>
                <strong>Tên người nhận:</strong>{" "}
                {DataOrder.addressUser.shipName}
              </p>
              <p>
                <strong>Số điện thoại:</strong>{" "}
                {DataOrder.addressUser.shipPhonenumber}
              </p>
              <p>
                <strong>Email:</strong> {DataOrder.addressUser.shipEmail}
              </p>
              <p>
                <strong>Địa chỉ:</strong> {DataOrder.addressUser.shipAdress}
              </p>
            </div>
          )}

          {DataOrder && DataOrder.note && (
            <div className="mb-3">
              <h5>Ghi chú:</h5>
              <p>{DataOrder.note}</p>
            </div>
          )}

          <div className="mb-3">
            <h5>Phương thức thanh toán:</h5>
            <p>
              {DataOrder.isPaymentOnlien == 0
                ? "Thanh toán tiền mặt"
                : "Thanh toán online"}
            </p>
          </div>

          <div className="mb-3">
            <h5>Trạng thái đơn hàng:</h5>
            <span
              className={`badge ${
                DataOrder.statusOrderData?.code === "S5"
                  ? "bg-warning"
                  : DataOrder.statusOrderData?.code === "S6"
                  ? "bg-success"
                  : "bg-secondary"
              }`}
            >
              {DataOrder.statusOrderData
                ? DataOrder.statusOrderData.value
                : "N/A"}
            </span>
          </div>

          {DataOrder.statusOrderData?.code === "S6" && DataOrder.image && (
            <div className="mt-4">
              <h5 className="text-success">
                <i className="fas fa-camera me-2" />
                Ảnh xác nhận giao hàng
              </h5>

              <img
                src={DataOrder.image}
                alt="Ảnh giao hàng"
                style={{
                  maxWidth: "350px",
                  borderRadius: "8px",
                  border: "1px solid #ddd",
                  marginTop: "10px",
                }}
              />
            </div>
          )}
        </div>
      </div>

      <div className="card mb-4">
        <div className="card-header">
          <i className="fas fa-shopping-cart me-1" />
          Sản phẩm trong đơn hàng
        </div>
        <div className="card-body">
          <div className="table-responsive">
            <table className="table">
              <thead>
                <tr>
                  <th scope="col">Sản phẩm</th>
                  <th scope="col">Giá</th>
                  <th style={{ textAlign: "center" }} scope="col">
                    Số lượng
                  </th>
                  <th style={{ textAlign: "center" }} scope="col">
                    Tổng tiền
                  </th>
                </tr>
              </thead>
              <tbody>
                {DataOrder.orderDetail &&
                  DataOrder.orderDetail.length > 0 &&
                  DataOrder.orderDetail.map((item, index) => {
                    price += item.quantity * item.productDetail.discountPrice;

                    let name = `${item.product.name} - ${item.productDetail.nameDetail} - ${item.productDetailSize.sizeData.value}`;
                    return (
                      <ShopCartItem
                        isOrder={true}
                        id={item.id}
                        productdetailsizeId={item.productDetailSize.id}
                        key={index}
                        name={name}
                        price={item.productDetail.discountPrice}
                        quantity={item.quantity}
                        image={item.productImage[0].image}
                      />
                    );
                  })}
              </tbody>
            </table>
          </div>

          <div className="mt-3">
            <div className="d-flex justify-content-between align-items-center">
              <div>
                <h6>Đơn vị vận chuyển:</h6>
                {DataOrder && DataOrder.typeShipData && (
                  <p>
                    {DataOrder.typeShipData.type} -{" "}
                    {CommonUtils.formatter.format(DataOrder.typeShipData.price)}
                  </p>
                )}
              </div>
              <div className="text-end">
                <h6>Tổng thanh toán:</h6>
                <h4 className="text-primary">
                  {DataOrder && DataOrder.voucherData && DataOrder.voucherId
                    ? CommonUtils.formatter.format(
                        totalPriceDiscount(price, DataOrder.voucherData) +
                          priceShip
                      )
                    : CommonUtils.formatter.format(price + +priceShip)}
                </h4>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="card mb-4">
        {DataOrder.statusOrderData?.code === "S5" && !DataOrder.image && (
          <div className="mb-3">
            <label className="form-label text-danger">
              Ảnh xác nhận giao hàng (bắt buộc)
            </label>
            <input
              type="file"
              className="form-control"
              accept="image/*"
              onChange={handleOnChangeImage}
            />
          </div>
        )}
        <div className="card-body">
          <div className="d-flex gap-2">
            {(DataOrder.statusOrderData?.code === "S3" ||
              DataOrder.statusOrderData?.code === "S4") &&
              !DataOrder.shipperId && (
                <button
                  className="btn btn-primary"
                  onClick={handleConfirmOrder}
                >
                  <i className="fas fa-check me-2" />
                  Nhận đơn hàng
                </button>
              )}
            {DataOrder.statusOrderData?.code === "S4" &&
              DataOrder.shipperId === user?.id && (
                <button
                  className="btn btn-warning"
                  onClick={handleConfirmOrder}
                >
                  <i className="fas fa-shipping-fast me-2" />
                  Bắt đầu giao hàng
                </button>
              )}

            {DataOrder.statusOrderData?.code === "S5" &&
              DataOrder.shipperId === user?.id &&
              !DataOrder.image && (
                <button
                  className="btn btn-success"
                  onClick={handleCompleteOrder}
                >
                  <i className="fas fa-check-circle me-2" />
                  Hoàn thành giao hàng
                </button>
              )}
            <button
              className="btn btn-secondary"
              onClick={() => navigate("/shipper")}
            >
              <i className="fas fa-arrow-left me-2" />
              Quay lại
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default DetailShipperOrder;
