import React from 'react';
import { useEffect, useState } from 'react';
import { getAllOrdersByShipper, updateStatusOrderService } from '../../services/userService';
import moment from 'moment';
import { toast } from 'react-toastify';
import { Link } from "react-router-dom";

const ManageShipperOrder = () => {
    const [dataOrder, setDataOrder] = useState([])
    const [statusFilter, setStatusFilter] = useState('all')
    const [user, setUser] = useState(null)

    useEffect(() => {
        const userData = JSON.parse(localStorage.getItem('userData'));
        setUser(userData)
        if (userData) {
            loadOrderData('all', userData.id)
        }
    }, [])

    let loadOrderData = async (status, shipperId) => {
        try {
            let params = {
                shipperId: shipperId
            }
            if (status === 'available') {
                params.status = 'available'
            } else if (status === 'working') {
                params.status = 'working'
            } else if (status === 'done') {
                params.status = 'done'
            }

            let arrData = await getAllOrdersByShipper(params)
            if (arrData && arrData.errCode === 0) {
                setDataOrder(arrData.data)
            }
        } catch (error) {
            console.log(error)
            toast.error('Lỗi khi tải dữ liệu đơn hàng')
        }
    }

    let handleOnchangeStatus = (event) => {
        const status = event.target.value
        setStatusFilter(status)
        if (user) {
            loadOrderData(status, user.id)
        }
    }

    let handleConfirmOrder = async (orderId) => {
        try {
            let res = await updateStatusOrderService({
                id: orderId,
                statusId: 'S5', // Đang giao hàng
                shipperId: user.id // Gán shipper vào đơn hàng
            })
            if (res && res.errCode === 0) {
                toast.success("Xác nhận nhận đơn thành công")
                if (user) {
                    loadOrderData(statusFilter, user.id)
                }
            } else {
                toast.error(res.errMessage || "Có lỗi xảy ra")
            }
        } catch (error) {
            toast.error('Lỗi khi xác nhận đơn hàng')
        }
    }

    let handleCompleteOrder = async (orderId) => {
        try {
            let res = await updateStatusOrderService({
                id: orderId,
                statusId: 'S6' // Đã giao hàng
            })
            if (res && res.errCode === 0) {
                toast.success("Xác nhận giao hàng thành công")
                if (user) {
                    loadOrderData(statusFilter, user.id)
                }
            } else {
                toast.error(res.errMessage || "Có lỗi xảy ra")
            }
        } catch (error) {
            toast.error('Lỗi khi xác nhận giao hàng')
        }
    }

    return (
        <div className="container-fluid px-4">
            <h1 className="mt-4">Quản lý đơn hàng giao</h1>

            <div className="card mb-4">
                <div className="card-header">
                    <i className="fas fa-table me-1" />
                    Danh sách đơn hàng
                </div>
                <div className="card-body p-3">
                    <select 
                        onChange={(event) => handleOnchangeStatus(event)} 
                        className="form-select col-3 mb-3"
                        value={statusFilter}
                    >
                        <option value="all">Tất cả đơn hàng</option>
                        <option value="available">Đơn hàng chưa có shipper</option>
                        <option value="working">Đang giao hàng</option>
                        <option value="done">Đã giao hàng</option>
                    </select>
                </div>
                <div className="card-body">
                    <div className="table-responsive">
                        <table className="table table-bordered" style={{ border: '1' }} width="100%" cellSpacing="0">
                            <thead>
                                <tr>
                                    <th>Mã đơn</th>
                                    <th>Khách hàng</th>
                                    <th>SDT</th>
                                    <th>Địa chỉ</th>
                                    <th>Ngày đặt</th>
                                    <th>Loại ship</th>
                                    <th>Trạng thái</th>
                                    <th>Thao tác</th>
                                </tr>
                            </thead>
                            <tbody>
                                {dataOrder && dataOrder.length > 0 ? (
                                    dataOrder.map((item, index) => {
                                        return (
                                            <tr key={index}>
                                                <td>{item.id}</td>
                                                <td>
                                                    {item.userData ? 
                                                        `${item.userData.firstName} ${item.userData.lastName}` 
                                                        : 'N/A'}
                                                </td>
                                                <td>
                                                    {item.addressUser ? item.addressUser.shipPhonenumber : 'N/A'}
                                                </td>
                                                <td>
                                                    {item.addressUser ? item.addressUser.shipAdress : 'N/A'}
                                                </td>
                                                <td>
                                                    {moment.utc(item.createdAt).local().format('DD/MM/YYYY HH:mm:ss')}
                                                </td>
                                                <td>{item.typeShipData ? item.typeShipData.type : 'N/A'}</td>
                                                <td>
                                                    <span className={`badge ${
                                                        item.statusOrderData?.code === 'S5' ? 'bg-warning' :
                                                        item.statusOrderData?.code === 'S6' ? 'bg-success' :
                                                        item.statusOrderData?.code === 'S3' || item.statusOrderData?.code === 'S4' ? 'bg-info' :
                                                        'bg-secondary'
                                                    }`}>
                                                        {item.statusOrderData ? item.statusOrderData.value : 'N/A'}
                                                    </span>
                                                    {!item.shipperId && (item.statusOrderData?.code === 'S3' || item.statusOrderData?.code === 'S4') && (
                                                        <span className="badge bg-danger ms-1">Chưa có shipper</span>
                                                    )}
                                                </td>
                                                <td>
                                                    <div className="btn-group" role="group">
                                                        <Link 
                                                            to={`/shipper/order-detail/${item.id}`}
                                                            className="btn btn-sm btn-info me-1"
                                                        >
                                                            Chi tiết
                                                        </Link>
                                                        {(item.statusOrderData?.code === 'S3' || item.statusOrderData?.code === 'S4') && !item.shipperId && (
                                                            <button
                                                                className="btn btn-sm btn-primary"
                                                                onClick={() => handleConfirmOrder(item.id)}
                                                            >
                                                                Nhận đơn hàng
                                                            </button>
                                                        )}
                                                        {item.statusOrderData?.code === 'S4' && item.shipperId === user?.id && (
                                                            <button
                                                                className="btn btn-sm btn-warning"
                                                                onClick={() => handleConfirmOrder(item.id)}
                                                            >
                                                                Bắt đầu giao hàng
                                                            </button>
                                                        )}
                                                        {item.statusOrderData?.code === 'S5' && (
                                                            <button
                                                                className="btn btn-sm btn-success"
                                                                onClick={() => handleCompleteOrder(item.id)}
                                                            >
                                                                Hoàn thành giao hàng
                                                            </button>
                                                        )}
                                                    </div>
                                                </td>
                                            </tr>
                                        )
                                    })
                                ) : (
                                    <tr>
                                        <td colSpan="8" className="text-center">
                                            Không có đơn hàng nào
                                        </td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default ManageShipperOrder;

