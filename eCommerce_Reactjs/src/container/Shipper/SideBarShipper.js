import React from 'react';
import { useEffect, useState } from 'react';
import { Link } from "react-router-dom";

const SideBarShipper = () => {
    const [user, setUser] = useState({})

    useEffect(() => {
        const userData = JSON.parse(localStorage.getItem('userData'));
        setUser(userData)
    }, [])

    return (
        <div id="layoutSidenav_nav">
            <nav className="sb-sidenav accordion sb-sidenav-dark" id="sidenavAccordion">
                <div className="sb-sidenav-menu">
                    <div className="nav">
                        <div className="sb-sidenav-menu-heading">Shipper</div>
                        <Link to="/shipper" className="nav-link">
                            <div className="sb-nav-link-icon"><i className="fas fa-tachometer-alt" /></div>
                            Trang chủ
                        </Link>
                        <Link to="/shipper" className="nav-link">
                            <div className="sb-nav-link-icon"><i className="fas fa-shipping-fast" /></div>
                            Quản lý đơn hàng
                        </Link>
                        <Link to="/shipper/chat" className="nav-link">
                            <div className="sb-nav-link-icon"><i className="fas fa-comments" /></div>
                            Chat với Admin
                        </Link>
                    </div>
                </div>
                <div className="sb-sidenav-footer">
                    <div className="small">Đăng nhập với tư cách:</div>
                    {user && `${user.firstName} ${user.lastName}`}
                </div>
            </nav>
        </div>
    );
}

export default SideBarShipper;


