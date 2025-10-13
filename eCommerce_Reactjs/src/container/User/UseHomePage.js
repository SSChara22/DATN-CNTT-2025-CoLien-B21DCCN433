import React, { useState, useEffect } from "react";
import { Route, Routes } from "react-router-dom";
import DetailUserPage from "./DetailUserPage";
import CategoryUser from "./CategoryUser";
import StoreVoucher from "./StoreVoucher";
import AddressUser from "./AddressUser";
import ChangePassword from "../System/User/ChangePassword";
import OrderUser from "./OrderUser";
import MessagePage from "../Message/MessagePage";

function UserHomePage(props) {
    const [user, setUser] = useState({});

    useEffect(() => {
        const userData = JSON.parse(localStorage.getItem("userData"));
        setUser(userData);
    }, []);

    return (
        <div
            style={{ display: "flex" }}
            className="container rounded bg-white mt-5 mb-5"
        >
            <Routes>
                <Route path="messenger" element={<MessagePage />} />
                <Route path="detail/:id" element={<DetailUserPage />} />
                <Route
                    path="store-voucher/:id"
                    element={<StoreVoucher id={user.id} />}
                />
                <Route
                    path="address/:id"
                    element={<AddressUser id={user.id} />}
                />
                <Route path="order/:id" element={<OrderUser id={user.id} />} />
                <Route
                    path="changepassword/:id"
                    element={<ChangePassword id={user.id} />}
                />
            </Routes>
            <CategoryUser id={user.id} />
        </div>
    );
}

export default UserHomePage;
