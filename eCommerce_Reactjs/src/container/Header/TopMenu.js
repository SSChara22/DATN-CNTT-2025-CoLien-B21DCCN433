import React from "react";
import { clearRecommendationsService } from "../../services/userService";
import { NavLink } from "react-router-dom";
import "./Header.scss";
const TopMenu = (props) => {
    let handleLogout = async () => {
        try { await clearRecommendationsService(); } catch (e) {}
        localStorage.removeItem("userData");
        localStorage.removeItem("token");
        window.location.href = "/login";
    };

    let name =
        props.user && props.user.id
            ? `${
                  props.user && props.user.firstName ? props.user.firstName : ""
              } ${props.user.lastName}`
            : "";
    return (
        <div className="top_menu">
            <div className="container">
                <div className="row">
                    <div className="col-lg-7">
                        <div className="float-left">
                            <p>Điện thoại: 0123456789 </p>
                            <p>email: ptitshop@gmail.com</p>
                        </div>
                    </div>
                    <div className="col-lg-5">
                        <div className="float-right">
                            <ul className="right_side">
                                <li>
                                    {props.user && props.user.id ? (
                                        <NavLink
                                            exact
                                            to={`/user/detail/${
                                                props.user && props.user.id
                                                    ? props.user.id
                                                    : ""
                                            }`}
                                        >
                                            {name}
                                        </NavLink>
                                    ) : (
                                        <a href="/login">Đăng nhập</a>
                                    )}
                                </li>
                                <li style={{ cursor: "pointer" }}>
                                    {props.user && props.user.id ? (
                                        <a onClick={() => handleLogout()}>
                                            Đăng xuất
                                        </a>
                                    ) : (
                                        <a href="/login">Đăng ký</a>
                                    )}
                                </li>
                                <li>
                                    <a>VI</a>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default TopMenu;
