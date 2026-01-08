import React, { useState } from "react";
import { Route, Routes } from "react-router-dom";
import Footer from "../System/Footer";
import Header from "../System/Header";
import SideBarShipper from "./SideBarShipper";
import ManageShipperOrder from "./ManageShipperOrder";
import DetailShipperOrder from "./DetailShipperOrder";
import ShipperMessagePage from "./ShipperMessagePage";

// Common Layout for Shipper
const ShipperLayout = ({ children }) => {
    const [isToggled, setIsToggled] = useState(false);

    return (
        <div className={`sb-nav-fixed ${isToggled ? "sb-sidenav-toggled" : ""}`}>
            <Header onToggle={() => setIsToggled(!isToggled)} />
            <div id="layoutSidenav">
                <SideBarShipper />
                <div id="layoutSidenav_content">
                    <main>{children}</main>
                    <Footer />
                </div>
            </div>
        </div>
    );
};

function HomePageShipper() {
    return (
        <ShipperLayout>
            <Routes>
                <Route path="/" element={<ManageShipperOrder />} />
                <Route path="/order-detail/:id" element={<DetailShipperOrder />} />
                <Route path="/chat" element={<ShipperMessagePage />} />
            </Routes>
        </ShipperLayout>
    );
}

export default HomePageShipper;


