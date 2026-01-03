import React from "react";
import { Route, Routes } from "react-router-dom";
import Footer from "../System/Footer";
import Header from "../System/Header";
import SideBarShipper from "./SideBarShipper";
import ManageShipperOrder from "./ManageShipperOrder";
import DetailShipperOrder from "./DetailShipperOrder";

// Common Layout for Shipper
const ShipperLayout = ({ children }) => (
    <div className="sb-nav-fixed">
        <Header />
        <div id="layoutSidenav">
            <SideBarShipper />
            <div id="layoutSidenav_content">
                <main>{children}</main>
                <Footer />
            </div>
        </div>
    </div>
);

function HomePageShipper() {
    return (
        <ShipperLayout>
            <Routes>
                <Route path="/" element={<ManageShipperOrder />} />
                <Route path="/order-detail/:id" element={<DetailShipperOrder />} />
            </Routes>
        </ShipperLayout>
    );
}

export default HomePageShipper;


