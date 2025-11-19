import React, { useState, useEffect } from 'react';
import HomeBanner from "../../component/HomeFeature/HomeBanner";
import MainFeature from "../../component/HomeFeature/MainFeature";
import ProductFeature from "../../component/HomeFeature/ProductFeature";
import NewProductFeature from "../../component/HomeFeature/NewProductFeature"
import HomeBlog from '../../component/HomeFeature/HomeBlog';
import ItemProduct from '../../component/Product/ItemProduct';
import { getAllBanner, getProductFeatureService, getProductNewService, getNewBlog, getProductRecommendService } from '../../services/userService';
import Slider from 'react-slick';
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";
function HomePage(props) {
    const [dataProductFeature, setDataProductFeature] = useState([])
    const [dataNewProductFeature, setNewProductFeature] = useState([])
    const [dataNewBlog, setdataNewBlog] = useState([])
    const [dataBanner, setdataBanner] = useState([])
    const [dataProductRecommend, setdataProductRecommend] = useState([])
    let settings = {
        dots: false,
        Infinity: false,
        speed: 500,
        slidesToShow: 1,
        slidesToScroll: 1,
        autoplaySpeed: 2000,
        autoplay: true,
        cssEase: "linear"
    }

    useEffect(() => {
        const userData = JSON.parse(localStorage.getItem('userData'));
        if (userData) {
            fetchProductRecommend(userData.id)
        }
        fetchBlogFeature()
        fetchDataBrand()
        fetchProductFeature()
        fetchProductNew()

        window.scrollTo(0, 0);
    }, [])
    let fetchBlogFeature = async () => {
        let res = await getNewBlog(3)
        if (res && res.errCode === 0) {
            setdataNewBlog(res.data)
        }
    }
    let fetchProductFeature = async () => {
        let res = await getProductFeatureService(6)
        if (res && res.errCode === 0) {
            setDataProductFeature(res.data)
        }
    }
    let fetchProductRecommend = async (userId) => {
        let res = await getProductRecommendService({
            limit: 5,
            userId: userId
        })
        if (res && res.errCode === 0) {
            setdataProductRecommend(res.data)
        }
    }
    let fetchDataBrand = async () => {
        let res = await getAllBanner({
            limit: 6,
            offset: 0,
            keyword: ''
        })
        if (res && res.errCode === 0) {
            setdataBanner(res.data)
        }
    }
    let fetchProductNew = async () => {
        let res = await getProductNewService(8)
        if (res && res.errCode === 0) {
            setNewProductFeature(res.data)
        }
    }
    return (
        <div>
            <Slider {...settings}>
                {dataBanner && dataBanner.length > 0 &&
                    dataBanner.map((item, index) => {
                        return (
                            <HomeBanner image={item.image} name={item.name}></HomeBanner>
                        )
                    })
                }


            </Slider>


            <MainFeature></MainFeature>
            {/* Gợi ý sản phẩm - render 5 items without slider */}
            <section className="feature_product_area section_gap_bottom_custom">
                <div className="container">
                    <div className="row" style={{marginBottom: '16px'}}>
                        <div className="col-12"><h3>Gợi ý sản phẩm</h3></div>
                    </div>
                    <div className="row box-productFeature">
                        {dataProductRecommend && dataProductRecommend.slice(0,5).map((item, index) => (
                            <ItemProduct
                                id={item.id}
                                key={index}
                                width={350}
                                height={419}
                                type="col-lg-4 col-md-6"
                                name={item.name}
                                img={item?.productDetail?.[0]?.productImage?.[0]?.image}
                                price={item?.productDetail?.[0]?.originalPrice}
                                discountPrice={item?.productDetail?.[0]?.discountPrice}
                            />
                        ))}
                    </div>
                </div>
            </section>
            <ProductFeature title={"Sản phẩm đặc trưng"} data={dataProductFeature}></ProductFeature>
            <NewProductFeature title="Sản phẩm mới" description="Những sản phẩm vừa ra mắt mới lạ cuốn hút người xem" data={dataNewProductFeature}></NewProductFeature>
            <HomeBlog data={dataNewBlog} />
        </div>
    );
}

export default HomePage;