import React, { useEffect, useState } from 'react';
import { toast } from 'react-toastify';
import {paymentOrderSuccessService} from '../../services/userService'
import {
    BrowserRouter as Router,
    Switch,
    Route,
    Link,
    Redirect,
    useParams,
    useLocation
} from "react-router-dom";
import useInteraction from '../../utils/useInteraction';


function useQuery() {
    const { search } = useLocation();

    return React.useMemo(() => new URLSearchParams(search), [search]);
  }

function PaymentSuccess(props) {
    let query = useQuery();
    const { trackPurchase } = useInteraction();

    useEffect(() => {
        let orderData =  JSON.parse(localStorage.getItem("orderData"))
        localStorage.removeItem("orderData")
        if(orderData){
            orderData.paymentId = query.get("paymentId")
            orderData.token = query.get("token")
            orderData.PayerID = query.get("PayerID")

            createNewOrder(orderData)
        }
    }, [])
    let createNewOrder = async (data) =>{
        let res = await paymentOrderSuccessService(data)
        if(res && res.errCode ==0){
            toast.success("Thanh toán hóa đơn thành công")

            // Track purchase interaction for each product in order
            // Note: We need to get productId from cart data stored separately
            // because arrDataShopCart only has productdetailsizeId
            const cartDataRaw = localStorage.getItem('cartData');
            if(data.userId && cartDataRaw) {
                try {
                    const cartData = JSON.parse(cartDataRaw);
                    if(cartData && cartData.length > 0) {
                        cartData.forEach(item => {
                            if(item.productData?.id) {
                                trackPurchase(data.userId, item.productData.id);
                            }
                        });
                    }
                    localStorage.removeItem('cartData');
                } catch (error) {
                    console.log('Error tracking purchase interactions:', error);
                }
            }

            const userData = JSON.parse(localStorage.getItem('userData'));
            setTimeout(()=>{
                window.location.href='/user/order/'+userData.id
            },2000)
        }else{
            toast.error(res.errMessgae)
        }
    }
    return (

        <div style={{height:'50vh',textAlign:'center'}}> 
          
        </div>

    );
}

export default PaymentSuccess;

