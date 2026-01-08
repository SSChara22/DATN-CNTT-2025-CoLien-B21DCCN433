import React, { useEffect, useState, useRef } from "react";
import socketIOClient from "socket.io-client";
import { loadMessageShipper, sendMessageShipper } from "../../services/userService";
import moment from "moment";

const host = process.env.REACT_APP_BACKEND_URL;
function ShipperChatWindow(props) {
    const [mess, setMess] = useState([]);
    const [message, setMessage] = useState("");
    const [id, setId] = useState();
    const [user, setUser] = useState({});
    const socketRef = useRef();

    useEffect(() => {
        socketRef.current = socketIOClient.connect(host);
        const userData = JSON.parse(localStorage.getItem("userData"));
        setUser(userData);

        socketRef.current.on("getId", (data) => {
            setId(data);
        });

        if (props.roomId) {
            fetchMessage();
        }

        socketRef.current.on("sendDataServer", () => {
            fetchMessage();
            let elem = document.getElementById("box-chat");
            if (elem) elem.scrollTop = elem.scrollHeight;
        });

        return () => {
            socketRef.current.disconnect();
        };
    }, [props.roomId]);

    let fetchMessage = async () => {
        let res = await loadMessageShipper(props.roomId, props.userId);
        if (res) {
            setMess(res.data);
        }
    };

    let sendMessage = async () => {
        if (message && message.trim() !== "" && props.roomId && user && user.id) {
            const msg = {
                text: message.trim(),
                userId: user.id,
                roomId: props.roomId,
            };
            try {
                const res = await sendMessageShipper(msg);
                if (res && res.errCode === 0) {
                    socketRef.current.emit("sendDataClient", msg);
                    setMessage("");
                } else {
                    console.error("Failed to send message:", res?.errMessage);
                    alert("Không thể gửi tin nhắn: " + (res?.errMessage || "Lỗi không xác định"));
                }
            } catch (error) {
                console.error("Error sending message:", error);
                socketRef.current.emit("sendDataClient", msg);
                setMessage("");
            }
        }
    };

    return (
        <div className="ks-messages ks-messenger__messages">
            <div className="ks-header">
                <div className="ks-description">
                    <div className="ks-name">Shipper Chat</div>
                    <div className="ks-amount">2 members</div>
                </div>
            </div>
            <div className="ks-body ks-scrollable jspScrollable" data-auto-height data-reduce-height=".ks-footer" data-fix-height={32} style={{height: "480px", overflow: "hidden", padding: "0px", width: "100%"}} tabIndex={0}>
                <div className="jspContainer" style={{ width: "100%", height: "481px" }}>
                    <div className="jspPane" style={{ padding: "0px", top: "0px", width: "100%" }}>
                        <ul id="box-chat" className="ks-items" style={{ overflowY: "scroll", maxHeight: "479px" }}>
                            {mess && mess.length > 0 && mess.map((item, index) => (
                                item.userData ? (
                                    <li key={index} className={item.userData.id == (user && user.id) ? "ks-item ks-from" : "ks-item ks-self"}>
                                        <span className="ks-avatar ks-offline">
                                            <img src={item.userData.image} width={36} height={36} className="rounded-circle" />
                                        </span>
                                        <div className="ks-body">
                                            <div className="ks-header">
                                                <span className="ks-name">{`${item.userData.firstName} ${item.userData.lastName}`}</span>
                                                <span className="ks-datetime">{moment(item.createdAt).fromNow()}</span>
                                            </div>
                                            <div className="ks-message">{item.text}</div>
                                        </div>
                                    </li>
                                ) : null
                            ))}
                        </ul>
                    </div>
                </div>
            </div>
            <div className="ks-footer">
                <textarea onChange={(e) => setMessage(e.target.value)} value={message} className="form-control" placeholder="Type something..." />
                <div className="ks-controls">
                    <button onClick={() => sendMessage()} className="btn btn-primary">Send</button>
                </div>
            </div>
        </div>
    );
}

export default ShipperChatWindow;
