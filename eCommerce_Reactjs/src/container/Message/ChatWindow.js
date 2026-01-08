import React, { useEffect, useState, useRef } from "react";
import { useParams } from "react-router-dom";
import socketIOClient from "socket.io-client";
import { loadMessage, sendMessage as sendMessageAPI } from "../../services/userService";
import moment from "moment";

const host = process.env.REACT_APP_BACKEND_URL;
function ChatWindow(props) {
    const [mess, setMess] = useState([]);
    const [userData, setuserData] = useState({});
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
        }); // phần này đơn giản để gán id cho mỗi phiên kết nối vào page. Mục đích chính là để phân biệt đoạn nào là của mình đang chat.

        if (props.roomId) {
            fetchMessage();
        }

        socketRef.current.on("sendDataServer", (dataGot) => {
            fetchMessage();
            let elem = document.getElementById("box-chat");
            if (elem) elem.scrollTop = elem.scrollHeight;
        }); // mỗi khi có tin nhắn thì mess sẽ được render thêm

        return () => {
            socketRef.current.disconnect();
        };
    }, [props.roomId]);
    let fetchMessage = async () => {
        let res = await loadMessage(props.roomId, props.userId);
        if (res) {
            setMess(res.data);
            setuserData(res.data.userData);
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
                // Gọi API REST để lưu vào DB trước
                const res = await sendMessageAPI(msg);
                
                if (res && res.errCode === 0) {
                    // Sau khi lưu thành công, emit socket để real-time update
                    socketRef.current.emit("sendDataClient", msg);
                    setMessage("");
                } else {
                    console.error("Failed to send message:", res?.errMessage);
                    alert("Không thể gửi tin nhắn: " + (res?.errMessage || "Lỗi không xác định"));
                }
            } catch (error) {
                console.error("Error sending message:", error);
                // Fallback: vẫn emit socket nếu API lỗi (socket handler sẽ cố lưu)
                socketRef.current.emit("sendDataClient", msg);
                setMessage("");
            }
        }
    };
    return (
        <div className="ks-messages ks-messenger__messages">
            <div className="ks-header">
                <div className="ks-description">
                    <div className="ks-name">Chat name</div>
                    <div className="ks-amount">2 members</div>
                </div>
                <div className="ks-controls">
                    <div className="dropdown">
                        <button
                            className="btn btn-primary-outline ks-light ks-no-text ks-no-arrow"
                            type="button"
                            id="dropdownMenuButton"
                            data-toggle="dropdown"
                            aria-haspopup="true"
                            aria-expanded="false"
                        >
                            <span className="la la-ellipsis-h ks-icon" />
                        </button>
                        <div
                            className="dropdown-menu dropdown-menu-right ks-simple"
                            aria-labelledby="dropdownMenuButton"
                        >
                            <a className="dropdown-item" href="#">
                                <span className="la la-user-plus ks-icon" />
                                <span className="ks-text">Add members</span>
                            </a>
                            <a className="dropdown-item" href="#">
                                <span className="la la-eye-slash ks-icon" />
                                <span className="ks-text">Mark as unread</span>
                            </a>
                            <a className="dropdown-item" href="#">
                                <span className="la la-bell-slash-o ks-icon" />
                                <span className="ks-text">
                                    Mute notifications
                                </span>
                            </a>
                            <a className="dropdown-item" href="#">
                                <span className="la la-mail-forward ks-icon" />
                                <span className="ks-text">Forward</span>
                            </a>
                            <a className="dropdown-item" href="#">
                                <span className="la la-ban ks-icon" />
                                <span className="ks-text">Spam</span>
                            </a>
                            <a className="dropdown-item" href="#">
                                <span className="la la-trash-o ks-icon" />
                                <span className="ks-text">Delete</span>
                            </a>
                        </div>
                    </div>
                </div>
            </div>
            <div
                className="ks-body ks-scrollable jspScrollable"
                data-auto-height
                data-reduce-height=".ks-footer"
                data-fix-height={32}
                style={{
                    height: "60vh",
                    overflow: "hidden",
                    padding: "0px",
                    width: "100%",
                }}
                tabIndex={0}
            >
                <div
                    className="jspContainer"
                    style={{ width: "100%", height: "60vh" }}
                >
                    <div
                        className="jspPane"
                        style={{ padding: "0px", top: "0px", width: "100%" }}
                    >
                        <ul
                            id="box-chat"
                            className="ks-items"
                            style={{ overflowY: "scroll", maxHeight: "58vh" }}
                        >
                            {mess &&
                                mess.length > 0 &&
                                mess.map((item, index) => {
                                    if (item.userData) {
                                        return (
                                            <li
                                                key={index}
                                                className={
                                                    item.userData.id == user.id
                                                        ? "ks-item ks-from"
                                                        : "ks-item ks-self"
                                                }
                                            >
                                                <span className="ks-avatar ks-offline">
                                                    <img
                                                        src={
                                                            item.userData.image
                                                        }
                                                        width={36}
                                                        height={36}
                                                        className="rounded-circle"
                                                    />
                                                </span>
                                                <div className="ks-body">
                                                    <div className="ks-header">
                                                        <span className="ks-name">
                                                            {item.userData
                                                                .firstName +
                                                                " " +
                                                                item.userData
                                                                    .lastName}
                                                        </span>
                                                        <span className="ks-datetime">
                                                            {moment(
                                                                item.createdAt
                                                            ).fromNow()}
                                                        </span>
                                                    </div>
                                                    <div className="ks-message">
                                                        {item.text}
                                                    </div>
                                                </div>
                                            </li>
                                        );
                                    }
                                })}
                        </ul>
                    </div>
                    <div className="jspVerticalBar">
                        <div className="jspCap jspCapTop" />
                        <div className="jspTrack" style={{ height: "481px" }}>
                            <div
                                className="jspDrag"
                                style={{ height: "206px" }}
                            >
                                <div className="jspDragTop" />
                                <div className="jspDragBottom" />
                            </div>
                        </div>
                        <div className="jspCap jspCapBottom" />
                    </div>
                </div>
            </div>
            <div className="ks-footer">
                <textarea
                    onChange={(e) => setMessage(e.target.value)}
                    value={message}
                    className="form-control"
                    placeholder="Type something..."
                    defaultValue={""}
                />
                <div className="ks-controls">
                    <button
                        onClick={() => sendMessage()}
                        className="btn btn-primary"
                    >
                        Send
                    </button>
                </div>
            </div>
        </div>
    );
}

export default ChatWindow;
