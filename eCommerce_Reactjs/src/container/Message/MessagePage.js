import React, { useEffect, useState, useRef } from "react";
import { useParams } from "react-router-dom";
import ChatWindow from "./ChatWindow";
import MessageDisscution from "./MessageDisscution";
import "./MessagePage.scss";
import { createNewRoom, listRoomOfUser } from "../../services/userService";
import socketIOClient from "socket.io-client";

function MessagePage(props) {
    const [dataRoom, setdataRoom] = useState([]);
    const [selectedRoom, setselectedRoom] = useState("");
    const [dataUser, setdataUser] = useState({});
    const host = process.env.REACT_APP_BACKEND_URL;
    const socketRef = useRef();
    const [id, setId] = useState();
    useEffect(() => {
        socketRef.current = socketIOClient.connect(host);
        const userData = JSON.parse(localStorage.getItem("userData"));
        setdataUser(userData);
        let createRoom = async () => {
            let res = await createNewRoom({
                userId1: userData.id,
            });
            // Fetch list room cả khi đã có phòng (errCode === 2) hoặc tạo thành công (errCode === 0)
            if (res && (res.errCode === 0 || res.errCode === 2)) {
                fetchListRoom(userData.id);
            } else if (res && res.errCode === 3) {
                console.error("Admin user not found. Please create user with email: chat@gmail.com");
                alert("Hệ thống chat chưa sẵn sàng. Vui lòng liên hệ admin.");
            }
        };
        if (userData) {
            socketRef.current.on("getId", (data) => {
                setId(data);
            }); // phần này đơn giản để gán id cho mỗi phiên kết nối vào page. Mục đích chính là để phân biệt đoạn nào là của mình đang chat.
            createRoom();

            fetchListRoom(userData.id);

            socketRef.current.on("sendDataServer", (dataGot) => {
                fetchListRoom(userData.id);
            });
            socketRef.current.on("loadRoomServer", (dataGot) => {
                fetchListRoom(userData.id);
            });
            return () => {
                socketRef.current.disconnect();
            };
        }
    }, []);
    let handleClickRoom = (roomId) => {
        socketRef.current.emit("loadRoomClient");
        setselectedRoom(roomId);
    };
    let fetchListRoom = async (userId) => {
        let res = await listRoomOfUser(userId);
        if (res && res.errCode == 0) {
            setdataRoom(res.data);
        }
    };
    return (
        <div className="container">
            <div className="ks-page-content">
                <div className="ks-page-content-body">
                    <div className="ks-messenger">
                        <MessageDisscution
                            userId={dataUser.id}
                            isAdmin={false}
                            handleClickRoom={handleClickRoom}
                            data={dataRoom}
                        />
                        {selectedRoom ? (
                            <ChatWindow
                                userId={dataUser.id}
                                roomId={selectedRoom}
                            />
                        ) : (
                            <div>
                                <span className="title">Chưa chọn phòng</span>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

export default MessagePage;
