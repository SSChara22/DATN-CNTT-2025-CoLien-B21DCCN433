import React, { useEffect, useState, useRef } from "react";
import MessageDisscution from "../Message/MessageDisscution";
import ShipperChatWindow from "./ShipperChatWindow";
import socketIOClient from "socket.io-client";
import { createNewRoomShipper, listRoomOfShipper } from "../../services/userService";

const host = process.env.REACT_APP_BACKEND_URL;

function ShipperMessagePage() {
  const [dataRoom, setdataRoom] = useState([]);
  const [selectedRoom, setselectedRoom] = useState("");
  const [dataUser, setdataUser] = useState({});
  const socketRef = useRef();
  const [id, setId] = useState();

  useEffect(() => {
    socketRef.current = socketIOClient.connect(host);
    const userData = JSON.parse(localStorage.getItem("userData"));
    setdataUser(userData);

    const init = async () => {
      if (!userData) return;
      // ensure room exists for shipper with admin
      const res = await createNewRoomShipper({ userId1: userData.id });
      // regardless of create result, fetch list
      await fetchListRoom(userData.id);
    };

    socketRef.current.on("getId", (data) => setId(data));
    init();

    socketRef.current.on("sendDataServer", () => {
      if (dataUser?.id) fetchListRoom(dataUser.id);
    });
    socketRef.current.on("loadRoomServer", () => {
      if (dataUser?.id) fetchListRoom(dataUser.id);
    });

    return () => {
      socketRef.current.disconnect();
    };
  }, []);

  const handleClickRoom = (roomId) => {
    if (socketRef.current) socketRef.current.emit("loadRoomClient");
    setselectedRoom(roomId);
  };

  const fetchListRoom = async (userId) => {
    const res = await listRoomOfShipper(userId);
    if (res && res.errCode === 0) {
      setdataRoom(res.data || []);
    }
  };

  return (
    <div className="container-fluid">
      <div className="ks-page-content">
        <div className="ks-page-content-body">
          <div className="row g-3">
            <div className="col-12 col-lg-4">
              <MessageDisscution
                userId={dataUser?.id}
                isAdmin={false}
                handleClickRoom={handleClickRoom}
                data={dataRoom}
              />
            </div>
            <div className="col-12 col-lg-8">
              {selectedRoom ? (
                <ShipperChatWindow userId={dataUser?.id} roomId={selectedRoom} />
              ) : (
                <div className="d-flex align-items-center justify-content-center border rounded p-4 w-100" style={{minHeight: 300}}>
                  <span className="text-muted">Chưa chọn phòng</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ShipperMessagePage;
