import db from "../models/index";
import { Op } from 'sequelize';

// Short-term in-memory de-duplication for messages (per Node process)
const _recentMsgKeys = new Map();
function _msgKey(roomId, userId, text) {
    const t = (text || '').trim().slice(0, 300);
    return `${roomId}:${userId}:${t}`;
}
function _isRecent(key) {
    const exp = _recentMsgKeys.get(key);
    const now = Date.now();
    if (exp && exp > now) return true;
    if (exp) _recentMsgKeys.delete(key);
    return false;
}
function _markRecent(key, ttlMs = 4000) {
    _recentMsgKeys.set(key, Date.now() + ttlMs);
}

let createNewRoom = (data) => {
    return new Promise(async (resolve, reject) => {
        try {
            if (!data.userId1) {
                resolve({
                    errCode: 1,
                    errMessage: 'Missing required parameters !'
                })
            } else {
                let userAdmin = await db.User.findOne({
                    where:{email:'chat@gmail.com'}
                })
                let room = await db.RoomMessage.findOne({where:{userOne:data.userId1}})
                if(room){
                    resolve({
                        errCode: 2,
                        errMessage: 'Da Co Phong'
                    })
                }else{
                    if(userAdmin){
                        let res = await db.RoomMessage.create({
                            userOne:data.userId1,
                            userTwo:userAdmin.id
                        })
                        if(res){
                            resolve({
                                errCode: 0,
                                errMessage: 'ok'
                            })
                        } else {
                            resolve({
                                errCode: -1,
                                errMessage: 'Failed to create room'
                            })
                        }
                    } else {
                        resolve({
                            errCode: 3,
                            errMessage: 'Admin user not found. Please create user with email: chat@gmail.com'
                        })
                    }
                }
            }
        } catch (error) {
            reject(error)
        }
    })
}

let sendMessage = (data) => {
    return new Promise(async (resolve, reject) => {
        try {
            const userId = data.userId;
            const roomId = data.roomId;
            const textRaw = (data.text || '').toString();
            const text = textRaw.trim();
            if (!userId || !roomId || !text) {
                resolve({
                    errCode: 1,
                    errMessage: 'Missing required parameters !'
                })
            } else {
                const key = _msgKey(roomId, userId, text);
                if (_isRecent(key)) {
                    return resolve({ errCode: 0, errMessage: 'ok', deduped: true });
                }

                // DB-side dedupe (avoid double-submit within short window)
                const fiveSecAgo = new Date(Date.now() - 5000);
                const existing = await db.Message.findOne({
                    where: {
                        roomId: roomId,
                        userId: userId,
                        text: text,
                        createdAt: { [Op.gte]: fiveSecAgo }
                    },
                    order: [['createdAt','DESC']]
                })
                if (existing) {
                    _markRecent(key);
                    return resolve({ errCode: 0, errMessage: 'ok', deduped: true });
                }

                let res = await db.Message.create({
                    text:text,
                    userId:userId,
                    roomId:roomId,
                    unRead:true
                })
                _markRecent(key);
                if(res){
                    return resolve({
                        errCode: 0,
                        errMessage: 'ok'
                    })
                }
                return resolve({ errCode: -1, errMessage: 'Failed to create message' })
            }
        } catch (error) {
            reject(error)
        }
    })
}

let loadMessage = (data) => {
    return new Promise(async (resolve, reject) => {
        try {
            if (!data.roomId) {
                resolve({
                    errCode: 1,
                    errMessage: 'Missing required parameters !'
                })
            } else {
                await db.Message.update({
                    unRead:false
                },{where:{roomId:data.roomId,
                    userId:{[Op.not]:data.userId}
                }})

                let message = await db.Message.findAll({
                    where:{roomId:data.roomId},
                    order: [['createdAt','ASC'], ['id','ASC']]
                })

                for(let i =0 ; i< message.length; i++){
                    message[i].userData = await db.User.findOne({where:{id:message[i].userId}})
                    if(message[i].userData.image){
                        message[i].userData.image = new Buffer(message[i].userData.image, 'base64').toString('binary');
                    }
                }
                resolve({
                    errCode: 0,
                    data: message
                })
            }
        } catch (error) {
            reject(error)
        }
    })
}

let listRoomOfUser = (userId) => {
    return new Promise(async (resolve, reject) => {
        try {
            if (!userId) {
                resolve({
                    errCode: 1,
                    errMessage: 'Missing required parameters !'
                })
            } else {
                let room = await db.RoomMessage.findAll({
                    where:{userOne:userId}
                })

                for(let i =0 ; i< room.length; i++){
                    room[i].messageData = await db.Message.findAll({where:{roomId:room[i].id}, order: [['createdAt','ASC'], ['id','ASC']]})

                    room[i].userOneData = await db.User.findOne({where:{id:room[i].userOne}})
                    if(room[i].userOneData.image){
                        room[i].userOneData.image = new Buffer(room[i].userOneData.image, 'base64').toString('binary');
                    }
                    room[i].userTwoData = await db.User.findOne({where:{id:room[i].userTwo}})
                    if(room[i].userTwoData.image){
                        room[i].userTwoData.image = new Buffer(room[i].userTwoData.image, 'base64').toString('binary');
                    }
                }
                resolve({
                    errCode: 0,
                    data: room
                })
            }
        } catch (error) {
            reject(error)
        }
    })
}

let listRoomOfAdmin = () => {
    return new Promise(async (resolve, reject) => {
        try {
            let user = await db.User.findOne({where:{email:'chat@gmail.com'}})
            if(user){
                let room = await db.RoomMessage.findAll({
                    where:{userTwo:user.id}
                  })
                 for(let i =0 ; i< room.length; i++){
                    room[i].messageData = await db.Message.findAll({where:{roomId:room[i].id}, order: [['createdAt','ASC'], ['id','ASC']]})
                    room[i].userOneData = await db.User.findOne({where:{id:room[i].userOne}})
                    if(room[i].userOneData.image){
                        room[i].userOneData.image = new Buffer(room[i].userOneData.image, 'base64').toString('binary');
                    }
                    room[i].userTwoData = await db.User.findOne({where:{id:room[i].userTwo}})
                    if(room[i].userTwoData.image){
                        room[i].userTwoData.image = new Buffer(room[i].userTwoData.image, 'base64').toString('binary');
                    }
                 }
                resolve({
                    errCode: 0,
                    data: room
                })
            }
              
          
        } catch (error) {
            reject(error)
        }
    })
}
module.exports = {
    createNewRoom: createNewRoom,
    sendMessage:sendMessage,
    loadMessage:loadMessage,
    listRoomOfUser:listRoomOfUser,
    listRoomOfAdmin:listRoomOfAdmin
}