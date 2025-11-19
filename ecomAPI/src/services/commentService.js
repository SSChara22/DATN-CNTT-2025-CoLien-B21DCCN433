import db from "../models/index";
require('dotenv').config();

/**
 * ================== REVIEW SẢN PHẨM ==================
 */

let createNewReview = (data) => {
    return new Promise(async (resolve, reject) => {
        try {
            if (!data.content || !data.productId || !data.userId || !data.star) {
                return resolve({
                    errCode: 1,
                    errMessage: 'Missing required parameter !'
                });
            }

            await db.Comment.create({
                content: data.content,
                productId: data.productId,
                userId: data.userId,
                star: data.star,
                image: data.image || null
            });

            return resolve({
                errCode: 0,
                errMessage: 'ok'
            });
        } catch (error) {
            console.log("createNewReview error:", error);
            return reject(error);
        }
    });
};

let getAllReviewByProductId = (id) => {
    return new Promise(async (resolve, reject) => {
        try {
            if (!id) {
                return resolve({
                    errCode: 1,
                    errMessage: 'Missing required parameter !'
                });
            }

            let res = await db.Comment.findAll({
                where: { productId: id },
                raw: true
            });

            if (res && res.length > 0) {
                for (let i = 0; i < res.length; i++) {
                    // ảnh của comment
                    res[i].image = res[i].image
                        ? Buffer.from(res[i].image, 'base64').toString('binary')
                        : '';

                    // comment con (reply)
                    res[i].childComment = await db.Comment.findAll({
                        where: { parentId: res[i].id },
                        raw: true
                    });

                    // user
                    let user = await db.User.findOne({
                        where: { id: res[i].userId },
                        attributes: {
                            exclude: ['password']
                        },
                        raw: true
                    });

                    if (user) {
                        user.image = user.image
                            ? Buffer.from(user.image, 'base64').toString('binary')
                            : '';
                        res[i].user = user;
                    } else {
                        res[i].user = null;
                    }
                }
            }

            return resolve({
                errCode: 0,
                data: res
            });
        } catch (error) {
            console.log("getAllReviewByProductId error:", error);
            return reject(error);
        }
    });
};

let ReplyReview = (data) => {
    return new Promise(async (resolve, reject) => {
        try {
            if (!data.content || !data.productId || !data.userId || !data.parentId) {
                return resolve({
                    errCode: 1,
                    errMessage: 'Missing required parameter !'
                });
            }

            await db.Comment.create({
                content: data.content,
                productId: data.productId,
                userId: data.userId,
                parentId: data.parentId
            });

            return resolve({
                errCode: 0,
                errMessage: 'ok'
            });
        } catch (error) {
            console.log("ReplyReview error:", error);
            return reject(error);
        }
    });
};

let deleteReview = (data) => {
    return new Promise(async (resolve, reject) => {
        try {
            if (!data.id) {
                return resolve({
                    errCode: 1,
                    errMessage: 'Missing required parameter !'
                });
            }

            let review = await db.Comment.findOne({
                where: { id: data.id }
            });

            if (review) {
                await db.Comment.destroy({ where: { id: data.id } });
                return resolve({
                    errCode: 0,
                    errMessage: 'ok'
                });
            } else {
                return resolve({
                    errCode: 2,
                    errMessage: 'Review not found !'
                });
            }
        } catch (error) {
            console.log("deleteReview error:", error);
            return reject(error);
        }
    });
};


/**
 * ================== COMMENT BLOG ==================
 */

let createNewComment = (data) => {
    return new Promise(async (resolve, reject) => {
        try {
            if (!data.content || !data.blogId || !data.userId) {
                return resolve({
                    errCode: 1,
                    errMessage: 'Missing required parameter !'
                });
            }

            await db.Comment.create({
                content: data.content,
                blogId: data.blogId,
                userId: data.userId,
                image: data.image || null
            });

            return resolve({
                errCode: 0,
                errMessage: 'ok'
            });
        } catch (error) {
            console.log("createNewComment error:", error);
            return reject(error);
        }
    });
};

let getAllCommentByBlogId = (id) => {
    return new Promise(async (resolve, reject) => {
        try {
            if (!id) {
                return resolve({
                    errCode: 1,
                    errMessage: 'Missing required parameter !'
                });
            }

            let res = await db.Comment.findAll({
                where: { blogId: id },
                order: [['createdAt', 'DESC']],
                raw: true
            });

            if (res && res.length > 0) {
                for (let i = 0; i < res.length; i++) {
                    res[i].image = res[i].image
                        ? Buffer.from(res[i].image, 'base64').toString('binary')
                        : '';

                    res[i].childComment = await db.Comment.findAll({
                        where: { parentId: res[i].id },
                        raw: true
                    });

                    let user = await db.User.findOne({
                        where: { id: res[i].userId },
                        attributes: { exclude: ['password'] },
                        raw: true
                    });

                    if (user) {
                        user.image = user.image
                            ? Buffer.from(user.image, 'base64').toString('binary')
                            : '';
                        res[i].user = user;
                    } else {
                        res[i].user = null;
                    }
                }
            }

            return resolve({
                errCode: 0,
                data: res
            });
        } catch (error) {
            console.log("getAllCommentByBlogId error:", error);
            return reject(error);
        }
    });
};

let ReplyComment = (data) => {
    return new Promise(async (resolve, reject) => {
        try {
            if (!data.content || !data.blogId || !data.userId || !data.parentId) {
                return resolve({
                    errCode: 1,
                    errMessage: 'Missing required parameter !'
                });
            }

            await db.Comment.create({
                content: data.content,
                blogId: data.blogId,
                userId: data.userId,
                parentId: data.parentId
            });

            return resolve({
                errCode: 0,
                errMessage: 'ok'
            });
        } catch (error) {
            console.log("ReplyComment error:", error);
            return reject(error);
        }
    });
};

let deleteComment = (data) => {
    return new Promise(async (resolve, reject) => {
        try {
            if (!data.id) {
                return resolve({
                    errCode: 1,
                    errMessage: 'Missing required parameter !'
                });
            }

            let comment = await db.Comment.findOne({
                where: { id: data.id }
            });

            if (comment) {
                await db.Comment.destroy({ where: { id: data.id } });
                return resolve({
                    errCode: 0,
                    errMessage: 'ok'
                });
            } else {
                return resolve({
                    errCode: 2,
                    errMessage: 'Comment not found !'
                });
            }
        } catch (error) {
            console.log("deleteComment error:", error);
            return reject(error);
        }
    });
};

module.exports = {
    createNewReview,
    getAllReviewByProductId,
    ReplyReview,
    deleteReview,
    createNewComment,
    getAllCommentByBlogId,
    deleteComment,
    ReplyComment
};
