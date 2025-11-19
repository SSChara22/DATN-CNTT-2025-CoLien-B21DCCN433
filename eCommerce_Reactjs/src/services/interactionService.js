import axios from "../axios";

//==================INTERACTION==========================//

/**
 * Ghi lại interaction của user với product
 * @param {Object} data - { userId, productId, actionCode, device }
 * actionCode: 'view' | 'cart' | 'purchase'
 * device: 'desktop' | 'mobile' | 'tablet' (optional)
 */
const logInteraction = (data) => {
    return axios.post(`/api/interaction`, data);
}

/**
 * Lấy tất cả interaction của một user
 * @param {number} userId
 * @param {string} actionCode - optional filter: 'view' | 'cart' | 'purchase'
 */
const getUserInteractions = (userId, actionCode = null) => {
    const query = actionCode ? `?action=${actionCode}` : '';
    return axios.get(`/api/interaction/user/${userId}${query}`);
}

/**
 * Lấy tất cả interaction (admin only)
 * @param {string} actionCode - optional filter
 */
const getAllInteractions = (actionCode = null) => {
    const query = actionCode ? `?action=${actionCode}` : '';
    return axios.get(`/api/interaction${query}`);
}

/**
 * Xóa interaction
 * @param {Object} data - { userId, productId }
 */
const deleteInteraction = (data) => {
    return axios.delete(`/api/interaction`, { data });
}

export {
    logInteraction,
    getUserInteractions,
    getAllInteractions,
    deleteInteraction
}
