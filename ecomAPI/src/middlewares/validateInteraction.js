// middleware/validateInteraction.js
export const validateLogInteraction = (req, res, next) => {
    const { userId, productId, actionCode, device } = req.body;
    const validActions = ['view', 'cart', 'purchase'];

    if (!userId || !productId || !actionCode) {
        return res.status(400).json({ success: false, message: 'Thiếu userId, productId hoặc actionCode' });
    }

    if (!validActions.includes(actionCode)) {
        return res.status(400).json({ success: false, message: 'actionCode không hợp lệ. Chỉ chấp nhận: view, cart, purchase' });
    }

    next();
};