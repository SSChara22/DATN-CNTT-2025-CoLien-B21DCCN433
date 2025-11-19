import { Interaction, Allcode, Product, User } from '../models/index.js';

export async function logInteraction(userId, productId, actionCode, device) {
    // Kiểm tra actionCode có tồn tại trong Allcode
    const action = await Allcode.findOne({ where: { type: 'ACTION', code: actionCode } });
    if (!action) throw new Error(`Action code "${actionCode}" không tồn tại`);

    // Kiểm tra userId, productId tồn tại
    const user = await User.findByPk(userId);
    const product = await Product.findByPk(productId);
    if (!user || !product) {
        throw new Error('User hoặc Product không tồn tại');
    }
    // Kiểm tra device_type hợp lệ nếu được cung cấp
    const validDevices = ['desktop', 'mobile', 'tablet'];
    if (device && !validDevices.includes(device)) {
        throw new Error('device_type không hợp lệ');
    }
    // Tạo bản ghi mới
    const record = await Interaction.create({
        userId,
        productId,
        actionCode: action.code,
        device_type: device,
        timestamp: new Date()
    },{raw: true }); // Đảm bảo trả plain object

    return record;
}

export async function getUserInteractions(userId, filterActionCode = null) {
    const whereClause = { userId };
    if (filterActionCode) whereClause.actionCode = filterActionCode;

    const interactions = await Interaction.findAll({
        where: whereClause,
        include: [
            { model: Product, as: 'productData' },
            { model: Allcode, as: 'actionData', foreignKey: 'actionCode', targetKey: 'code', attributes: ['code','value'] }
        ],
        order: [['timestamp','DESC']],
        raw: true,        // BẮT BUỘC
        nest: true        // Quan trọng: giữ cấu trúc include
    });

    return interactions;
}

export async function getAllInteractions(filterActionCode = null) {
    const whereClause = {};
    if (filterActionCode) whereClause.actionCode = filterActionCode;

    const interactions = await Interaction.findAll({
        where: whereClause,
        include: [
            { model: User, as: 'userData', attributes: ['id','email','firstName','lastName'] },
            { model: Product, as: 'productData' },
            { model: Allcode, as: 'actionData', foreignKey: 'actionCode', targetKey: 'code', attributes: ['code','value'] }
        ],
        order: [['timestamp','DESC']],
        raw: true,
        nest: true,
    });

    return interactions;
}

export async function deleteInteraction(userId, productId) {
    return await Interaction.destroy({ where: { userId, productId } });
}
