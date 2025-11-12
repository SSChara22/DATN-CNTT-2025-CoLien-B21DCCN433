import * as interactionService from "../services/interactionService.js";

export async function logInteractionController(req, res) {
    try {
        const { userId, productId, actionCode, device } = req.body;
        const record = await interactionService.logInteraction(userId, productId, actionCode, device);

        res.status(200).json({ success: true, data: record.get({ plain: true }) });
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
}

export async function getUserInteractionsController(req, res) {
    try {
        const userId = req.params.userId;
        const actionCode = req.query.action || null;

        const interactions = await interactionService.getUserInteractions(userId, actionCode);
        const data = interactions.map(i => i.get({ plain: true }));

        res.status(200).json({ success: true, data });
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
}

export async function getAllInteractionsController(req, res) {
    try {
        const actionCode = req.query.action || null;

        const interactions = await interactionService.getAllInteractions(actionCode);
        const data = interactions.map(i => i.get({ plain: true }));

        res.status(200).json({ success: true, data });
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
}

export async function deleteInteractionController(req, res) {
    try {
        const { userId, productId } = req.body;
        await interactionService.deleteInteraction(userId, productId);

        res.status(200).json({ success: true, message: "Interaction deleted" });
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
}
