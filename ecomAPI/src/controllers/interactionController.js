import * as interactionService from "../services/interactionService.js";
import recommendationService from "../services/recommendationService";

export async function logInteractionController(req, res) {
    try {
        const { userId, productId, actionCode, device } = req.body;
        const record = await interactionService.logInteraction(userId, productId, actionCode, device);
        // Recompute Top-10 recommendations for this user in real-time
        try {
            await recommendationService.initForUser(userId, 10);
        } catch (e) {
            // non-blocking: ignore errors to not affect UX
        }

        res.status(200).json({ success: true, data:record});
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
}

// getUserInteractionsController
export async function getUserInteractionsController(req, res) {
    try {
        const userId = req.params.userId;
        const actionCode = req.query.action || null;

        const interactions = await interactionService.getUserInteractions(userId, actionCode);

        res.status(200).json({ success: true, data: interactions });
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
}

// getAllInteractionsController
export async function getAllInteractionsController(req, res) {
    try {
        const actionCode = req.query.action || null;

        const interactions = await interactionService.getAllInteractions(actionCode);

        res.status(200).json({ success: true, data: interactions });
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
