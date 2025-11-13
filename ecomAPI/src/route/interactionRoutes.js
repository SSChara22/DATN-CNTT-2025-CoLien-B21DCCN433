import express from "express";
import * as interactionController from "../controllers/interactionController.js"; // ESM import
const router = express.Router();
import { validateLogInteraction } from "../middlewares/validateInteraction.js";

// Ghi interaction mới với middleware xác thực
router.post("/", validateLogInteraction, interactionController.logInteractionController);

// Lấy tất cả interaction của user
router.get("/user/:userId", interactionController.getUserInteractionsController);

// Xóa interaction (admin hoặc test)
router.delete("/", interactionController.deleteInteractionController);

// Lấy tất cả interaction (tùy chọn lọc action)
router.get("/", interactionController.getAllInteractionsController);

export default router;
