import express from 'express';
import { analyzeImageEmotion, analyzeAudioEmotion, analyzeTextEmotion } from '../controllers/mlController.js';
import { authenticateToken } from '../middleware/authMiddleware.js';

const router = express.Router();

// All ML routes require authentication
router.use(authenticateToken);

// Emotion analysis endpoints
router.post('/analyze-image', analyzeImageEmotion);
router.post('/analyze-audio', analyzeAudioEmotion);
router.post('/analyze-text', analyzeTextEmotion);

export default router;
