import express from 'express';
import { authenticateToken as requireAuth } from '../middleware/authMiddleware.js';
import { 
    getHealthStatus, 
    switchPersona, 
    logEmotion, 
    getEmotionHistory, 
    getUserProfile, 
    updateUserProfile 
} from '../controllers/userProfileController.js';

const router = express.Router();

// All routes in this file are protected
router.use(requireAuth);

// Health and status endpoints
router.get('/health', getHealthStatus);
router.get('/profile', getUserProfile);
router.put('/profile', updateUserProfile);

// Persona management
router.put('/persona', switchPersona);

// Emotion logging
router.post('/emotion-log', logEmotion);
router.get('/emotion-history', getEmotionHistory);

export default router;
