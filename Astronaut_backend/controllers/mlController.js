import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import EmotionLog from '../models/EmotionLog.js';

// Helper function to run Python scripts
const runPythonScript = (scriptPath, args = []) => {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [scriptPath, ...args]);
    let output = '';
    let error = '';

    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      error += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(output);
          resolve(result);
        } catch (e) {
          resolve({ output, error });
        }
      } else {
        reject(new Error(error || 'Python script failed'));
      }
    });
  });
};

// Analyze emotion from image
export async function analyzeImageEmotion(req, res) {
  try {
    const { imageData } = req.body;
    if (!imageData) {
      return res.status(400).json({ message: 'Image data is required' });
    }

    // Save image temporarily
    const imagePath = `temp_${Date.now()}.jpg`;
    const fullPath = path.join(process.cwd(), 'temp', imagePath);
    
    // Ensure temp directory exists
    if (!fs.existsSync(path.dirname(fullPath))) {
      fs.mkdirSync(path.dirname(fullPath), { recursive: true });
    }

    // Convert base64 to image file
    const base64Data = imageData.replace(/^data:image\/[a-z]+;base64,/, '');
    fs.writeFileSync(fullPath, base64Data, 'base64');

    // Run ML model
    const mlScriptPath = path.join(process.cwd(), '../../models/ml/main.py');
    const result = await runPythonScript(mlScriptPath, [fullPath]);

    // Clean up temp file
    fs.unlinkSync(fullPath);

    // Log emotion to database
    if (req.user) {
      await EmotionLog.create({
        userId: req.user.id,
        source: 'face',
        emotion: result.emotion,
        confidence: result.confidence,
        stressLevel: result.stressLevel || 'low'
      });
    }

    res.json({
      emotion: result.emotion,
      confidence: result.confidence,
      stressLevel: result.stressLevel
    });
  } catch (error) {
    console.error('Image emotion analysis error:', error);
    res.status(500).json({ message: 'Failed to analyze image emotion' });
  }
}

// Analyze emotion from audio
export async function analyzeAudioEmotion(req, res) {
  try {
    const { audioData } = req.body;
    if (!audioData) {
      return res.status(400).json({ message: 'Audio data is required' });
    }

    // Save audio temporarily
    const audioPath = `temp_audio_${Date.now()}.wav`;
    const fullPath = path.join(process.cwd(), 'temp', audioPath);
    
    // Ensure temp directory exists
    if (!fs.existsSync(path.dirname(fullPath))) {
      fs.mkdirSync(path.dirname(fullPath), { recursive: true });
    }

    // Convert base64 to audio file
    const base64Data = audioData.replace(/^data:audio\/[a-z]+;base64,/, '');
    fs.writeFileSync(fullPath, base64Data, 'base64');

    // Run speech emotion analysis
    const speechScriptPath = path.join(process.cwd(), '../../models/speech/main.py');
    const result = await runPythonScript(speechScriptPath, [fullPath]);

    // Clean up temp file
    fs.unlinkSync(fullPath);

    // Log emotion to database
    if (req.user) {
      await EmotionLog.create({
        userId: req.user.id,
        source: 'voice',
        emotion: result.emotion,
        confidence: result.confidence,
        stressLevel: result.stressLevel || 'low'
      });
    }

    res.json({
      emotion: result.emotion,
      confidence: result.confidence,
      stressLevel: result.stressLevel
    });
  } catch (error) {
    console.error('Audio emotion analysis error:', error);
    res.status(500).json({ message: 'Failed to analyze audio emotion' });
  }
}

// Analyze emotion from text
export async function analyzeTextEmotion(req, res) {
  try {
    const { text } = req.body;
    if (!text) {
      return res.status(400).json({ message: 'Text is required' });
    }

    // For now, use a simple keyword-based approach
    // In production, you'd use a proper NLP model
    const emotionKeywords = {
      happy: ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing'],
      sad: ['sad', 'depressed', 'down', 'terrible', 'awful', 'horrible'],
      angry: ['angry', 'mad', 'furious', 'annoyed', 'frustrated'],
      anxious: ['anxious', 'worried', 'nervous', 'stressed', 'concerned'],
      calm: ['calm', 'peaceful', 'relaxed', 'serene', 'content']
    };

    const words = text.toLowerCase().split(/\s+/);
    const emotionScores = {};

    Object.keys(emotionKeywords).forEach(emotion => {
      emotionScores[emotion] = words.filter(word => 
        emotionKeywords[emotion].includes(word)
      ).length;
    });

    const topEmotion = Object.keys(emotionScores).reduce((a, b) => 
      emotionScores[a] > emotionScores[b] ? a : b
    );

    const confidence = Math.min(0.9, emotionScores[topEmotion] / words.length * 10);

    // Log emotion to database
    if (req.user) {
      await EmotionLog.create({
        userId: req.user.id,
        source: 'text',
        emotion: topEmotion,
        confidence: confidence,
        stressLevel: confidence > 0.7 ? 'high' : confidence > 0.4 ? 'medium' : 'low'
      });
    }

    res.json({
      emotion: topEmotion,
      confidence: confidence,
      stressLevel: confidence > 0.7 ? 'high' : confidence > 0.4 ? 'medium' : 'low'
    });
  } catch (error) {
    console.error('Text emotion analysis error:', error);
    res.status(500).json({ message: 'Failed to analyze text emotion' });
  }
}
