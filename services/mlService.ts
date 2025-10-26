import { ml } from './api';

export interface MLAnalysisResult {
  emotion: string;
  confidence: number;
  stressLevel: 'low' | 'medium' | 'high';
}

export interface ImageAnalysisRequest {
  imageData: string; // base64 encoded image
}

export interface AudioAnalysisRequest {
  audioData: string; // base64 encoded audio
}

export interface TextAnalysisRequest {
  text: string;
}

// Convert canvas to base64 image
export const canvasToBase64 = (canvas: HTMLCanvasElement): string => {
  return canvas.toDataURL('image/jpeg', 0.8);
};

// Convert audio blob to base64
export const audioBlobToBase64 = (blob: Blob): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      resolve(result);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
};

// Analyze emotion from image
export const analyzeImageEmotion = async (imageData: string): Promise<MLAnalysisResult> => {
  try {
    const response = await ml.analyzeImage(imageData);
    return response.data;
  } catch (error) {
    console.error('Image emotion analysis failed:', error);
    throw new Error('Failed to analyze image emotion');
  }
};

// Analyze emotion from audio
export const analyzeAudioEmotion = async (audioData: string): Promise<MLAnalysisResult> => {
  try {
    const response = await ml.analyzeAudio(audioData);
    return response.data;
  } catch (error) {
    console.error('Audio emotion analysis failed:', error);
    throw new Error('Failed to analyze audio emotion');
  }
};

// Analyze emotion from text
export const analyzeTextEmotion = async (text: string): Promise<MLAnalysisResult> => {
  try {
    const response = await ml.analyzeText(text);
    return response.data;
  } catch (error) {
    console.error('Text emotion analysis failed:', error);
    throw new Error('Failed to analyze text emotion');
  }
};

// Capture image from video element
export const captureImageFromVideo = (video: HTMLVideoElement): string => {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  
  if (!context) {
    throw new Error('Could not get canvas context');
  }
  
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  
  return canvasToBase64(canvas);
};

// Record audio from microphone
export const recordAudio = (duration: number = 5): Promise<string> => {
  return new Promise((resolve, reject) => {
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        const mediaRecorder = new MediaRecorder(stream);
        const audioChunks: Blob[] = [];
        
        mediaRecorder.ondataavailable = (event) => {
          audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
          try {
            const base64Audio = await audioBlobToBase64(audioBlob);
            resolve(base64Audio);
          } catch (error) {
            reject(error);
          }
          // Stop all tracks to release microphone
          stream.getTracks().forEach(track => track.stop());
        };
        
        mediaRecorder.start();
        
        setTimeout(() => {
          mediaRecorder.stop();
        }, duration * 1000);
      })
      .catch(reject);
  });
};
