import React, { useRef, useState, useCallback } from 'react';
import { analyzeImageEmotion, analyzeAudioEmotion, analyzeTextEmotion, captureImageFromVideo, recordAudio } from '../services/mlService';
import type { MLAnalysisResult, MLSource } from '../types';

interface CameraAnalysisProps {
  onAnalysisComplete: (result: MLSource) => void;
  isAnalyzing: boolean;
  setIsAnalyzing: (analyzing: boolean) => void;
}

export const CameraAnalysis: React.FC<CameraAnalysisProps> = ({ onAnalysisComplete, isAnalyzing, setIsAnalyzing }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);

  const startCamera = useCallback(async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        setStream(mediaStream);
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
  }, [stream]);

  const captureAndAnalyze = useCallback(async () => {
    if (!videoRef.current || isAnalyzing) return;

    setIsAnalyzing(true);
    try {
      const imageData = captureImageFromVideo(videoRef.current);
      const result = await analyzeImageEmotion(imageData);
      
      onAnalysisComplete({
        type: 'camera',
        timestamp: Date.now(),
        result
      });
    } catch (error) {
      console.error('Camera analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  }, [isAnalyzing, onAnalysisComplete, setIsAnalyzing]);

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <button
          onClick={startCamera}
          disabled={isAnalyzing}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg transition-colors"
        >
          Start Camera
        </button>
        <button
          onClick={stopCamera}
          disabled={isAnalyzing}
          className="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 text-white rounded-lg transition-colors"
        >
          Stop Camera
        </button>
        <button
          onClick={captureAndAnalyze}
          disabled={!stream || isAnalyzing}
          className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white rounded-lg transition-colors"
        >
          {isAnalyzing ? 'Analyzing...' : 'Capture & Analyze'}
        </button>
      </div>
      
      <div className="relative aspect-video bg-black rounded-md overflow-hidden border border-purple-400/30">
        <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover"></video>
        {isAnalyzing && (
          <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
            <div className="text-white text-lg">Analyzing emotion...</div>
          </div>
        )}
      </div>
    </div>
  );
};

interface AudioAnalysisProps {
  onAnalysisComplete: (result: MLSource) => void;
  isAnalyzing: boolean;
  setIsAnalyzing: (analyzing: boolean) => void;
}

export const AudioAnalysis: React.FC<AudioAnalysisProps> = ({ onAnalysisComplete, isAnalyzing, setIsAnalyzing }) => {
  const [isRecording, setIsRecording] = useState(false);

  const recordAndAnalyze = useCallback(async () => {
    if (isAnalyzing || isRecording) return;

    setIsAnalyzing(true);
    setIsRecording(true);
    
    try {
      const audioData = await recordAudio(5); // Record for 5 seconds
      const result = await analyzeAudioEmotion(audioData);
      
      onAnalysisComplete({
        type: 'microphone',
        timestamp: Date.now(),
        result
      });
    } catch (error) {
      console.error('Audio analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
      setIsRecording(false);
    }
  }, [isAnalyzing, isRecording, onAnalysisComplete, setIsAnalyzing]);

  return (
    <div className="space-y-4">
      <button
        onClick={recordAndAnalyze}
        disabled={isAnalyzing || isRecording}
        className="w-full px-4 py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white rounded-lg transition-colors font-semibold"
      >
        {isRecording ? 'Recording... (5s)' : isAnalyzing ? 'Analyzing...' : 'Record & Analyze Voice'}
      </button>
      
      {isRecording && (
        <div className="text-center text-purple-400">
          <div className="animate-pulse">🎤 Recording audio...</div>
        </div>
      )}
    </div>
  );
};

interface TextAnalysisProps {
  onAnalysisComplete: (result: MLSource) => void;
  isAnalyzing: boolean;
  setIsAnalyzing: (analyzing: boolean) => void;
}

export const TextAnalysis: React.FC<TextAnalysisProps> = ({ onAnalysisComplete, isAnalyzing, setIsAnalyzing }) => {
  const [text, setText] = useState('');

  const analyzeText = useCallback(async () => {
    if (!text.trim() || isAnalyzing) return;

    setIsAnalyzing(true);
    try {
      const result = await analyzeTextEmotion(text);
      
      onAnalysisComplete({
        type: 'text',
        timestamp: Date.now(),
        result
      });
    } catch (error) {
      console.error('Text analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  }, [text, isAnalyzing, onAnalysisComplete, setIsAnalyzing]);

  return (
    <div className="space-y-4">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter your thoughts or feelings here..."
        className="w-full p-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:border-purple-400 focus:outline-none resize-none"
        rows={4}
        disabled={isAnalyzing}
      />
      
      <button
        onClick={analyzeText}
        disabled={!text.trim() || isAnalyzing}
        className="w-full px-4 py-3 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-600 text-white rounded-lg transition-colors font-semibold"
      >
        {isAnalyzing ? 'Analyzing...' : 'Analyze Text Emotion'}
      </button>
    </div>
  );
};

interface AnalysisResultProps {
  result: MLAnalysisResult;
  source: string;
}

export const AnalysisResult: React.FC<AnalysisResultProps> = ({ result, source }) => {
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-400';
    if (confidence >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getStressColor = (stressLevel: string) => {
    switch (stressLevel) {
      case 'low': return 'text-green-400';
      case 'medium': return 'text-yellow-400';
      case 'high': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  return (
    <div className="bg-gray-800/50 border border-gray-600 rounded-lg p-4 space-y-2">
      <div className="flex items-center justify-between">
        <h4 className="font-semibold text-white">Analysis Result</h4>
        <span className="text-sm text-gray-400 capitalize">{source}</span>
      </div>
      
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-sm text-gray-400">Emotion</p>
          <p className="text-lg font-semibold text-purple-400 capitalize">{result.emotion}</p>
        </div>
        <div>
          <p className="text-sm text-gray-400">Confidence</p>
          <p className={`text-lg font-semibold ${getConfidenceColor(result.confidence)}`}>
            {(result.confidence * 100).toFixed(1)}%
          </p>
        </div>
      </div>
      
      <div>
        <p className="text-sm text-gray-400">Stress Level</p>
        <p className={`text-lg font-semibold ${getStressColor(result.stressLevel)}`}>
          {result.stressLevel.toUpperCase()}
        </p>
      </div>
    </div>
  );
};
