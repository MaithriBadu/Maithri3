import React, { useRef, useEffect, useState, useCallback } from 'react';
import type { CrewMember, MLSource } from '../types';
import { Panel } from './Panel';
import { HeartIcon, BatteryIcon, UserIcon, AlertIcon } from './IconComponents';
import { CameraAnalysis, AnalysisResult } from './MLAnalysisComponents';

interface DashboardProps {
  user: CrewMember;
}

const EmotionIndicator: React.FC<{ emotion: string; onMLAnalysis?: (result: MLSource) => void }> = ({ emotion, onMLAnalysis }) => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [latestAnalysis, setLatestAnalysis] = useState<MLSource | null>(null);

  const handleAnalysisComplete = useCallback((result: MLSource) => {
    setLatestAnalysis(result);
    if (onMLAnalysis) {
      onMLAnalysis(result);
    }
  }, [onMLAnalysis]);

  const emotionColor: { [key: string]: string } = {
    Calm: 'text-green-400',
    Stressed: 'text-red-400',
    Fatigued: 'text-yellow-400',
    Anxious: 'text-orange-400',
    Neutral: 'text-blue-400',
  };

  return (
    <Panel title="Real-time Emotion Detection">
      <div className="space-y-4">
        <CameraAnalysis 
          onAnalysisComplete={handleAnalysisComplete}
          isAnalyzing={isAnalyzing}
          setIsAnalyzing={setIsAnalyzing}
        />
        
        {latestAnalysis && (
          <AnalysisResult 
            result={latestAnalysis.result} 
            source="Camera Analysis" 
          />
        )}
        
        <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
          <div>
            <p className="text-sm text-gray-400">Current Status</p>
            <p className={`text-xl font-bold ${emotionColor[emotion] || 'text-gray-300'}`}>{emotion}</p>
          </div>
          <div className="text-right">
            <p className="text-sm text-gray-400">Last Updated</p>
            <p className="text-sm text-gray-300">{new Date().toLocaleTimeString()}</p>
          </div>
        </div>
      </div>
    </Panel>
  );
};

const VitalsPanel: React.FC<{ vitals: CrewMember['vitals'] }> = ({ vitals }) => (
  <Panel title="Physical Vitals">
    <div className="grid grid-cols-3 gap-4 text-center">
      <div>
        <HeartIcon className="w-8 h-8 mx-auto text-red-400" />
        <p className="mt-2 text-2xl font-semibold">{vitals?.heartRate}</p>
        <p className="text-sm text-gray-400">BPM</p>
      </div>
      <div>
        <BatteryIcon className="w-8 h-8 mx-auto text-blue-400" />
        <p className="mt-2 text-2xl font-semibold">{(100 - (vitals?.fatigueIndex || 0) * 100).toFixed(0)}%</p>
        <p className="text-sm text-gray-400">Energy</p>
      </div>
      <div>
        <UserIcon className="w-8 h-8 mx-auto text-green-400" />
        <p className="mt-2 text-2xl font-semibold">{vitals?.posture}</p>
        <p className="text-sm text-gray-400">Posture</p>
      </div>
    </div>
  </Panel>
);

const AlertsPanel: React.FC = () => (
    <Panel title="Alerts & Recommendations">
        <div className="space-y-3">
            <div className="flex items-start gap-3 p-3 bg-yellow-500/10 rounded-lg">
                <AlertIcon className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-1" />
                <div>
                    <p className="font-semibold text-yellow-300">Fatigue Alert</p>
                    <p className="text-sm text-gray-300">Fatigue index rising. Consider a 10-minute rest period.</p>
                </div>
            </div>
             <div className="flex items-start gap-3 p-3 bg-orange-500/10 rounded-lg">
                <AlertIcon className="w-5 h-5 text-orange-400 flex-shrink-0 mt-1" />
                <div>
                    <p className="font-semibold text-orange-300">Group Morale</p>
                    <p className="text-sm text-gray-300">Crew emotion desynchronized. Suggestion: Shared meal break.</p>
                </div>
            </div>
        </div>
    </Panel>
);


export const Dashboard: React.FC<DashboardProps> = ({ user }) => {
  const [mlAnalysis, setMlAnalysis] = useState<MLSource[]>([]);

  const handleMLAnalysis = useCallback((result: MLSource) => {
    setMlAnalysis(prev => [...prev, result].slice(-10)); // Keep last 10 analyses
  }, []);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
      <div className="lg:col-span-2 xl:col-span-2">
        <EmotionIndicator emotion={user.emotion} onMLAnalysis={handleMLAnalysis} />
      </div>
      <VitalsPanel vitals={user.vitals} />
      <div className="lg:col-span-2 xl:col-span-3">
        <AlertsPanel />
      </div>
    </div>
  );
};
