import React, { useMemo, useState, useCallback } from 'react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import type { CrewMember, MLSource } from '../types';
import { Panel } from './Panel';
import { AudioAnalysis, TextAnalysis, AnalysisResult } from './MLAnalysisComponents';
// import { getInterventionSuggestions } from '../../services/geminiService';

interface IndividualMonitorProps {
  user: CrewMember;
}

const ChartPanel: React.FC<{ data: any[]; dataKey: string; color: string; name: string; domain: [number, number] }> = ({ data, dataKey, color, name, domain }) => {
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id={`color${dataKey}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.8}/>
              <stop offset="95%" stopColor={color} stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(192, 132, 252, 0.2)" />
          <XAxis dataKey="time" stroke="#9ca3af" tick={{ fontSize: 12 }} />
          <YAxis stroke="#9ca3af" domain={domain} tick={{ fontSize: 12 }} />
          <Tooltip contentStyle={{ backgroundColor: 'rgba(10, 10, 20, 0.8)', border: '1px solid #4f46e5', backdropFilter: 'blur(4px)', color: '#e5e7eb' }} />
          <Legend wrapperStyle={{ color: '#d1d5db' }}/>
          <Area type="monotone" dataKey={dataKey} name={name} stroke={color} fillOpacity={1} fill={`url(#color${dataKey})`} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};


const MLAnalysisPanel: React.FC<{ onMLAnalysis: (result: MLSource) => void }> = ({ onMLAnalysis }) => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [latestAnalysis, setLatestAnalysis] = useState<MLSource | null>(null);

  const handleAnalysisComplete = useCallback((result: MLSource) => {
    setLatestAnalysis(result);
    onMLAnalysis(result);
  }, [onMLAnalysis]);

  return (
    <Panel title="Multi-Modal Emotion Analysis">
      <div className="space-y-6">
        <div>
          <h4 className="text-lg font-semibold text-purple-400 mb-3">Voice Analysis</h4>
          <AudioAnalysis 
            onAnalysisComplete={handleAnalysisComplete}
            isAnalyzing={isAnalyzing}
            setIsAnalyzing={setIsAnalyzing}
          />
        </div>
        
        <div>
          <h4 className="text-lg font-semibold text-purple-400 mb-3">Text Analysis</h4>
          <TextAnalysis 
            onAnalysisComplete={handleAnalysisComplete}
            isAnalyzing={isAnalyzing}
            setIsAnalyzing={setIsAnalyzing}
          />
        </div>
        
        {latestAnalysis && (
          <AnalysisResult 
            result={latestAnalysis.result} 
            source={latestAnalysis.type} 
          />
        )}
      </div>
    </Panel>
  );
};

const InterventionsPanel: React.FC<{user: CrewMember}> = ({ user }) => {
    const [suggestions, setSuggestions] = useState<string[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchSuggestions = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            // Mock suggestions for now
            const mockSuggestions = [
                'Take a 5-minute breathing exercise',
                'Consider a short walk around the station',
                'Engage in a relaxing activity'
            ];
            setSuggestions(mockSuggestions);
        } catch (e) {
            setError('Failed to fetch suggestions from MAITRI core.');
            console.error(e);
        } finally {
            setLoading(false);
        }
    }, [user.vitals, user.emotion]);

    return (
        <Panel title="Intervention Suggestions">
            <button onClick={fetchSuggestions} disabled={loading} className="w-full bg-indigo-500/80 hover:bg-indigo-500 disabled:bg-indigo-900/50 text-white font-bold py-3 px-4 rounded-lg transition-colors mb-4 text-base">
                {loading ? 'Analyzing...' : 'Ask MAITRI for Advice'}
            </button>
            {error && <p className="text-red-400 text-sm">{error}</p>}
            <ul className="space-y-2">
                {suggestions.length > 0 ? suggestions.map((suggestion, index) => (
                    <li key={index} className="bg-purple-500/10 p-3 rounded-md text-gray-300">{suggestion}</li>
                )) : <p className="text-gray-400">Click button to generate suggestions based on current status.</p>}
            </ul>
        </Panel>
    )
}

export const IndividualMonitor: React.FC<IndividualMonitorProps> = ({ user }) => {
  const [mlAnalysis, setMlAnalysis] = useState<MLSource[]>([]);

  const handleMLAnalysis = useCallback((result: MLSource) => {
    setMlAnalysis(prev => [...prev, result].slice(-10)); // Keep last 10 analyses
  }, []);

  const chartData = useMemo(() => {
    return (user.vitalsHistory || []).map((entry, i) => ({
      time: `T-${(user.vitalsHistory?.length || 0) - i}`,
      fatigue: entry.vitals.fatigueIndex,
      stress: (entry.vitals.heartRate > 85 ? Math.random() * 0.4 + 0.5 : Math.random() * 0.4 + 0.1), // Simulated stress
      sleepQuality: Math.random() * 0.5 + 0.4 // Simulated sleep
    }));
  }, [user.vitalsHistory]);

  return (
    <div className="space-y-6">
        <Panel title={`Health and Emotion Monitor: ${user.name}`}>
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-x-8 gap-y-6">
                <div>
                    <h3 className="font-semibold text-purple-400 mb-2">Fatigue & Stress Levels</h3>
                    <ChartPanel data={chartData} dataKey="fatigue" color="#facc15" name="Fatigue Index" domain={[0, 1]}/>
                    <ChartPanel data={chartData} dataKey="stress" color="#f87171" name="Stress Index" domain={[0, 1]}/>
                </div>
                <div>
                    <h3 className="font-semibold text-purple-400 mb-2">Sleep Cycle Insights</h3>
                    <ChartPanel data={chartData} dataKey="sleepQuality" color="#60a5fa" name="Sleep Quality" domain={[0,1]} />
                </div>
            </div>
        </Panel>
        
        <MLAnalysisPanel onMLAnalysis={handleMLAnalysis} />
        
        <InterventionsPanel user={user} />
    </div>
  );
};
