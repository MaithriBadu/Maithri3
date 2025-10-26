import React, { useMemo } from 'react';
import { ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend, Tooltip } from 'recharts';
import type { CrewMember } from '../types';
import { Panel } from './Panel';
import { AlertIcon } from './IconComponents';

interface CrewSyncPanelProps {
  crew: CrewMember[];
}

export const CrewSyncPanel: React.FC<CrewSyncPanelProps> = ({ crew }) => {
  const emotionToScore = (emotion: string) => {
    switch (emotion) {
      case 'Calm': return 5;
      case 'Neutral': return 4;
      case 'Anxious': return 3;
      case 'Fatigued': return 2;
      case 'Stressed': return 1;
      default: return 3;
    }
  };

  const chartData = useMemo(() => {
    return crew.map(member => ({
      subject: member.name.split(' ').pop(), // Use last name
      A: emotionToScore(member.emotion),
      fullMark: 5,
    }));
  }, [crew]);
  
  const outlier = useMemo(() => {
    const avgScore = chartData.reduce((sum, d) => sum + d.A, 0) / chartData.length;
    const memberScores = crew.map(m => ({ ...m, score: emotionToScore(m.emotion) }));
    return memberScores.find(m => Math.abs(m.score - avgScore) > 1.5);
  }, [chartData, crew]);

  return (
    <Panel title="Crew Emotional Synchronization">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 h-96">
                <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={chartData}>
                    <PolarGrid stroke="rgba(192, 132, 252, 0.3)" />
                    <PolarAngleAxis dataKey="subject" stroke="#a78bfa" />
                    <PolarRadiusAxis angle={30} domain={[0, 5]} stroke="none" />
                    <Radar name="Emotional State" dataKey="A" stroke="#818cf8" fill="#818cf8" fillOpacity={0.6} />
                    <Tooltip contentStyle={{ backgroundColor: 'rgba(31, 41, 55, 0.8)', border: '1px solid #4f46e5', color: '#e5e7eb' }} />
                    <Legend />
                </RadarChart>
                </ResponsiveContainer>
            </div>
            <div className="space-y-4">
                 <h3 className="font-semibold text-purple-400">Crew Status</h3>
                 <ul className="space-y-2">
                    {crew.map(member => (
                        <li key={member.id} className={`flex justify-between items-center p-2 rounded ${outlier?.id === member.id ? 'bg-yellow-500/20' : 'bg-gray-500/10'}`}>
                            <span>{member.name}</span>
                            <span className="font-medium">{member.emotion}</span>
                        </li>
                    ))}
                 </ul>
                 {outlier && (
                    <div className="flex items-start gap-3 p-3 bg-yellow-500/10 rounded-lg border border-yellow-400/30">
                        <AlertIcon className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-1" />
                        <div>
                            <p className="font-semibold text-yellow-300">Emotional Outlier Detected</p>
                            <p className="text-sm text-gray-300">{outlier.name}'s emotional state is significantly different from the crew average. Consider a private check-in.</p>
                        </div>
                    </div>
                 )}
            </div>
        </div>
    </Panel>
  );
};
