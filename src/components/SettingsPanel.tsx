import React, { useState } from 'react';
import { Panel } from './Panel';

const mockLogs = [
    { timestamp: '2024-07-21 14:30', mood: 'Calm', hr: 75, fatigue: 0.2, posture: 'Good' },
    { timestamp: '2024-07-21 15:00', mood: 'Neutral', hr: 78, fatigue: 0.25, posture: 'Good' },
    { timestamp: '2024-07-21 15:30', mood: 'Anxious', hr: 88, fatigue: 0.3, posture: 'Poor' },
    { timestamp: '2024-07-21 16:00', mood: 'Stressed', hr: 95, fatigue: 0.4, posture: 'Poor' },
    { timestamp: '2024-07-21 16:30', mood: 'Fatigued', hr: 80, fatigue: 0.5, posture: 'Good' },
];

export const SettingsPanel: React.FC = () => {
  const [voiceTone, setVoiceTone] = useState('calm');

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Panel title="Companion Mode Settings">
        <div className="space-y-6">
          <div>
            <label className="block text-gray-300 font-medium mb-2">Voice Tone</label>
            <div className="flex space-x-2 sm:space-x-4">
              {['Calm', 'Cheerful', 'Assertive'].map(tone => (
                <button
                  key={tone}
                  onClick={() => setVoiceTone(tone.toLowerCase())}
                  className={`px-4 py-2 rounded-lg transition-colors text-sm sm:text-base ${
                    voiceTone === tone.toLowerCase()
                      ? 'bg-purple-600 text-white'
                      : 'bg-white/10 hover:bg-white/20'
                  }`}
                >
                  {tone}
                </button>
              ))}
            </div>
          </div>
          <div className="flex items-center justify-between">
            <label htmlFor="emergency-mode" className="text-gray-300 font-medium">Emergency Report Mode</label>
            <label htmlFor="emergency-mode" className="flex items-center cursor-pointer">
              <div className="relative">
                <input type="checkbox" id="emergency-mode" className="sr-only peer" />
                <div className="block bg-gray-600 w-14 h-8 rounded-full peer-checked:bg-purple-600 transition"></div>
                <div className="dot absolute left-1 top-1 bg-white w-6 h-6 rounded-full transition-transform peer-checked:translate-x-full"></div>
              </div>
            </label>
          </div>
        </div>
      </Panel>
      <Panel title="Research Mode and Logs">
        <div className="space-y-4">
          <div>
             <div className="h-64 overflow-y-auto bg-black/30 rounded-lg p-2 border border-white/10">
                <table className="w-full text-sm text-left">
                    <thead className="text-xs text-purple-300 uppercase bg-black/50 sticky top-0">
                        <tr>
                            <th scope="col" className="px-4 py-2">Timestamp</th>
                            <th scope="col" className="px-4 py-2">Mood</th>
                            <th scope="col" className="px-4 py-2">HR</th>
                            <th scope="col" className="px-4 py-2">Fatigue</th>
                        </tr>
                    </thead>
                    <tbody>
                        {mockLogs.map(log => (
                            <tr key={log.timestamp} className="border-b border-white/10 last:border-b-0 hover:bg-white/5">
                                <td className="px-4 py-2 text-gray-400">{log.timestamp}</td>
                                <td className="px-4 py-2">{log.mood}</td>
                                <td className="px-4 py-2">{log.hr}</td>
                                <td className="px-4 py-2">{log.fatigue}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
             </div>
          </div>
          <div className="flex gap-4">
            <button className="flex-1 bg-green-600/80 hover:bg-green-600 text-white font-bold py-2.5 px-4 rounded-lg transition-colors">Export CSV</button>
            <button className="flex-1 bg-blue-600/80 hover:bg-blue-600 text-white font-bold py-2.5 px-4 rounded-lg transition-colors">Export Encrypted</button>
          </div>
          <div className="border-t border-white/10 pt-4">
             <h3 className="font-semibold text-purple-400 mb-2">Summary Insights</h3>
             <p className="text-sm text-gray-300 bg-black/30 p-3 rounded-lg">
                Trend analysis indicates a rise in stress and fatigue during afternoon hours. Recommend scheduling a mandatory relaxation period at 15:00.
             </p>
          </div>
        </div>
      </Panel>
    </div>
  );
};
