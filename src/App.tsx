import React, { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { Dashboard } from './components/Dashboard';
import { IndividualMonitor } from './components/IndividualMonitor';
import { CrewSyncPanel } from './components/CrewSyncPanel';
import { SettingsPanel } from './components/SettingsPanel';
import { Shortcuts } from './components/Shortcuts';
import { AuthPage } from './components/AuthPage';
import type { CrewMember, EmotionalState, Vitals } from './types';
import { EMOTIONAL_STATES } from './constants';

const App: React.FC = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [isDrawerOpen, setIsDrawerOpen] = useState(true);
  const [currentUser, setCurrentUser] = useState<CrewMember>({
    id: 'astro-01',
    name: 'Cmdr. Eva Rostova',
    vitals: { heartRate: 72, fatigueIndex: 0.3, posture: 'Good' },
    emotion: 'Calm',
    emotionHistory: [],
    vitalsHistory: [],
  });
  const [crew, setCrew] = useState<CrewMember[]>([
    currentUser,
    { id: 'astro-02', name: 'Dr. Kenji Tanaka', emotion: 'Neutral' },
    { id: 'astro-03', name: 'Maj. Priya Singh', emotion: 'Anxious' },
    { id: 'astro-04', name: 'Lt. Omar Hassan', emotion: 'Fatigued' },
  ]);

  const updateSimulatedData = useCallback(() => {
    const randomEmotion = () => EMOTIONAL_STATES[Math.floor(Math.random() * EMOTIONAL_STATES.length)];
    
    // Update current user
    setCurrentUser(prevUser => {
      const newVitals: Vitals = {
        heartRate: prevUser.vitals.heartRate + Math.floor(Math.random() * 5) - 2,
        fatigueIndex: Math.max(0, Math.min(1, prevUser.vitals.fatigueIndex + (Math.random() * 0.1 - 0.05))),
        posture: Math.random() > 0.8 ? 'Poor' : 'Good',
      };

      const newEmotion: EmotionalState = randomEmotion();

      const updatedUser = {
        ...prevUser,
        vitals: newVitals,
        emotion: newEmotion,
        emotionHistory: [...prevUser.emotionHistory, { timestamp: Date.now(), emotion: newEmotion }].slice(-50),
        vitalsHistory: [...prevUser.vitalsHistory, { timestamp: Date.now(), vitals: newVitals }].slice(-50),
      };

      // Also update the user in the crew list
      setCrew(prevCrew => 
        prevCrew.map(member => (member.id === updatedUser.id ? updatedUser : member))
      );

      return updatedUser;
    });

    // Update rest of the crew
    setCrew(prevCrew => 
      prevCrew.map(member => {
        if(member.id === 'astro-01') return member; // Will be updated above
        return { ...member, emotion: randomEmotion() };
      })
    );

  }, []);
  
  useEffect(() => {
    const interval = setInterval(updateSimulatedData, 5000);
    return () => clearInterval(interval);
  }, [updateSimulatedData]);

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard user={currentUser} />;
      case 'monitor':
        return <IndividualMonitor user={currentUser} />;
      case 'crew':
        return <CrewSyncPanel crew={crew} />;
      case 'settings':
        return <SettingsPanel />;
      default:
        return <Dashboard user={currentUser} />;
    }
  };

  const handleLogin = () => {
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-200 font-sans relative overflow-hidden">
        <div className="fixed inset-0 z-0 pointer-events-none">
            <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-black via-gray-900 to-purple-900/80 opacity-40"></div>
            <div className="absolute -top-1/4 left-0 w-1/2 h-1/2 bg-purple-600/40 rounded-full filter blur-3xl animate-pulse"></div>
            <div className="absolute -bottom-1/4 right-0 w-1/2 h-1/2 bg-indigo-600/40 rounded-full filter blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
        </div>
        
        {!isAuthenticated ? (
            <AuthPage onLoginSuccess={handleLogin} />
        ) : (
            <>
                <Shortcuts isDrawerOpen={isDrawerOpen} setIsDrawerOpen={setIsDrawerOpen} activeTab={activeTab} setActiveTab={setActiveTab} onLogout={handleLogout} />
                <div className={`relative flex flex-col min-h-screen transition-all duration-300 ease-in-out ${isDrawerOpen ? 'pl-64' : 'pl-20'}`}>
                    <Header />
                    <main className="flex-grow p-4 sm:p-6 lg:p-8 overflow-y-auto">
                        {renderContent()}
                    </main>
                </div>
            </>
        )}
    </div>
  );
};

export default App;