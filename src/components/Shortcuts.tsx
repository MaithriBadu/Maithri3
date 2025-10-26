import React from 'react';
import { DashboardIcon, MonitorIcon, UsersIcon, SettingsIcon, ChevronDoubleLeftIcon, MicrophoneIcon, SparklesIcon, DocumentTextIcon, LogoutIcon } from './IconComponents';

interface ShortcutsProps {
  isDrawerOpen: boolean;
  setIsDrawerOpen: (isOpen: boolean) => void;
  activeTab: string;
  setActiveTab: (tab: string) => void;
  onLogout: () => void;
}

const NavButton: React.FC<{
  icon: React.ReactNode;
  label: string;
  isActive: boolean;
  isDrawerOpen: boolean;
  onClick: () => void;
}> = ({ icon, label, isActive, isDrawerOpen, onClick }) => (
  <button
    onClick={onClick}
    className={`w-full flex items-center gap-4 px-4 py-3 rounded-lg text-left transition-all duration-200 group ${
      isActive
        ? 'bg-purple-600/50 text-white shadow-lg shadow-purple-900/50'
        : 'text-gray-400 hover:bg-white/10 hover:text-white'
    } ${!isDrawerOpen && 'justify-center'}`}
    aria-label={label}
  >
    <div className="flex-shrink-0">{icon}</div>
    <span className={`transition-opacity duration-200 whitespace-nowrap ${isDrawerOpen ? 'opacity-100' : 'opacity-0 h-0 w-0'}`}>{label}</span>
  </button>
);

const InteractionButton: React.FC<{
  icon: React.ReactNode;
  label: string;
  isDrawerOpen: boolean;
  isPrimary?: boolean;
}> = ({ icon, label, isDrawerOpen, isPrimary }) => (
    <button className={`w-full flex items-center gap-4 px-4 py-2.5 rounded-lg text-left transition-all duration-200 group text-white font-semibold ${
        isPrimary ? 'bg-indigo-500/80 hover:bg-indigo-500' : 'bg-white/10 hover:bg-white/20'
    } ${!isDrawerOpen && 'justify-center'}`}
    aria-label={label}
    >
        <div className="flex-shrink-0">{icon}</div>
        <span className={`transition-opacity duration-200 whitespace-nowrap ${isDrawerOpen ? 'opacity-100' : 'opacity-0 h-0 w-0'}`}>{label}</span>
    </button>
);


export const Shortcuts: React.FC<ShortcutsProps> = ({ activeTab, setActiveTab, isDrawerOpen, setIsDrawerOpen, onLogout }) => {
  return (
    <aside className={`fixed top-0 left-0 h-full z-30 bg-black/40 backdrop-blur-xl border-r border-white/10 transition-all duration-300 ease-in-out ${isDrawerOpen ? 'w-64' : 'w-20'}`}>
        <div className="flex flex-col h-full">
            {/* Header */}
            <div className={`flex items-center flex-shrink-0 border-b border-white/10 transition-all duration-300 relative ${isDrawerOpen ? 'px-6' : 'px-0 justify-center'}`} style={{ height: '73px' }}>
                <div className={`transition-all duration-300 ease-in-out ${isDrawerOpen ? 'opacity-100' : 'opacity-0 scale-50'}`}>
                    <h1 className="text-2xl font-bold tracking-wider text-purple-300 whitespace-nowrap">M<span className="text-gray-300">A</span>ITRI</h1>
                </div>
                <div className={`absolute left-1/2 -translate-x-1/2 transition-all duration-300 ease-in-out ${!isDrawerOpen ? 'opacity-100' : 'opacity-0 scale-50'}`}>
                     <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-lg shadow-lg"></div>
                </div>
            </div>

            {/* Main Navigation */}
            <nav className="flex-grow p-4 space-y-2">
                <NavButton icon={<DashboardIcon />} label="Dashboard" isActive={activeTab === 'dashboard'} isDrawerOpen={isDrawerOpen} onClick={() => setActiveTab('dashboard')} />
                <NavButton icon={<MonitorIcon />} label="Health Monitor" isActive={activeTab === 'monitor'} isDrawerOpen={isDrawerOpen} onClick={() => setActiveTab('monitor')} />
                <NavButton icon={<UsersIcon />} label="Crew Sync" isActive={activeTab === 'crew'} isDrawerOpen={isDrawerOpen} onClick={() => setActiveTab('crew')} />
                <NavButton icon={<SettingsIcon />} label="Settings & Logs" isActive={activeTab === 'settings'} isDrawerOpen={isDrawerOpen} onClick={() => setActiveTab('settings')} />
            </nav>

            {/* Interactions */}
            <div className="flex-shrink-0 p-4 space-y-2 border-t border-white/10">
                <h3 className={`text-xs font-semibold text-gray-500 tracking-wider uppercase transition-all duration-300 ${isDrawerOpen ? 'px-4 mb-2' : 'text-center'}`}>
                    {isDrawerOpen ? "Interactions" : "AI"}
                </h3>
                <InteractionButton icon={<MicrophoneIcon />} label="Talk to MAITRI" isDrawerOpen={isDrawerOpen} isPrimary />
                <InteractionButton icon={<UsersIcon />} label="Scan Crew" isDrawerOpen={isDrawerOpen} />
                <InteractionButton icon={<SparklesIcon />} label="Relaxation" isDrawerOpen={isDrawerOpen} />
                <InteractionButton icon={<DocumentTextIcon />} label="Summary" isDrawerOpen={isDrawerOpen} />
            </div>
            
            {/* Logout & Toggle Button */}
            <div className="flex-shrink-0 p-4 border-t border-white/10">
                <button
                    onClick={onLogout}
                    className={`w-full flex items-center gap-4 px-4 py-3 mb-2 rounded-lg text-left transition-all duration-200 group text-red-400 hover:bg-red-500/20 hover:text-red-300 ${!isDrawerOpen && 'justify-center'}`}
                    aria-label="Logout"
                >
                    <LogoutIcon className="w-5 h-5" />
                    <span className={`transition-opacity duration-200 whitespace-nowrap ${isDrawerOpen ? 'opacity-100' : 'opacity-0 h-0 w-0'}`}>Logout</span>
                </button>
                <button 
                    onClick={() => setIsDrawerOpen(!isDrawerOpen)} 
                    className="w-full flex items-center p-3 rounded-lg text-gray-400 hover:bg-white/10 hover:text-white transition-colors"
                    aria-label={isDrawerOpen ? 'Collapse sidebar' : 'Expand sidebar'}
                >
                   <span className={`flex-grow text-left transition-opacity duration-200 whitespace-nowrap ${isDrawerOpen ? 'opacity-100' : 'opacity-0 h-0 w-0'}`}>Collapse</span>
                   <ChevronDoubleLeftIcon className={`w-5 h-5 transition-transform duration-300 ${!isDrawerOpen && 'rotate-180'}`} />
                </button>
            </div>
        </div>
    </aside>
  );
};