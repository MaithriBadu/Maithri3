import React from 'react';
import { PowerIcon } from './IconComponents';

export const Header: React.FC = () => {
  return (
    <header className="flex-shrink-0 flex justify-end items-center bg-black/20 backdrop-blur-md border-b border-white/10" style={{ height: '73px' }}>
      <div className="flex items-center gap-6 px-8">
        <div className="flex items-center gap-2 text-sm text-yellow-400">
          <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
          <span>Local AI Mode</span>
        </div>
        <div className="flex items-center gap-2 text-sm text-green-400">
          <PowerIcon className="w-4 h-4" />
          <span>Core Status: Nominal</span>
        </div>
      </div>
    </header>
  );
};
