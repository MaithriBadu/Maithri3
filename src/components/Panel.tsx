import React from 'react';

interface PanelProps {
  title: string;
  children: React.ReactNode;
  className?: string;
  titleClassName?: string;
}

export const Panel: React.FC<PanelProps> = ({ title, children, className, titleClassName }) => {
  return (
    <div className={`bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 shadow-2xl shadow-black/20 ${className}`}>
      <h2 className={`text-xl font-bold text-purple-300 mb-4 tracking-wide ${titleClassName}`}>{title}</h2>
      {children}
    </div>
  );
};