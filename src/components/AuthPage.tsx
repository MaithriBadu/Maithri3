import React, { useState } from 'react';
import { Panel } from './Panel';
import { AtSymbolIcon, LockClosedIcon, UserIcon } from './IconComponents';

interface AuthPageProps {
  onLoginSuccess: () => void;
}

const InputField: React.FC<{ icon: React.ReactNode; type: string; placeholder: string; id: string; }> = ({ icon, type, placeholder, id }) => (
  <div className="relative">
    <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none text-gray-400">
      {icon}
    </div>
    <input
      type={type}
      id={id}
      className="w-full pl-10 pr-4 py-3 bg-black/30 border border-white/20 rounded-lg placeholder-gray-500 text-white focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
      placeholder={placeholder}
    />
  </div>
);

export const AuthPage: React.FC<AuthPageProps> = ({ onLoginSuccess }) => {
  const [isLoginView, setIsLoginView] = useState(true);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Simulate successful authentication
    onLoginSuccess();
  };

  return (
    <div className="relative z-10 flex items-center justify-center min-h-screen p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
            <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-2xl shadow-lg mx-auto mb-4"></div>
            <h1 className="text-4xl font-bold tracking-wider text-purple-300">
                M<span className="text-gray-300">A</span>ITRI
            </h1>
            <p className="text-gray-400 mt-2">Multi-modal AI Therapeutic Intelligent Agent</p>
        </div>

        <Panel title={isLoginView ? 'Astronaut Login' : 'Create Account'} className="w-full">
          <form className="space-y-6" onSubmit={handleSubmit}>
            {!isLoginView && (
              <InputField icon={<UserIcon className="w-5 h-5" />} type="text" placeholder="Full Name" id="full-name" />
            )}
            <InputField icon={<AtSymbolIcon className="w-5 h-5" />} type="text" placeholder="Astronaut ID" id="astronaut-id" />
            <InputField icon={<LockClosedIcon className="w-5 h-5" />} type="password" placeholder="Password" id="password" />
            {!isLoginView && (
              <InputField icon={<LockClosedIcon className="w-5 h-5" />} type="password" placeholder="Confirm Password" id="confirm-password" />
            )}
            
            <button
              type="submit"
              className="w-full py-3 font-semibold text-white bg-purple-600 rounded-lg hover:bg-purple-700 transition-colors focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 focus:ring-offset-gray-900 shadow-lg shadow-purple-900/40"
            >
              {isLoginView ? 'Access MAITRI Core' : 'Register & Engage'}
            </button>
          </form>

          <div className="mt-6 text-center">
            <button onClick={() => setIsLoginView(!isLoginView)} className="text-sm text-purple-400 hover:text-purple-300 transition-colors">
              {isLoginView ? 'Need an account? Sign Up' : 'Already registered? Login'}
            </button>
          </div>
        </Panel>
      </div>
    </div>
  );
};