import axios from 'axios';
import io from 'socket.io-client';

// Create axios instance
const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true,
});

// Add token to requests if available
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Socket.IO instance
let socket: ReturnType<typeof io> | null = null;

export const connectSocket = () => {
  const token = localStorage.getItem('token');
  if (!token) return null;

  socket = io('/', {
    path: '/socket.io',
    auth: { token },
    transports: ['websocket', 'polling'],
  });

  socket.on('connect', () => {
    console.log('Socket connected');
  });

  socket.on('disconnect', () => {
    console.log('Socket disconnected');
  });

  return socket;
};

export const getSocket = () => socket;

export const disconnectSocket = () => {
  if (socket) {
    socket.disconnect();
    socket = null;
  }
};

// Auth API
export const auth = {
  login: (credentials: { email: string; password: string }) => 
    api.post('/auth/login', credentials),
  register: (userData: { email: string; password: string; name: string }) => 
    api.post('/auth/register', userData),
  logout: () => api.post('/auth/logout'),
  getProfile: () => api.get('/user/profile'),
};

// Health check
export const checkHealth = () => api.get('/health');

// ML Analysis API
export const ml = {
  analyzeImage: (imageData: string) => api.post('/ml/analyze-image', { imageData }),
  analyzeAudio: (audioData: string) => api.post('/ml/analyze-audio', { audioData }),
  analyzeText: (text: string) => api.post('/ml/analyze-text', { text }),
};

export default api;