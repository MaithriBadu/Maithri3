// Configuration for the frontend application
export const config = {
  api: {
    baseURL: (import.meta as any).env?.VITE_API_URL || 'http://localhost:3001/api',
  },
  socket: {
    url: (import.meta as any).env?.VITE_SOCKET_URL || 'http://localhost:3001',
  },
  app: {
    name: 'MAITRI',
    version: '1.0.0',
  },
};
