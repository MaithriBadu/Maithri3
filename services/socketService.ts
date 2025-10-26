import io from 'socket.io-client';

interface SocketService {
  socket: ReturnType<typeof io> | null;
  connect: () => void;
  disconnect: () => void;
  emit: (event: string, data: any) => void;
  on: (event: string, callback: (data: any) => void) => void;
  off: (event: string, callback?: (data: any) => void) => void;
}

class SocketServiceClass implements SocketService {
  socket: ReturnType<typeof io> | null = null;
  private serverURL: string;

  constructor() {
    this.serverURL = (import.meta as any).env?.VITE_SOCKET_URL || 'http://localhost:3001';
  }

  connect() {
    if (this.socket?.connected) return;

    this.socket = io(this.serverURL, {
      autoConnect: true,
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    this.socket.on('connect', () => {
      console.log('🔌 Connected to server');
    });

    this.socket.on('disconnect', () => {
      console.log('🔌 Disconnected from server');
    });

    this.socket.on('connect_error', (error) => {
      console.error('🔌 Connection error:', error);
    });
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  emit(event: string, data: any) {
    if (this.socket?.connected) {
      this.socket.emit(event, data);
    } else {
      console.warn('Socket not connected, cannot emit event:', event);
    }
  }

  on(event: string, callback: (data: any) => void) {
    if (this.socket) {
      this.socket.on(event, callback);
    }
  }

  off(event: string, callback?: (data: any) => void) {
    if (this.socket) {
      if (callback) {
        this.socket.off(event, callback);
      } else {
        this.socket.off(event);
      }
    }
  }
}

export const socketService = new SocketServiceClass();
export default socketService;
