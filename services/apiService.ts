const API_BASE_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:3001/api';

interface ApiResponse<T> {
  data?: T;
  message?: string;
  error?: string;
}

class ApiService {
  private baseURL: string;
  private token: string | null = null;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
    this.token = localStorage.getItem('authToken');
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseURL}${endpoint}`;
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string>),
    };

    if (this.token) {
      headers.Authorization = `Bearer ${this.token}`;
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || 'Request failed');
      }

      return { data };
    } catch (error) {
      return { error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  // Authentication methods
  async login(email: string, password: string) {
    const response = await this.request<{ token: string; user: any }>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });

    if (response.data?.token) {
      this.token = response.data.token;
      localStorage.setItem('authToken', this.token);
    }

    return response;
  }

  async signup(name: string, email: string, password: string) {
    return this.request<{ user: any }>('/auth/signup', {
      method: 'POST',
      body: JSON.stringify({ name, email, password }),
    });
  }

  async logout() {
    this.token = null;
    localStorage.removeItem('authToken');
  }

  // User profile methods
  async getUserProfile() {
    return this.request<any>('/user/profile');
  }

  async updateUserProfile(profileData: any) {
    return this.request<any>('/user/profile', {
      method: 'PUT',
      body: JSON.stringify(profileData),
    });
  }

  // Emotion logging methods
  async logEmotion(emotionData: {
    source: 'face' | 'voice' | 'text';
    emotion: string;
    stressLevel?: 'low' | 'medium' | 'high';
    confidence?: number;
  }) {
    return this.request<any>('/user/emotion-log', {
      method: 'POST',
      body: JSON.stringify(emotionData),
    });
  }

  async getEmotionHistory(limit: number = 50) {
    return this.request<any[]>(`/user/emotion-history?limit=${limit}`);
  }

  // ML Model integration methods
  async analyzeEmotionFromImage(imageData: string) {
    return this.request<{ emotion: string; confidence: number }>('/ml/analyze-image', {
      method: 'POST',
      body: JSON.stringify({ imageData }),
    });
  }

  async analyzeEmotionFromAudio(audioData: string) {
    return this.request<{ emotion: string; confidence: number }>('/ml/analyze-audio', {
      method: 'POST',
      body: JSON.stringify({ audioData }),
    });
  }

  async analyzeEmotionFromText(text: string) {
    return this.request<{ emotion: string; confidence: number }>('/ml/analyze-text', {
      method: 'POST',
      body: JSON.stringify({ text }),
    });
  }

  // Health check
  async healthCheck() {
    return this.request<{ status: string; timestamp: string }>('/health');
  }
}

export const apiService = new ApiService();
export default apiService;
