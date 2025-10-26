# MAITRI - Multi-modal AI Therapeutic Intelligent Agent

A comprehensive full-stack application that connects frontend, backend, and ML models for emotion detection and therapeutic assistance.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   ML Models    │
│   (React/TS)    │◄──►│   (Node.js)     │◄──►│   (Python)     │
│                 │    │                 │    │                 │
│ • Dashboard     │    │ • REST API      │    │ • Emotion ML    │
│ • Auth System   │    │ • WebSocket     │    │ • Speech AI     │
│ • Real-time UI  │    │ • MongoDB       │    │ • Text Analysis │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Features

- **Multi-modal Emotion Detection**: Face, voice, and text analysis
- **Real-time Communication**: WebSocket integration for live updates
- **User Authentication**: Secure JWT-based auth system
- **Emotion Logging**: Persistent storage of emotional states
- **Therapeutic Interface**: AI-powered therapeutic assistance
- **Crew Monitoring**: Multi-user emotional state tracking

## Tech Stack

### Frontend
- React 19 with TypeScript
- Vite for build tooling
- Socket.io-client for real-time communication
- Tailwind CSS for styling
- Recharts for data visualization

### Backend
- Node.js with Express
- MongoDB with Mongoose
- JWT authentication
- Socket.io for WebSocket support
- CORS enabled for cross-origin requests

### ML Models
- Python-based emotion detection
- Computer vision for facial emotion analysis
- Speech emotion recognition
- Text sentiment analysis

## Quick Start

### Prerequisites
- Node.js (v18+)
- MongoDB
- Python (v3.8+)

### Installation

1. **Clone and setup the project:**
```bash
git clone <repository-url>
cd maitri-pull
```

2. **Install dependencies:**
```bash
# Backend
cd backend
npm install

# Frontend
cd ../frontend
npm install

# ML Models (optional - for local ML processing)
cd ../models/ml
pip install -r requirements.txt
cd ../speech
pip install -r requirements.txt
```

3. **Start the development environment:**

**Windows:**
```bash
start-dev.bat
```

**Linux/Mac:**
```bash
chmod +x start-dev.sh
./start-dev.sh
```

**Manual start:**
```bash
# Terminal 1 - Backend
cd backend
npm run dev

# Terminal 2 - Frontend
cd frontend
npm run dev
```

### Environment Setup

Create a `.env` file in the `backend` directory:
```env
MONGODB_URI=mongodb://localhost:27017/maitri
JWT_SECRET=your-super-secret-jwt-key-here
PORT=3001
NODE_ENV=development
FRONTEND_URL=http://localhost:5173
```

## API Endpoints

### Authentication
- `POST /api/auth/login` - User login
- `POST /api/auth/signup` - User registration

### User Profile
- `GET /api/user/profile` - Get user profile
- `PUT /api/user/profile` - Update user profile
- `GET /api/user/health` - Get health status
- `POST /api/user/emotion-log` - Log emotion data
- `GET /api/user/emotion-history` - Get emotion history

### ML Integration
- `POST /api/ml/analyze-image` - Analyze emotion from image
- `POST /api/ml/analyze-audio` - Analyze emotion from audio
- `POST /api/ml/analyze-text` - Analyze emotion from text

## Project Structure

```
maitri-pull/
├── frontend/                 # React frontend
│   ├── components/       # React components
│   ├── services/          # API and WebSocket services
│   ├── types.ts          # TypeScript definitions
│   └── App.tsx           # Main application
├── backend/               # Node.js backend
│   └── Astronaut_backend/
│       ├── controllers/   # API controllers
│       ├── models/        # Database models
│       ├── routes/        # API routes
│       ├── middleware/    # Auth middleware
│       └── server.js     # Main server file
└── models/                # ML models
    ├── ml/               # Computer vision models
    └── speech/           # Speech emotion models
```

## Development

### Frontend Development
- Hot reload enabled with Vite
- TypeScript support
- Real-time WebSocket connection
- API service layer for backend communication

### Backend Development
- Express server with middleware
- MongoDB integration
- JWT authentication
- WebSocket support for real-time features
- ML model integration endpoints

### ML Model Integration
- Python scripts for emotion detection
- REST API endpoints for model inference
- Automatic emotion logging to database
- Support for multiple input modalities

## Production Deployment

1. **Build the frontend:**
```bash
cd frontend
npm run build
```

2. **Start the backend in production:**
```bash
cd backend
npm start
```

3. **Configure environment variables for production**

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.
