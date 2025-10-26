# 🚀 MAITRI Quick Installation Guide

## Current Status
✅ **Python ML Dependencies**: Already installed and ready  
❌ **Node.js**: Not installed (required for frontend/backend)  
❓ **MongoDB**: Status unknown (required for database)  

## Step 1: Install Node.js (REQUIRED)

### Download and Install Node.js
1. **Visit**: https://nodejs.org/
2. **Download**: LTS version (recommended for most users)
3. **Install**: Run the downloaded installer
4. **Restart**: Close and reopen your terminal/command prompt
5. **Verify**: Run `node --version` in terminal

### Alternative: Using Chocolatey (if you have it)
```powershell
choco install nodejs
```

### Alternative: Using Winget (Windows 10/11)
```powershell
winget install OpenJS.NodeJS
```

## Step 2: Install MongoDB (REQUIRED)

### Option A: MongoDB Community Server (Local)
1. **Visit**: https://www.mongodb.com/try/download/community
2. **Download**: MongoDB Community Server
3. **Install**: Run installer with default settings
4. **Start**: MongoDB service should start automatically

### Option B: MongoDB Atlas (Cloud - Easier)
1. **Visit**: https://www.mongodb.com/atlas
2. **Sign up**: Create free account
3. **Create cluster**: Choose free tier
4. **Get connection string**: Copy the connection string
5. **Update .env**: Replace `mongodb://localhost:27017/maitri` with your Atlas connection string

## Step 3: Run the Application

After installing Node.js and MongoDB:

### Option A: Automated (Recommended)
```bash
# Simply run:
install-and-run.bat
```

### Option B: Manual
```bash
# Terminal 1 - Backend
cd backend
npm install
npm run dev

# Terminal 2 - Frontend  
cd frontend
npm install
npm run dev
```

## Step 4: Access the Application

- **Frontend**: http://localhost:5173
- **Backend**: http://localhost:3001
- **Health Check**: http://localhost:3001/api/health

## What's Already Ready

✅ **Complete Application Code**
- Frontend React app with TypeScript
- Backend Express server with MongoDB
- ML models for emotion detection
- WebSocket real-time communication
- JWT authentication system

✅ **Python ML Dependencies**
- OpenCV for computer vision
- PyTorch for deep learning
- Transformers for NLP
- All emotion detection models ready

## Troubleshooting

### Node.js Issues
- Ensure Node.js is in your PATH
- Restart terminal after installation
- Try running `npm --version` to verify

### MongoDB Issues
- For local: Ensure MongoDB service is running
- For Atlas: Check connection string and network access
- Test connection: `mongosh "your-connection-string"`

### Port Conflicts
- Backend uses port 3001
- Frontend uses port 5173
- Change ports in package.json if needed

## Next Steps After Installation

1. **Create User Account**: Register your first user
2. **Test Authentication**: Login/logout functionality
3. **Test Emotion Detection**: Upload images or record audio
4. **Explore Features**: Dashboard, crew monitoring, settings

The application is fully integrated and ready to run once Node.js and MongoDB are installed!
