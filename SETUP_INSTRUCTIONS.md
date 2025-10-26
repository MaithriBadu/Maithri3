# MAITRI Setup Instructions

## Prerequisites Installation

### 1. Install Node.js and npm

**Download and install Node.js from:**
- Visit: https://nodejs.org/
- Download the LTS version (recommended)
- Run the installer
- Verify installation by running:
  ```bash
  node --version
  npm --version
  ```

### 2. Install MongoDB

**Option A: MongoDB Community Server**
- Visit: https://www.mongodb.com/try/download/community
- Download and install MongoDB Community Server
- Start MongoDB service

**Option B: MongoDB Atlas (Cloud)**
- Visit: https://www.mongodb.com/atlas
- Create a free account
- Create a cluster
- Get connection string

### 3. Python Dependencies (Already Installed)

The ML model dependencies are being installed. This includes:
- OpenCV for computer vision
- PyTorch for deep learning
- Transformers for NLP
- TensorFlow for additional ML capabilities

## Installation Steps

### Step 1: Install Backend Dependencies
```bash
cd backend
npm install
```

### Step 2: Install Frontend Dependencies
```bash
cd frontend
npm install
```

### Step 3: Install ML Model Dependencies
```bash
# ML Models
cd models/ml
pip install -r requirements.txt

# Speech Models
cd ../speech
pip install -r requirements.txt
```

### Step 4: Environment Configuration

Create a `.env` file in the `backend` directory:
```env
MONGODB_URI=mongodb://localhost:27017/maitri
JWT_SECRET=your-super-secret-jwt-key-here-change-this-in-production
PORT=3001
NODE_ENV=development
FRONTEND_URL=http://localhost:5173
```

### Step 5: Start the Application

**Option A: Use the startup script**
```bash
# Windows
start-dev.bat

# Linux/Mac
./start-dev.sh
```

**Option B: Manual startup**
```bash
# Terminal 1 - Backend
cd backend
npm run dev

# Terminal 2 - Frontend
cd frontend
npm run dev
```

## Access Points

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:3001
- **Health Check**: http://localhost:3001/api/health

## Troubleshooting

### Node.js Not Found
- Ensure Node.js is installed and added to PATH
- Restart your terminal/command prompt
- Try using `nvm` (Node Version Manager) if available

### MongoDB Connection Issues
- Ensure MongoDB is running
- Check connection string in `.env` file
- For local MongoDB: `mongodb://localhost:27017/maitri`
- For MongoDB Atlas: Use the provided connection string

### Python Dependencies Issues
- Ensure Python 3.8+ is installed
- Use virtual environment if needed:
  ```bash
  python -m venv venv
  venv\Scripts\activate  # Windows
  source venv/bin/activate  # Linux/Mac
  pip install -r requirements.txt
  ```

## Development Workflow

1. **Start MongoDB** (if using local instance)
2. **Start Backend**: `cd backend && npm run dev`
3. **Start Frontend**: `cd frontend && npm run dev`
4. **Access Application**: http://localhost:5173

## Features Available

✅ **User Authentication** - Register and login system  
✅ **Emotion Detection** - Multi-modal emotion analysis  
✅ **Real-time Updates** - WebSocket communication  
✅ **Database Persistence** - MongoDB integration  
✅ **ML Model Integration** - Python-based emotion detection  
✅ **Crew Monitoring** - Multi-user emotional state tracking  

## Next Steps

1. Install Node.js and npm
2. Install MongoDB
3. Run the installation commands above
4. Start the application
5. Create your first user account
6. Test the emotion detection features
