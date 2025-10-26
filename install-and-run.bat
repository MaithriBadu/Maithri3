@echo off
echo ========================================
echo MAITRI Installation and Setup
echo ========================================
echo.

echo Checking for Node.js installation...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed!
    echo.
    echo Please install Node.js from: https://nodejs.org/
    echo Download the LTS version and run the installer.
    echo After installation, restart this script.
    echo.
    pause
    exit /b 1
) else (
    echo ✅ Node.js is installed
    node --version
)

echo.
echo Checking for MongoDB...
echo Please ensure MongoDB is running on localhost:27017
echo If using MongoDB Atlas, update the .env file with your connection string.
echo.

echo Installing Backend Dependencies...
cd backend
call npm install
if %errorlevel% neq 0 (
    echo ❌ Failed to install backend dependencies
    pause
    exit /b 1
)
echo ✅ Backend dependencies installed

echo.
echo Installing Frontend Dependencies...
cd ..\frontend
call npm install
if %errorlevel% neq 0 (
    echo ❌ Failed to install frontend dependencies
    pause
    exit /b 1
)
echo ✅ Frontend dependencies installed

echo.
echo Creating environment file...
cd ..\backend
if not exist .env (
    echo MONGODB_URI=mongodb://localhost:27017/maitri > .env
    echo JWT_SECRET=your-super-secret-jwt-key-here-change-this-in-production >> .env
    echo PORT=3001 >> .env
    echo NODE_ENV=development >> .env
    echo FRONTEND_URL=http://localhost:5173 >> .env
    echo ✅ Environment file created
) else (
    echo ✅ Environment file already exists
)

echo.
echo ========================================
echo Starting MAITRI Application
echo ========================================
echo.
echo Starting Backend Server...
start cmd /k "cd backend && npm run dev"

echo Waiting 3 seconds for backend to start...
timeout /t 3 /nobreak >nul

echo Starting Frontend Server...
start cmd /k "cd frontend && npm run dev"

echo.
echo ✅ Application started successfully!
echo.
echo 🌐 Frontend: http://localhost:5173
echo 🔧 Backend: http://localhost:3001
echo 📊 Health Check: http://localhost:3001/api/health
echo.
echo Press any key to exit this window...
pause >nul
