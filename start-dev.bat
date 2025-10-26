@echo off
echo Starting MAITRI Development Environment...
echo.

echo Starting Backend Server...
cd backend
start cmd /k "npm install && npm run dev"
cd ..

echo.
echo Starting Frontend Development Server...
cd frontend
start cmd /k "npm install && npm run dev"
cd ..

echo.
echo Starting MongoDB (if not already running)...
echo Please ensure MongoDB is installed and running on localhost:27017
echo.

echo Development servers started!
echo Backend: http://localhost:3001
echo Frontend: http://localhost:5173
echo.
pause
