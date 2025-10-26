#!/bin/bash

echo "Starting MAITRI Development Environment..."
echo

echo "Starting Backend Server..."
cd backend
npm install
npm run dev &
BACKEND_PID=$!
cd ..

echo
echo "Starting Frontend Development Server..."
cd frontend
npm install
npm run dev &
FRONTEND_PID=$!
cd ..

echo
echo "Starting MongoDB (if not already running)..."
echo "Please ensure MongoDB is installed and running on localhost:27017"
echo

echo "Development servers started!"
echo "Backend: http://localhost:3001"
echo "Frontend: http://localhost:5173"
echo

# Function to cleanup on exit
cleanup() {
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit
}

# Trap Ctrl+C
trap cleanup INT

# Wait for user to stop
wait
