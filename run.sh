#!/usr/bin/env bash

# AI Job Application Chatbot Runner
# Usage:
#   ./run.sh          -> Starts both FastAPI backend and Chainlit frontend
#   ./run.sh backend  -> Starts FastAPI backend only
#   ./run.sh frontend -> Starts Chainlit frontend only

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Check .env file
if [ ! -f ".env" ]; then
    echo "⚠️ .env file not found. Creating from .env.example..."
    cp .env.example .env
    echo "ℹ️ Please make sure to add your OPENAI_API_KEY in the .env file!"
fi

MODE="${1:-all}"

start_backend() {
    echo "🚀 Starting FastAPI Backend at http://localhost:8000 (Docs: http://localhost:8000/docs)..."
    python3 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
}

start_frontend() {
    echo "🌟 Starting Chainlit UI at http://localhost:8501..."
    chainlit run chainlit_app/app.py --port 8501 -w
}

case "$MODE" in
    backend)
        start_backend
        ;;
    frontend)
        start_frontend
        ;;
    all)
        echo "=========================================================="
        echo "   Starting AI Job Application Chatbot (Full Stack)       "
        echo "=========================================================="
        
        # Start backend in background
        python3 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &
        BACKEND_PID=$!
        echo "✅ Backend started with PID $BACKEND_PID"

        # Wait for backend to be ready
        echo "⏳ Waiting for backend to initialize..."
        sleep 2

        # Trap SIGINT and SIGTERM to kill backend when frontend exits
        trap "echo 'Stopping backend...'; kill $BACKEND_PID 2>/dev/null; exit" SIGINT SIGTERM

        # Start Chainlit in foreground
        start_frontend

        # Cleanup
        kill $BACKEND_PID 2>/dev/null
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: ./run.sh [all|backend|frontend]"
        exit 1
        ;;
esac
