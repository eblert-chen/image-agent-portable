#!/bin/bash
# Image Generation Agent — macOS launcher
cd "$(dirname "$0")"

echo "========================================"
echo "  Image Generation Agent"
echo "========================================"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3 not found."
    echo "        Install from https://www.python.org/downloads/"
    read -p "Press Enter to exit..."
    exit 1
fi

# Create venv if missing
if [ ! -d ".venv" ]; then
    echo "[1/2] Creating virtual environment..."
    python3 -m venv .venv
    echo ""
fi

# Activate and install
source .venv/bin/activate
echo "[2/2] Installing dependencies..."
pip install -q -r requirements-portable.txt
echo ""

echo "Starting server at http://localhost:8000"
echo "Press Ctrl+C to stop."
echo ""

# Open browser
sleep 1
open http://localhost:8000 2>/dev/null &

python app.py
