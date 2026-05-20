@echo off
chcp 65001 >nul
title Image Generation Agent — Portable Edition

echo ========================================
echo   Image Generation Agent — Portable
echo   Cloud-only edition, no GPU required
echo ========================================
echo.

cd /d "%~dp0"

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+ and add it to PATH.
    echo         https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Create venv if missing
if not exist ".venv" (
    echo [1/2] Creating virtual environment...
    python -m venv .venv
    echo.
)

REM Activate and install
call .venv\Scripts\activate.bat
echo [2/2] Installing dependencies...
pip install -q -r requirements-portable.txt
echo.

echo Starting server at http://localhost:8000
echo Open your browser and fill in API keys under the "设置" tab.
echo Press Ctrl+C to stop.
echo.

python app.py
pause
