@echo off
title InsightFace ArcFace 512D Service
cd /d "%~dp0"

echo ========================================================
echo Starting InsightFace ArcFace 512D Service on Port 8000
echo ========================================================
echo.

if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call "venv\Scripts\activate.bat"
)

python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
pause
