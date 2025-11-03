@echo off
title CCIT Face Attendance Monitoring System
color 0A
echo =======================================================
echo    🎓 CCIT Face Attendance Monitoring System Launcher
echo =======================================================
cd /d "%~dp0"

:: Step 1 — Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b
)

:: Step 2 — Create venv if not found
if not exist "venv\" (
    echo 🧩 Creating virtual environment...
    python -m venv venv
)

:: Step 3 — Activate venv
call "%~dp0venv\Scripts\activate"

:: Step 4 — Ensure pip is up to date
echo 🔄 Updating pip...
python -m ensurepip --upgrade >nul 2>&1
python -m pip install --upgrade pip >nul 2>&1

:: Step 5 — Install requirements
echo 🧰 Checking and installing dependencies...
if exist "requirements.txt" (
    python -m pip install -r requirements.txt --quiet
) else (
    echo ⚠️ requirements.txt not found — skipping dependency install.
)

:: Step 6 — Check models folder
if not exist "models\resnet34_final.pth" (
    echo ⚠️ Model file not found: models\resnet34_final.pth
    echo Make sure to include your anti-spoofing model in the models\ folder.
    pause
)

:: Step 7 — Run the main app
echo 🚀 Launching Attendance App...
python attendance_app.py

:: Step 8 — After exit
echo.
echo ✅ Application closed. Goodbye!
pause
