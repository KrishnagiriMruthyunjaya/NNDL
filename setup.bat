@echo off
echo =========================================
echo Emotion Detection System Setup
echo =========================================

REM Check Python version
echo Checking Python version...
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ Python found
    python --version
) else (
    echo ✗ Python not found! Please install Python 3.7+
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
if exist "emotion_env" (
    echo ✓ Virtual environment already exists
) else (
    python -m venv emotion_env
    echo ✓ Virtual environment created
)

REM Activate virtual environment
echo Activating virtual environment...
call emotion_env\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
pip install --upgrade pip

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt

echo =========================================
echo ✓ Setup completed successfully!
echo =========================================
echo.
echo To run the emotion detection system:
echo 1. Activate the virtual environment:
echo    emotion_env\Scripts\activate.bat
echo 2. Run the application:
echo    python EmotionDetection.py
echo.
echo Press 'q' to quit the application when running.
pause