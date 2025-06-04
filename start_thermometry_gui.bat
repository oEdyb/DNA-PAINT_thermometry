@echo off
echo ===================================
echo    Thermometry Analysis GUI
echo ===================================
echo.

REM Check if Python is available
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python 3.8+ first.
    echo Download from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo Found Python installation.
echo Checking for virtual environment...

REM Create virtual environment if it doesn't exist
if not exist "thermometry_env_new" (
    echo Creating virtual environment...
    python -m venv thermometry_env_new
)

REM Activate virtual environment
echo Activating virtual environment...
call thermometry_env_new\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing required packages...
pip install -r requirements.txt


echo.
echo Starting Thermometry GUI...
python parameter_GUI_v2.py

echo.
echo GUI closed. Press any key to exit...
pause 