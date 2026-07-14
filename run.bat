@echo off
title Genix Launcher

echo ===================================================
echo               Launching Genix
echo ===================================================
echo.

cd /d "%~dp0"

if exist venv\Scripts\activate.bat (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment not found.
    pause
    exit /b
)

echo.
echo [INFO] Starting Genix...
echo.

python app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application crashed.
    pause
)