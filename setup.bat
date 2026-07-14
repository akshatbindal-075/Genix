@echo off
title Genix Setup

color 0A

echo =====================================================
echo                GENIX INITIAL SETUP
echo =====================================================
echo.

:: -------------------------------------------------------
:: Check Python Installation
:: -------------------------------------------------------

python --version >nul 2>&1

if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed.
    echo Please install Python 3.11.x and try again.
    pause
    exit /b
)

echo [1/8] Python detected.
echo.

:: -------------------------------------------------------
:: Create Virtual Environment
:: -------------------------------------------------------

if not exist "venv" (
    echo [2/8] Creating Virtual Environment...
    python -m venv venv
) else (
    echo [2/8] Virtual Environment already exists.
)

echo.

:: -------------------------------------------------------
:: Activate Virtual Environment
:: -------------------------------------------------------

call venv\Scripts\activate

echo [3/8] Virtual Environment Activated.
echo.

:: -------------------------------------------------------
:: Upgrade pip
:: -------------------------------------------------------

echo [4/8] Upgrading pip...

python -m pip install --upgrade pip

echo.

:: -------------------------------------------------------
:: Install PyTorch (CUDA 12.1)
:: -------------------------------------------------------

echo [5/8] Installing PyTorch...

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] PyTorch installation failed.
    pause
    exit /b
)

echo.

:: -------------------------------------------------------
:: Install Project Requirements
:: -------------------------------------------------------

echo [6/8] Installing Project Dependencies...

pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Dependency installation failed.
    pause
    exit /b
)

echo.

:: -------------------------------------------------------
:: Create Project Directories
:: -------------------------------------------------------

if not exist "outputs" mkdir outputs
if not exist "models" mkdir models

echo [7/8] Project directories verified.
echo.

:: -------------------------------------------------------
:: Download Models
:: -------------------------------------------------------

echo [8/8] Downloading AI Models...
echo.
echo NOTE:
echo The first setup may download several GB of models.
echo Please wait...
echo.

python download_models.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Model download failed.
    pause
    exit /b
)

echo.
echo =====================================================
echo             GENIX SETUP COMPLETED
echo =====================================================
echo.
echo You can now launch the application using:
echo.
echo                run.bat
echo.
pause