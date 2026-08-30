@echo off
title Smart AI Drug Dispensing - Launcher
cd /d "%~dp0"

echo Starting Smart AI Drug Dispensing System...
echo.

REM =====================================
REM VENV SETUP (portable across machines)
REM =====================================
REM Uses a local .venv instead of a hardcoded, machine-specific Python path
REM so this script works on any machine that has Python + the "py" launcher
REM installed, regardless of where Python lives on disk.

set VENV_DIR=%~dp0.venv
set VENV_PY="%VENV_DIR%\Scripts\python.exe"

if not exist %VENV_PY% (
    echo No virtual environment found - creating one at .venv ...

    REM Pinned to 3.12: paddlepaddle/paddleocr in requirements.txt do not
    REM yet ship wheels for newer Python versions (e.g. 3.14), so a plain
    REM "py -3" here can silently pick an incompatible interpreter.
    py -3.12 -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to create the virtual environment with Python 3.12.
        echo Install Python 3.12 from https://www.python.org/downloads/
        echo ^(paddlepaddle/paddleocr do not yet support newer Python versions^).
        pause
        exit /b 1
    )

    echo Installing dependencies from requirements.txt ...
    %VENV_PY% -m pip install --upgrade pip >nul
    %VENV_PY% -m pip install -r "%~dp0requirements.txt"
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install dependencies.
        pause
        exit /b 1
    )
)

REM =====================================
REM START SERVER
REM =====================================

start "Smart AI Drug Dispensing - SERVER (close this window to stop the app)" %VENV_PY% -m uvicorn backend.app:app --host 127.0.0.1 --port 8000

ping -n 4 127.0.0.1 >nul

start "" http://127.0.0.1:8000

echo.
echo The app should now be open in your browser at http://127.0.0.1:8000
echo.
echo A separate window titled "Smart AI Drug Dispensing - SERVER" is now running
echo the app. Leave that window open while you use the app. Close it whenever
echo you want to stop the app.
echo.
echo This window is safe to close now.
pause
