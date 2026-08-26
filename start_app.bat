@echo off
title Smart AI Drug Dispensing - Launcher
cd /d "%~dp0"

echo Starting Smart AI Drug Dispensing System...
echo.

start "Smart AI Drug Dispensing - SERVER (close this window to stop the app)" "C:\Users\veere\AppData\Local\Programs\Python\Python312\python.exe" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000

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
