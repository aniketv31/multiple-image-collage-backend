@echo off
title Asset API - LAN (Wi-Fi)
cd /d "%~dp0"

echo.
echo  STOP any other uvicorn window first (Ctrl+C), then this starts LAN mode.
echo.

for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000.*LISTENING"') do (
  echo  Port 8000 in use by PID %%a - stopping it...
  taskkill /F /PID %%a >nul 2>&1
)

if exist ".venv\Scripts\python.exe" (
  .venv\Scripts\python.exe serve.py
) else (
  python serve.py
)

pause
