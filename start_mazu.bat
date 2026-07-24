@echo off
REM MAZU — Auto-start KG Dashboard + Streamlit Agent
REM Place a shortcut to this file in shell:startup to run on boot

cd /d f:\Saudi

echo Starting MAZU services...

REM 1. Start KG Dashboard in background
start "MAZU-KG" cmd /c "python dashboard/server.py"

REM 2. Wait for KG to initialize
timeout /t 5 /nobreak >nul

REM 3. Start Streamlit Agent
start "MAZU-Server" cmd /c "python server.py"

echo MAZU services started.
echo   KG Dashboard: http://127.0.0.1:5000
echo   Agent UI:     http://127.0.0.1:8501
