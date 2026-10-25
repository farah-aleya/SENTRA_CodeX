@echo off
REM SENTRA live demo: guard console + one camera
cd /d "%~dp0"
start "SENTRA console" cmd /k .venv\Scripts\python.exe scripts\run_console.py
timeout /t 3 >nul
.venv\Scripts\python.exe scripts\run_sentra.py --camera gate --config config\showcase.yaml --realtime --loop --show
