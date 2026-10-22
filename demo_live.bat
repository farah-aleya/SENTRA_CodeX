@echo off
REM SENTRA live demo: guard console + Telegram bot + one camera
cd /d "%~dp0"
start "SENTRA console" cmd /k .venv\Scripts\python.exe scripts\run_console.py
if exist config\telegram.yaml start "SENTRA Telegram bot" cmd /k .venv\Scripts\python.exe scripts\run_bot.py
timeout /t 3 >nul
.venv\Scripts\python.exe scripts\run_sentra.py --camera demo --config config\demo.yaml --realtime --loop --show
