@echo off
REM SENTRA live demo with a PUBLIC link: console (Cloudflare tunnel) + Telegram bot + one camera
cd /d "%~dp0"
start "SENTRA console (public)" cmd /k .venv\Scripts\python.exe scripts\run_console.py --tunnel
if exist config\telegram.yaml start "SENTRA Telegram bot" cmd /k .venv\Scripts\python.exe scripts\run_bot.py
timeout /t 5 >nul
.venv\Scripts\python.exe scripts\run_sentra.py --camera demo --config config\demo.yaml --realtime --loop --show
