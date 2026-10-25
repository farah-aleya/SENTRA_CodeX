@echo off
REM SENTRA live demo with a PUBLIC link: console (Cloudflare tunnel) + one camera
cd /d "%~dp0"
start "SENTRA console (public)" cmd /k .venv\Scripts\python.exe scripts\run_console.py --tunnel
timeout /t 5 >nul
.venv\Scripts\python.exe scripts\run_sentra.py --camera gate --config config\showcase.yaml --realtime --loop --show
