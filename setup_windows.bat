@echo off
REM SENTRA one-time setup for Windows with an NVIDIA RTX GPU
cd /d "%~dp0"
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
REM GPU build of PyTorch (CUDA 12.8). Needs an up-to-date NVIDIA driver.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(\"CUDA available:\", torch.cuda.is_available()); print(\"GPU:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none - update your NVIDIA driver\")"
pause
