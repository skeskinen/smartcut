@echo off
REM TODO: Fix windows build
REM Navigate to the project directory; adjust this if your script is elsewhere
cd %~dp0

REM Package the project with PyInstaller
pyinstaller --distpath .\dist --workpath .\build --onedir --windowed ^
 --icon=icon.ico --add-data="smc/denoiser.onnx;smc" --add-data="smc/libmpv-2.dll;smc" --add-data="smc/watermark*.png:smc" ^
 --add-data="LICENSE:." --add-data="LICENSE.LGPL:." ^
 --additional-hooks-dir="hooks" -n smart-media-cutter -y smc\mainwindow.py

REM Pause the script to view any messages post-execution
pause
