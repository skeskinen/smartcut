@echo off
REM
REM Navigate to the project directory; adjust this if your script is elsewhere
cd %~dp0

REM Package the project with PyInstaller
pyinstaller --distpath .\dist --workpath .\build --onefile ^
 -n smartcut -y smartcut\__main__.py

..\sign.bat .\dist\smartcut.exe
REM Pause the script to view any messages post-execution
pause
