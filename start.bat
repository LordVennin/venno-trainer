@echo off
REM Start Venno Trainer (installs Node dependencies on first run).
cd /d "%~dp0"
if not exist node_modules (
    echo Installing dependencies...
    call npm install || exit /b 1
)
node server.js %*
pause
