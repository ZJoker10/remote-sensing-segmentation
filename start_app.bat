@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  Remote Sensing Segmentation — local web app
REM  Double-click to start. Opens http://127.0.0.1:7860 in your browser.
REM ─────────────────────────────────────────────────────────────────────────────
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo.
    echo   ERROR: virtual environment not found at venv\Scripts\python.exe
    echo.
    echo   Create it once with:
    echo       python -m venv venv
    echo       venv\Scripts\activate
    echo       pip install -r requirements.txt
    echo       pip install -r requirements.lock.txt
    echo.
    pause
    exit /b 1
)

echo Starting segmentation app...
venv\Scripts\python.exe app\server.py --port 7860 %*

if errorlevel 1 (
    echo.
    echo   The app exited with an error. Copy the message above for debugging.
    pause
)
endlocal
