@echo off
setlocal

REM Create venv if missing
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

if not exist ".venv\Scripts\activate.bat" (
    echo Failed to create virtual environment.
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup complete. Activate with:
echo   call .venv\Scripts\activate.bat
echo Then run: run.bat

endlocal
