@echo off
REM Double-click this file to launch the demo.
REM Works regardless of where the project folder lives, because it changes to
REM its own directory rather than assuming a fixed path -- which matters on
REM machines where OneDrive has redirected the Desktop.

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo.
    echo   No virtual environment found in this folder.
    echo   Run the one-time setup first:
    echo.
    echo       py -m venv .venv
    echo       .venv\Scripts\activate
    echo       pip install -r requirements.txt
    echo       python -m src.download_weights
    echo.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

if not exist "outputs\models\efficientnet_best.pt" (
    echo.
    echo   Trained weights are missing. Fetching them now...
    echo.
    python -m src.download_weights
)

echo.
echo   Starting the app. A browser window will open shortly.
echo   Leave this window open while you use it; press Ctrl+C to stop.
echo.
streamlit run app/app.py

pause
