@echo off
title Stock Scanner
cd /d "%~dp0"

REM ---- 1. Find Python (install it automatically if missing) ----
set "PY="
where py >nul 2>nul && set "PY=py"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Launcher\py.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Launcher\py.exe"
if not defined PY (
  echo Python is not installed. Installing it now - please wait, this takes a few minutes...
  winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
  if exist "%LOCALAPPDATA%\Programs\Python\Launcher\py.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Launcher\py.exe"
  if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" if not defined PY set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
)
if not defined PY (
  echo.
  echo Could not install Python automatically.
  echo Please install it from https://www.python.org/downloads/  and tick "Add python.exe to PATH".
  echo Then double-click this file again.
  pause
  exit /b
)

REM ---- 2. Install the app's packages (first run only) ----
"%PY%" -c "import streamlit, yfinance, plotly, pypdf" >nul 2>nul
if errorlevel 1 (
  echo Installing the app's packages - first run only, please wait...
  "%PY%" -m pip install --upgrade pip
  "%PY%" -m pip install streamlit yfinance pandas numpy plotly requests pypdf
)

REM ---- 3. Start the app ----
echo.
echo Starting the Stock Scanner... (close this window to stop it)
"%PY%" -m streamlit run scanner_app.py
pause
