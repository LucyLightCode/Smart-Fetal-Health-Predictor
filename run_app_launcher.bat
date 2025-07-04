@echo off
chcp 65001 >nul
title Smart Fetal Health App Launcher
color 0B

:: ---------------------
:: 🐳 Check Docker status
:: ---------------------
docker info >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Docker is not running. Launching Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    echo ⏳ Waiting for Docker to start...
    :WAIT_DOCKER
    timeout /t 5 >nul
    docker info >nul 2>&1
    if errorlevel 1 goto WAIT_DOCKER
)
echo ✅ Docker is running!

:: ---------------------
:: 🧠 App Menu
:: ---------------------
:MENU
cls
echo ================================================
echo         🧠 Smart Fetal Health App Launcher
echo ================================================
echo.
echo Choose the app to run:
echo.
echo [1] 🚀 Flask API
echo [2] 📊 Streamlit Dashboard
echo [0] 🛑 Exit
echo.

set /p choice=Enter your choice [0-2]:

if "%choice%"=="1" goto FLASK
if "%choice%"=="2" goto STREAMLIT
if "%choice%"=="0" exit
goto MENU

:FLASK
echo 🔄 Killing any process on port 5000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5000') do (
    taskkill /F /PID %%a >nul 2>&1
)
echo ✅ Launching Flask...
start "" http://localhost:5000
docker run -p 5000:5000 -e APP=flask fetal_health_app
pause
goto MENU

:STREAMLIT
echo 🔄 Killing any process on port 8501...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8501') do (
    taskkill /F /PID %%a >nul 2>&1
)
echo ✅ Launching Streamlit...
start "" http://localhost:8501
docker run -p 8501:8501 -e APP=streamlit fetal_health_app
pause
goto MENU
