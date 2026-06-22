@echo off
echo =========================================
echo  GoldBot NSSM Service Installer
echo =========================================
echo.
echo Please ensure nssm.exe is in your PATH.
echo Installing GoldBot_Main and GoldBot_Dashboard as Windows Services...

set "PROJECT_DIR=%~dp0.."
set "PYTHON_EXE=%PROJECT_DIR%\venv\Scripts\python.exe"

:: 1. Install Main Loop Service
nssm install GoldBot_Main "%PYTHON_EXE%"
nssm set GoldBot_Main AppParameters "-m src.main"
nssm set GoldBot_Main AppDirectory "%PROJECT_DIR%"
nssm set GoldBot_Main AppStdout "%PROJECT_DIR%\logs\service_main.log"
nssm set GoldBot_Main AppStderr "%PROJECT_DIR%\logs\service_main_err.log"
nssm set GoldBot_Main AppExit Default Restart
nssm set GoldBot_Main Description "Automated Gold Trading Bot (Main Loop)"

:: 2. Install Dashboard Service
nssm install GoldBot_Dashboard "%PYTHON_EXE%"
nssm set GoldBot_Dashboard AppParameters "-m src.dashboard.app"
nssm set GoldBot_Dashboard AppDirectory "%PROJECT_DIR%"
nssm set GoldBot_Dashboard AppStdout "%PROJECT_DIR%\logs\service_dashboard.log"
nssm set GoldBot_Dashboard AppStderr "%PROJECT_DIR%\logs\service_dashboard_err.log"
nssm set GoldBot_Dashboard AppExit Default Restart
nssm set GoldBot_Dashboard Description "Premium Web Dashboard for GoldBot"

echo.
echo Services installed successfully!
echo To start them, open Services.msc or run:
echo nssm start GoldBot_Main
echo nssm start GoldBot_Dashboard
echo.
pause
