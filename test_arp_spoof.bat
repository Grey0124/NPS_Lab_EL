@echo off
echo ========================================
echo ARP Spoofing Test for Windows
echo ========================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo Checking for required packages...
python -c "import scapy" >nul 2>&1
if errorlevel 1 (
    echo Installing scapy package...
    pip install scapy
    if errorlevel 1 (
        echo ERROR: Failed to install scapy
        echo Please run: pip install scapy
        pause
        exit /b 1
    )
)

echo.
echo ========================================
echo Starting ARP Spoofing Test
echo ========================================
echo.
echo IMPORTANT: Run this script as Administrator!
echo.
echo Make sure your ARP Guardian is running and monitoring.
echo.
pause

echo Running test script...
python arp_test_script.py

echo.
echo Test completed!
pause 