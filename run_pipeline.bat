@echo off
REM Dependency Extraction Script
REM Extracts dependency graphs from code repositories

echo ================================================================================
echo   Dependency Graph Extraction
echo ================================================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found
    pause
    exit /b 1
)

REM Check dependencies
python -c "import networkx, pydot, yaml, tqdm" >nul 2>&1
if errorlevel 1 (
    echo Installing Python packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Check depends tool
if not exist "depends-0.9.7-package-20221104a\depends.exe" (
    echo ERROR: depends.exe not found
    echo Please extract depends-0.9.7-package-20221104a.zip
    pause
    exit /b 1
)

REM Check repos folder
if not exist "repos" (
    echo WARNING: repos folder not found
    mkdir repos
    echo Please add repositories to repos\ folder
    pause
    exit /b 1
)

echo Starting extraction...
echo.

python extract_dependencies.py %*

if errorlevel 1 (
    echo.
    echo ERROR: Extraction failed
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo   Extraction completed!
echo   .dot files saved to: depends-out\
echo ================================================================================
echo.

pause