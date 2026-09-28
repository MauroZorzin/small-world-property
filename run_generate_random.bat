@echo off
setlocal enabledelayedexpansion
cd /d %~dp0

:: Runs generate_random_sample.py for every project cloned under repos\.
:: Usage:
::   run_generate_random.bat count 100        fixed count per project
::   run_generate_random.bat converge 2.0      loop until under 2%% relative SEM
::   run_generate_random.bat                   defaults to count 10

set OUT_DIR=results
set MODE=%1
set VALUE=%2

if "%MODE%"=="" (
    set MODE=count
    set VALUE=10
)

if /i "%MODE%"=="count" (
    set EXTRA_ARGS=--count %VALUE%
) else if /i "%MODE%"=="converge" (
    set EXTRA_ARGS=--converge-threshold %VALUE%
) else (
    echo Usage: run_generate_random.bat [count N ^| converge PCT]
    exit /b 1
)

echo Generating random samples for every project in repos\ using: %EXTRA_ARGS%
echo ----------------------------------------------

for /d %%F in ("repos\*") do (
    set NAME=%%~nxF
    if exist "%OUT_DIR%\!NAME!\degree_sequence.json" (
        echo Processing !NAME!
        python generate_random_sample.py "%OUT_DIR%\!NAME!\degree_sequence.json" --out-dir "%OUT_DIR%\!NAME!" !EXTRA_ARGS!
    ) else (
        echo Skipping !NAME!, no degree_sequence.json found run run_compute_metrics.bat first
    )
)

echo ----------------------------------------------
echo Done.
pause
