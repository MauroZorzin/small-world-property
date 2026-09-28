@echo off
setlocal enabledelayedexpansion
cd /d %~dp0

:: Runs combine_results.py for every project cloned under repos\.

set OUT_DIR=results

echo Combining results for every project in repos\
echo ----------------------------------------------

for /d %%F in ("repos\*") do (
    set NAME=%%~nxF
    if exist "%OUT_DIR%\!NAME!\real_metrics.csv" (
        echo Processing !NAME!
        python combine_results.py --project-dir "%OUT_DIR%\!NAME!"
    ) else (
        echo Skipping !NAME!, no real_metrics.csv found run run_compute_metrics.bat first
    )
)

echo ----------------------------------------------
echo Done. Results are in %OUT_DIR%
pause
