@echo off
setlocal enabledelayedexpansion
cd /d %~dp0

:: Runs compute_real_metrics.py for every project cloned under repos\.
:: Split out of the old process-dot-files-modular.bat so each pipeline step
:: can be re-run on its own. DOT_DIR must match the OUTPUT_ROOT in mine.bat.

set DOT_DIR=depends_9_16_2026_out
set OUT_DIR=results

echo Computing real metrics for every project in repos\
echo ----------------------------------------------

for /d %%F in ("repos\*") do (
    set NAME=%%~nxF
    if exist "%DOT_DIR%\!NAME!-file.dot" (
        echo Processing !NAME!
        python compute_real_metrics.py "%DOT_DIR%\!NAME!-file.dot" --out-dir "%OUT_DIR%\!NAME!"
    ) else (
        echo Skipping !NAME!, no dot file found in %DOT_DIR% run mine.bat first
    )
)

echo ----------------------------------------------
echo Done. Results are in %OUT_DIR%
pause
