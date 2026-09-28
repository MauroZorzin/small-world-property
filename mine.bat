@echo off
setlocal enabledelayedexpansion
cd /d %~dp0

:: Batch equivalent of mine.ps1: runs Depends on every project folder under
:: repos\ and writes <name>-file.dot / <name>-file.json to the output root.
:: Skips a folder if its .dot output already exists, so this can be re-run
:: after cloning new projects without re-extracting the ones already done.
::
:: Depends writes the output using an absolute Windows path (D:\...) as if it
:: were relative, duplicating the current directory into the result (e.g.
:: D:\...\D:\...\out-file.json) and crashing. Passing a path relative to the
:: cwd instead avoids this, hence the cd /d above and the relative paths below.

set TOOL_PATH=depends-0.9.7-package-20221104a\depends-0.9.7\depends.bat
set REPO_ROOT=repos
set OUTPUT_ROOT=depends_9_16_2026_out

if not exist "%OUTPUT_ROOT%" mkdir "%OUTPUT_ROOT%"

for /d %%F in ("%REPO_ROOT%\*") do (
    set FOLDER_NAME=%%~nxF
    if exist "%OUTPUT_ROOT%\!FOLDER_NAME!-file.dot" (
        echo Skipping !FOLDER_NAME!, output already exists
    ) else (
        echo Processing folder: !FOLDER_NAME!
        call "%TOOL_PATH%" java "%%F" "%OUTPUT_ROOT%\!FOLDER_NAME!" -g=file "-p=windows" -s "-f=json,dot" "--type-filter=Import,Call,Return,Throw,Implement,Extend,Create,Use,Cast,Annotation"
    )
)

echo ----------------------------------------------
echo Batch processing complete.
pause
