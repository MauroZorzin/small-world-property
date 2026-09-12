@echo off
setlocal

:: Settings
set REAL_SCRIPT=compute_real_metrics.py
set RAND_SCRIPT=generate_random_sample.py
set COMBINE_SCRIPT=combine_results.py
set COUNT=10
set DOT_DIR=D:\GITHUB\small-world-property\depends_9_10_2026_out
set OUT_DIR=D:\GITHUB\small-world-property\results

echo Starting Batch Processing (modular pipeline, %COUNT% new random samples per project)...
echo Re-run this script anytime: compute_real_metrics.py skips the .dot file on
echo its own if real_metrics.csv is already up to date for the current code
echo version (see SCHEMA_VERSION in compute_real_metrics.py), and recomputes it
echo automatically if not. Random samples are always added on top of whatever
echo is already in random_samples\.
echo ----------------------------------------------

for %%P in (activemq archiva druid geode jackrabbit jena karaf depends phoenix solr) do (
    echo Processing %%P...

    python %REAL_SCRIPT% "%DOT_DIR%\%%P-file.dot" --out-dir "%OUT_DIR%\%%P"
    python %RAND_SCRIPT% "%OUT_DIR%\%%P\degree_sequence.json" --out-dir "%OUT_DIR%\%%P" --count %COUNT%
    python %COMBINE_SCRIPT% --project-dir "%OUT_DIR%\%%P"
)

echo ----------------------------------------------
echo All processing complete. Results are in %OUT_DIR%
pause
