@echo off
setlocal

:: Settings
set SCRIPT=pipeline.py
set ITERATIONS=20
set DOT_DIR=D:\GITHUB\small-world-property\depends-out-dot
set OUT_DIR=D:\GITHUB\small-world-property\results

echo Starting Batch Processing...
echo ----------------------------------------------

:: --- ACTIVEMQ ---
echo Processing activemq...
if not exist "%OUT_DIR%\activemq" mkdir "%OUT_DIR%\activemq"
python %SCRIPT% "%DOT_DIR%\activemq-file.dot" "%OUT_DIR%\activemq\results.csv" --random-iterations %ITERATIONS% --per-node "%OUT_DIR%\activemq\results-per-node.csv"

:: --- ARCHIVA ---
echo Processing archiva...
if not exist "%OUT_DIR%\archiva" mkdir "%OUT_DIR%\archiva"
python %SCRIPT% "%DOT_DIR%\archiva-file.dot" "%OUT_DIR%\archiva\results.csv" --random-iterations %ITERATIONS% --per-node "%OUT_DIR%\archiva\results-per-node.csv"

:: --- DRUID ---
echo Processing druid...
if not exist "%OUT_DIR%\druid" mkdir "%OUT_DIR%\druid"
python %SCRIPT% "%DOT_DIR%\druid-file.dot" "%OUT_DIR%\druid\results.csv" --random-iterations %ITERATIONS% --per-node "%OUT_DIR%\druid\results-per-node.csv"

:: --- GEODE ---
echo Processing geode...
if not exist "%OUT_DIR%\geode" mkdir "%OUT_DIR%\geode"
python %SCRIPT% "%DOT_DIR%\geode-file.dot" "%OUT_DIR%\geode\results.csv" --random-iterations %ITERATIONS% --per-node "%OUT_DIR%\geode\results-per-node.csv"

:: --- JACKRABBIT ---
echo Processing jackrabbit...
if not exist "%OUT_DIR%\jackrabbit" mkdir "%OUT_DIR%\jackrabbit"
python %SCRIPT% "%DOT_DIR%\jackrabbit-file.dot" "%OUT_DIR%\jackrabbit\results.csv" --random-iterations %ITERATIONS% --per-node "%OUT_DIR%\jackrabbit\results-per-node.csv"

:: --- JENA ---
echo Processing jena...
if not exist "%OUT_DIR%\jena" mkdir "%OUT_DIR%\jena"
python %SCRIPT% "%DOT_DIR%\jena-file.dot" "%OUT_DIR%\jena\results.csv" --random-iterations %ITERATIONS% --per-node "%OUT_DIR%\jena\results-per-node.csv"

:: --- KARAF ---
echo Processing karaf...
if not exist "%OUT_DIR%\karaf" mkdir "%OUT_DIR%\karaf"
python %SCRIPT% "%DOT_DIR%\karaf-file.dot" "%OUT_DIR%\karaf\results.csv" --random-iterations %ITERATIONS% --per-node "%OUT_DIR%\karaf\results-per-node.csv"

:: --- LUCENE-SOLR ---
echo Processing lucene-solr...
if not exist "%OUT_DIR%\lucene-solr" mkdir "%OUT_DIR%\lucene-solr"
python %SCRIPT% "%DOT_DIR%\lucene-solr-file.dot" "%OUT_DIR%\lucene-solr\results.csv" --random-iterations %ITERATIONS% --per-node "%OUT_DIR%\lucene-solr\results-per-node.csv"

:: --- PHOENIX ---
echo Processing phoenix...
if not exist "%OUT_DIR%\phoenix" mkdir "%OUT_DIR%\phoenix"
python %SCRIPT% "%DOT_DIR%\phoenix-file.dot" "%OUT_DIR%\phoenix\results.csv" --random-iterations %ITERATIONS% --per-node "%OUT_DIR%\phoenix\results-per-node.csv"

:: --- SOLR ---
echo Processing solr...
if not exist "%OUT_DIR%\solr" mkdir "%OUT_DIR%\solr"
python %SCRIPT% "%DOT_DIR%\solr-file.dot" "%OUT_DIR%\solr\results.csv" --random-iterations %ITERATIONS% --per-node "%OUT_DIR%\solr\results-per-node.csv"

echo ----------------------------------------------
echo All processing complete. Results are in %OUT_DIR%
pause