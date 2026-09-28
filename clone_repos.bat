@echo off
setlocal enabledelayedexpansion

:: Clones the full extended Sas et al. project dataset into repos\ from
:: scratch, so the whole extraction pipeline is reproducible from this one
:: script alone (no dependency on repos already present from the original
:: 10-project sample, or on a repos-bk\ backup). Already-cloned projects are
:: skipped, so this is safe to re-run.
:: Shallow clones (--depth 1) are used since Depends only needs the current
:: working tree, not full history. core.longpaths=true works around the
:: Windows 260-char path limit (spring-boot and libgdx both have generated
:: file names long enough to fail checkout without it). Each clone is
:: retried once on failure to absorb transient network glitches.
::
:: Pre-selected to stay under an 8000 Java-file limit per project (checked via
:: the GitHub API before cloning anything): elasticsearch was excluded
:: entirely (~24000 java files, far over even after trimming); spring-boot
:: (8688) is kept but has its test-only modules removed right after cloning,
:: bringing it to ~7752. That leaves 30 of the 31 dataset projects here.

set REPOS_DIR=%~dp0repos
if not exist "%REPOS_DIR%" mkdir "%REPOS_DIR%"

for %%A in (
    "accumulo=https://github.com/apache/accumulo.git"
    "activemq=https://github.com/apache/activemq.git"
    "calcite=https://github.com/apache/calcite.git"
    "cassandra=https://github.com/apache/cassandra.git"
    "chukwa=https://github.com/apache/chukwa.git"
    "druid=https://github.com/alibaba/druid.git"
    "jackson-databind=https://github.com/FasterXML/jackson-databind.git"
    "httpcomponents-client=https://github.com/apache/httpcomponents-client.git"
    "jackrabbit=https://github.com/apache/jackrabbit.git"
    "jena=https://github.com/apache/jena.git"
    "jspwiki=https://github.com/apache/jspwiki.git"
    "lucene=https://github.com/apache/lucene.git"
    "retrofit=https://github.com/square/retrofit.git"
    "spring-boot=https://github.com/spring-projects/spring-boot.git"
    "struts=https://github.com/apache/struts.git"
    "tika=https://github.com/apache/tika.git"
    "ant-ivy=https://github.com/apache/ant-ivy.git"
    "jenkins=https://github.com/jenkinsci/jenkins.git"
    "jgit=https://github.com/eclipse-jgit/jgit.git"
    "selenium=https://github.com/SeleniumHQ/selenium.git"
    "testng=https://github.com/testng-team/testng.git"
    "pdfbox=https://github.com/apache/pdfbox.git"
    "poi=https://github.com/apache/poi.git"
    "xerces2-j=https://github.com/apache/xerces2-j.git"
    "pgjdbc=https://github.com/pgjdbc/pgjdbc.git"
    "mina=https://github.com/apache/mina.git"
    "libgdx=https://github.com/libgdx/libgdx.git"
    "fastjson=https://github.com/alibaba/fastjson.git"
    "gson=https://github.com/google/gson.git"
    "guava=https://github.com/google/guava.git"
) do (
    for /f "tokens=1,2 delims==" %%N in (%%A) do (
        if exist "%REPOS_DIR%\%%N" (
            echo Skipping %%N, already exists in %REPOS_DIR%
        ) else (
            echo Cloning %%N from %%O
            git -c core.longpaths=true clone --depth 1 "%%O" "%REPOS_DIR%\%%N"
            if errorlevel 1 (
                echo Retrying %%N after a failed clone
                rmdir /s /q "%REPOS_DIR%\%%N" 2>nul
                git -c core.longpaths=true clone --depth 1 "%%O" "%REPOS_DIR%\%%N"
            )
            if "%%N"=="spring-boot" (
                for %%D in (smoke-test integration-test system-test test-support) do (
                    if exist "%REPOS_DIR%\spring-boot\%%D" (
                        echo Removing spring-boot\%%D to stay under the java-file limit
                        rmdir /s /q "%REPOS_DIR%\spring-boot\%%D"
                    )
                )
            )
        )
    )
)

echo ----------------------------------------------
echo Cloning complete. Repos are in %REPOS_DIR%
