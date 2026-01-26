# Define the paths to the tool and the output directory
$toolPath = ".\depends-0.9.7-package-20221104a\depends-0.9.7\depends.bat"
$repoRoot = ".\repos"
$outputRoot = ".\depends-out-json"

# Create the output root directory if it doesn't exist
if (!(Test-Path -Path $outputRoot)) {
    New-Item -ItemType Directory -Path $outputRoot | Out-Null
}

# Iterate through each subdirectory in the repos folder
Get-ChildItem -Path $repoRoot -Directory | ForEach-Object {
    $folderName = $_.Name
    $inputPath = $_.FullName
    $outputPath = Join-Path -ChildPath $folderName -Path $outputRoot

    Write-Host "Processing folder: $folderName" -ForegroundColor Cyan

    # Execute the command with the specified arguments
    # --type-filter is passed as a single string to ensure correct parsing
    & $toolPath java $inputPath $outputPath -g=file -p=windows -s --type-filter=Import,Extend,Implement,Call,Create
}

Write-Host "Batch processing complete." -ForegroundColor Green