$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Project

$BuildPython = Join-Path $Project ".venv-build\Scripts\python.exe"
if (-not (Test-Path $BuildPython)) {
    python -m venv .venv-build
    if ($LASTEXITCODE -ne 0) { throw "Could not create build environment." }
}

& $BuildPython -m pip install -r requirements-build.txt
if ($LASTEXITCODE -ne 0) { throw "Could not install build requirements." }

& $BuildPython tools\make_icon.py
if ($LASTEXITCODE -ne 0) { throw "Could not generate application icon." }

& $BuildPython -m PyInstaller --noconfirm --clean --onefile --windowed `
    --name LlamaServerLauncher `
    --icon (Join-Path $Project "assets\llama-server-launcher.ico") `
    --distpath . `
    --workpath build `
    --specpath build `
    main.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

Write-Host "Built: $Project\LlamaServerLauncher.exe"
