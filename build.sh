#!/usr/bin/env bash
# Linux counterpart of build.ps1: one-file windowed executable in dist/.
# Needs python3-venv and python3-tk (sudo apt install python3-venv python3-tk).
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
BUILD_PYTHON=".venv-build/bin/python"

if [ ! -x "$BUILD_PYTHON" ]; then
    "$PYTHON" -m venv .venv-build
fi

"$BUILD_PYTHON" -m pip install -r requirements-build.txt

"$BUILD_PYTHON" -m PyInstaller --noconfirm --clean --onefile --windowed \
    --name LlamaServerLauncher \
    --distpath dist \
    --workpath build \
    --specpath build \
    main.py

echo "Built: $PWD/dist/LlamaServerLauncher"
