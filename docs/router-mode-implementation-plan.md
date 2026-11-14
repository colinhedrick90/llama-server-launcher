# Router Mode Implementation Plan

**Goal:** Preserve single-model profiles and add native launcher support for
llama.cpp router mode.

**Architecture:** Extend the existing profile dictionary with a mode and router
fields. Keep validation and command construction in `launcher_core.py`; make
`launcher_app.py` switch enabled controls based on the selected profile mode.

**Tech Stack:** Python 3, Tkinter/Tcl, unittest, PyInstaller, Windows PowerShell.

## Tasks

1. Add failing core tests for router command construction, router validation,
   and loading old single-model settings.
2. Extend profile defaults, validation, and command construction while keeping
   the original single-model command test unchanged.
3. Add profile mode and router controls to the native UI, disabling controls
   that do not apply to the active mode.
4. Document router profiles and local GGUF downloads with the current `hf` CLI.
5. Run unit tests, compile checks, a real router-server lifecycle check, rebuild
   the executable, and smoke-test the packaged app.
