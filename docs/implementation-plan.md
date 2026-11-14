# Llama Server Launcher Implementation Plan

**Goal:** Build and package a lightweight native Windows launcher that manages a
configurable `llama-server` process.

**Architecture:** Keep process/configuration behavior in a testable core module
and use Tkinter only for presentation and event handling. Package the entry point
with PyInstaller as a windowed one-file executable.

**Tech Stack:** Python 3, Tkinter/Tcl, unittest, PyInstaller, Windows PowerShell.

## Tasks

1. Write failing tests for defaults, profile persistence, model discovery,
   command construction, and validation.
2. Implement the core module until all core tests pass.
3. Implement the Tkinter interface using the tested core APIs.
4. Add packaging metadata, icon generation, and build script.
5. Build the standalone executable and verify tests, launch, and managed process
   lifecycle.

