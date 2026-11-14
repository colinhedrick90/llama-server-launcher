# Llama Server Launcher Design

## Purpose

Provide a lightweight native interface for configuring and managing one
`llama-server` process on Windows and Linux from a single codebase. The packaged
application must be suitable for pinning to the taskbar or app grid and must
stop its managed server when the application closes.

## Interface

- Select and save multiple named profiles.
- Automatically detect `llama-server` and available devices.
- Scan the Hugging Face cache and one optional custom folder recursively for
  `.gguf` model files.
- Configure model, device, GPU layers, context size, parallel slots, host, port,
  fit behavior, flash attention, Jinja, idle sleep timeout, alias, and
  additional command-line arguments.
- Configure shared performance and memory flags: CPU threads, KV cache K/V
  type, batch and micro-batch size, MoE CPU offload (all or first N layers),
  mlock, and mmap behavior.
- Preserve those single-model profiles while also supporting router profiles
  that configure models directory, optional preset, maximum loaded models, and
  autoload behavior.
- Inspect Hugging Face cache model repositories from a configured cache
  directory and remove a selected repository through the `hf` CLI.
- List models currently holding VRAM in router mode and unload one or all of
  them through the server's `/models/unload` HTTP endpoint for a clean slate.
- Start, stop, and restart the managed server.
- Display process status and live server logs on a dedicated Log tab with
  clear, copy, and auto-scroll controls.
- Control the FIM context proxy and edit the `fim-ghost` ghost-text, sampling,
  and retrieval settings from a FIM Proxy tab.

## Behavior

Settings are stored per user: `%LOCALAPPDATA%\LlamaServerLauncher\settings.json`
on Windows and `$XDG_CONFIG_HOME/LlamaServerLauncher/settings.json` on Linux.
No machine-specific paths or credentials live in the source tree; platform
differences are isolated behind `IS_WINDOWS` in `launcher_core.py`.
Profiles contain all launch settings. The application prevents launch when the
configured port is already occupied. It only stops a server process that it
started. Closing the application gracefully terminates that process, waits, and
then force-terminates it if necessary; exiting `llama-server` releases its model
and KV-cache VRAM allocations.

Single-model profiles launch with `--model` and the existing model tuning
arguments. Router profiles launch without `--model` and omit the single-model
device, GPU layer, fit, context size, and parallel-slot arguments; per-model
settings can instead be defined by the optional router preset or additional
arguments.
Router profiles always pass `--models-dir`, `--models-max`, and an explicit
autoload choice. Flash attention, Jinja, CPU threads, KV cache type, batch and
micro-batch size, MoE CPU offload, mlock, mmap behavior, idle sleep timeout,
alias, host, port, and additional arguments are shared between both modes. The
shared performance and memory flags are omitted from the launch command when
left at their blank or unchecked defaults. The UI disables controls that do not
apply to the selected mode.

GGUF downloading remains an external `hf download` workflow. Cache cleanup is a
smaller local operation: the launcher lists model repository IDs with
`hf cache ls --format quiet --cache-dir <path>`, previews deletion with
`hf cache rm <repo> --dry-run`, and deletes only after confirmation using
`hf cache rm <repo> --yes`. This deliberately does not delete arbitrary local
folders because plain GGUF directories are not Hugging Face cache repositories.

## Architecture

- `launcher_core.py`: settings, discovery, validation, command construction,
  port checks, process control, and GPU memory lookup.
- `launcher_app.py`: Tkinter widgets, event handling, profile editing, log
  display, status updates, and shutdown behavior.
- `main.py`: packaged application entry point.
- `tests/`: standard-library unit tests for core behavior.
