# Llama Server Launcher

A lightweight Tkinter launcher for configuring and managing
[`llama-server`](https://github.com/ggml-org/llama.cpp). One codebase, runs on
Windows and Linux.

Three tabs:

- **Server** — profiles, launch options, router controls, Hugging Face
  download and cache cleanup, loaded-model (VRAM) management. Scrolls when the
  window is shorter than the content.
- **FIM Proxy** — start/stop the FIM context proxy and edit the `fim-ghost`
  ghost-text and sampling settings.
- **Log** — live server output with **Clear**, **Copy All**, and an
  **Auto-scroll** toggle so reading back is not interrupted by new lines.

Closing the launcher stops only the server process that the launcher started.

## Running it

Requires Python 3.11+ with Tkinter and a `llama-server` binary on `PATH` (or
browse to one in the Server tab). There are no third-party runtime
dependencies.

| | Windows | Linux |
|---|---|---|
| From source | `python main.py` | `python3 main.py` |
| Prebuilt | `LlamaServerLauncher.exe` | `./LlamaServerLauncher` |
| Tk package | bundled with python.org installer | `sudo apt install python3-tk` |

Prebuilt binaries for both platforms are attached to the
[latest release](../../releases), rebuilt automatically from `main`.

## Platform defaults

Nothing machine-specific is stored in this repository. Paths are derived per
user at runtime, and everything you configure is saved locally:

| | Windows | Linux |
|---|---|---|
| Settings | `%LOCALAPPDATA%\LlamaServerLauncher\settings.json` | `$XDG_CONFIG_HOME/LlamaServerLauncher/settings.json` (default `~/.config`) |
| Model root | `C:\localai` | `~/models` |
| VRAM readout | PowerShell GPU performance counter | DRM fdinfo under `/proc/<pid>/fdinfo` |
| FIM proxy control | Scheduled Task `llama-fim-context-proxy` | not wired up yet (reports *Unavailable*) |
| FIM proxy config | `~/llama-fim-context-proxy/config.toml` | same |

Secrets are never written to profiles by the launcher itself:

- Hugging Face access uses `HF_TOKEN` or the `hf` CLI's stored login.
- A router API key is read from `LLAMA_API_KEY`, or from an `--api-key` you put
  in a profile's additional arguments — which then lives only in your local
  `settings.json`.

## Launch modes

- **single** selects one GGUF and configures its device, GPU layers, fit,
  context size, and parallel slots.
- **router** starts `llama-server` without `--model`, serves GGUF files from a
  models directory, and configures an optional preset, maximum simultaneously
  loaded models, and automatic model loading.

Single-model controls are disabled while editing a router profile and vice
versa. Flash attention, Jinja, idle sleep, alias, host, port, and additional
arguments apply to both. Context size, parallel slots, device, GPU layers, and
fit remain single-model controls; use a router preset for per-model values.

## Common llama-server flags

Profiles expose first-class controls for common server flags:

- **Flash attention** — `--flash-attn auto|on|off`.
- **Jinja** — `--jinja` / `--no-jinja`.
- **Sleep idle** — when set, `--sleep-idle-seconds N`; blank omits the flag.
- **Alias** — when set, `--alias NAME`.

These apply to both modes and are omitted when left blank/unchecked:

- **Threads** — `--threads N`.
- **KV cache K/V** — `--cache-type-k TYPE` / `--cache-type-v TYPE` (`q8_0`,
  `q4_0`, …) to quantize the KV cache and cut VRAM. Flash attention must be on.
- **Batch / U-batch** — `--batch-size N` / `--ubatch-size N`.
- **CPU MoE (all)** — `--cpu-moe`, keeping all Mixture-of-Experts weights on the
  CPU. Takes precedence over CPU MoE layers when both are set.
- **CPU MoE layers** — `--n-cpu-moe N`, keeping MoE weights of the first N
  layers on the CPU to fit large MoE models in limited VRAM.
- **mlock** — `--mlock`, keeping the model resident in RAM.
- **No mmap** — `--no-mmap`, disabling memory-mapping (slower load).

Additional arguments cover new or uncommon flags. Enter them one line at a
time; non-empty lines are parsed in order and appended after the profile
controls.

In router mode keep per-model values such as `ctx-size`, `parallel`,
`n-gpu-layers`, `model`, and `mmproj` in the `--models-preset` INI rather than
setting one global launcher value for every routed model. The router also
auto-discovers top-level GGUFs from `--models-dir`, so `/v1/models` can list
more entries than the preset defines.

## FIM Proxy tab

**Start / Stop / Restart Proxy** drive the `llama-fim-context-proxy` Scheduled
Task on Windows. On Linux those buttons report *Unavailable* until a
`systemd --user` unit is wired up.

The settings below the controls work on both platforms:

- **Ghost-text settings** — enable, debounce, prefix/suffix caps, max predicted
  tokens, stop tokens, status-bar indicator. Saved to the `fim-ghost` VS Code
  extension's `fim.config.json` (found under `~/.vscode/extensions/local.fim-ghost*`),
  which the extension hot-reloads.
- **Sampling** — temperature, top-p, top-k, and the frequency / presence /
  repeat penalties passed through to `/infill`. Hover a name for what it does.
- **Retrieval enabled** — flips `[retrieval].enabled` in the proxy's
  `config.toml` in place, preserving comments. Takes effect after
  **Restart Proxy**.

## Download GGUF files locally

The **Download Model (Hugging Face)** panel downloads GGUF files without
leaving the launcher:

1. Paste a repo id or full URL (`unsloth/GLM-4.6-GGUF` or
   `https://huggingface.co/unsloth/GLM-4.6-GGUF`) and choose **List Files**.
2. Enter a quant filter such as `Q4_K_XL` and choose **Select Matching**. That
   selects every GGUF whose path contains the filter (including all multipart
   shards) plus every `mmproj` and `mtp` auxiliary GGUF in the repo. Manual
   multi-select also works.
3. Choose **Download Selected**. Files land in
   `<custom model folder>/<repo-name>` (default `<model root>/<repo-name>`),
   preserving the repo's subfolder layout. The download runs `hf download` in
   the background and streams output to the Log tab; the model list rescans
   when it finishes.

Gated or private repos use the `hf` CLI's stored login (`hf auth login`).

The CLI works directly too, which avoids launch-time downloads. Substitute your
model root (`C:\localai` or `~/models`) for `MODELS`:

```bash
# Preview matching files before downloading.
hf download bartowski/Example-GGUF --include "*Q4_K_M*.gguf" --dry-run

# Download matching GGUF files into the router directory.
hf download bartowski/Example-GGUF --include "*Q4_K_M*.gguf" --local-dir MODELS

# Download one exact file into its own folder.
hf download bartowski/Example-GGUF Example-Q4_K_M.gguf --local-dir MODELS/Example-GGUF

# Pin a branch, tag, or commit when reproducibility matters.
hf download bartowski/Example-GGUF Example-Q4_K_M.gguf --revision main --local-dir MODELS
```

Multipart GGUF models need every shard, so prefer an include pattern that
selects all parts. Router mode treats top-level GGUF files as routable models,
so keep auxiliaries such as `mmproj` GGUFs out of the router root or define
models explicitly in the preset.

Prefer `--local-dir` over the HF cache when models will be copied between
machines: the cache stores files as symlinks into `blobs/`, which do not
survive a plain copy.

## Remove cached Hugging Face models

The **Hugging Face Cache Removal** panel lists cached repositories with:

```bash
hf cache ls --cache-dir MODELS --format quiet
```

Use **Preview Remove** before deletion:

```bash
hf cache rm model/owner/repo --cache-dir MODELS --dry-run
```

**Remove** asks for confirmation, then runs:

```bash
hf cache rm model/owner/repo --cache-dir MODELS --yes
```

Only Hugging Face cache repositories are tracked. Plain folders holding
hand-placed GGUFs are invisible to `hf cache rm`.

## Unload models from VRAM

In router mode the server loads models on demand and evicts them with its own
LRU logic, which can leave stale models in VRAM or trigger out-of-memory errors
when switching workloads. The **Loaded Models (VRAM)** panel gives a manual
clean slate:

- **Refresh** queries `GET /v1/models` at the configured host and port and lists
  every model whose status holds VRAM (`loaded`, `loading`, or `sleeping`).
- **Unload Selected** unloads the chosen model.
- **Unload All** unloads every model currently holding VRAM.

Unload posts `{"model": ID}` to `/models/unload`, falling back to
`/v1/models/unload` on newer builds. The server must be running and serving the
router API at the configured host and port.

## Linux notes

- **Device indices are not stable.** On a machine with an iGPU and a discrete
  card, `Vulkan0` may be the iGPU driving the desktop. Confirm after any driver
  or hardware change and pin the right one in the profile:

  ```bash
  llama-server --list-devices
  ```

- **Coexisting with a systemd unit.** If you also run `llama-server` as a
  `systemd --user` service, give launcher profiles different ports — and
  remember both still claim the same GPU, so running them together can exhaust
  VRAM:

  ```bash
  systemctl --user status llama-server
  systemctl --user stop llama-server
  ```

- **Desktop entry.** To launch from the app grid, create
  `~/.local/share/applications/llama-server-launcher.desktop` with an `Exec=`
  line pointing at either `python3 /path/to/main.py` or the built binary.

## Development

All edits happen in this repository; both platforms run the same source.

```bash
python -m unittest discover -s tests -v
```

Layout:

- `launcher_core.py` — settings, discovery, validation, command construction,
  process control, platform-specific helpers. No Tk imports; unit tested.
- `launcher_app.py` — Tkinter UI.
- `main.py` — entry point.
- `tests/` — standard-library unit tests for the core.
- `tools/` — icon generator and real-server verification scripts.

### Building

| Windows | Linux |
|---|---|
| `.\build.ps1` → `LlamaServerLauncher.exe` | `./build.sh` → `dist/LlamaServerLauncher` |

Both create a local `.venv-build`, install PyInstaller, and produce a one-file
windowed executable. Linux additionally needs `python3-venv` and `python3-tk`.

Every push to `main` runs the tests and builds both binaries in GitHub Actions
(`.github/workflows/build.yml`). The results are published as the rolling
**latest** pre-release; pushing a `v*` tag publishes a versioned release.

### Verifying against a real server

```bash
python tools/verify_router_server.py   # router mode
python tools/verify_real_server.py     # single-model mode
```

## License

MIT — see [LICENSE](LICENSE).
