# Llama Server Launcher

A lightweight native Windows launcher for configuring and managing
`llama-server`.

Run `LlamaServerLauncher.exe`, configure a profile, and choose **Start Server**.
Closing the launcher stops only the server process that the launcher started.

Profiles support two launch modes:

- **single** selects one GGUF and configures its device, GPU layers, fit,
  context size, and parallel slots.
- **router** starts `llama-server` without `--model`, serves GGUF files from a
  models directory, and configures an optional preset, maximum simultaneously
  loaded models, and automatic model loading.

The launcher disables single-model controls while editing a router profile and
disables router controls while editing a single-model profile. Flash attention,
Jinja, idle sleep, alias, host, port, and additional arguments apply to both
modes. Context size, parallel slots, device, GPU layers, and fit remain
single-model controls; use a router models preset for per-model router values.

## Common llama-server flags

Profiles expose first-class controls for common server flags:

- **Flash attention**: emits `--flash-attn auto`, `--flash-attn on`, or
  `--flash-attn off`.
- **Jinja**: emits `--jinja` or `--no-jinja`.
- **Sleep idle**: when set, emits `--sleep-idle-seconds N`; leave blank to omit
  the flag.
- **Alias**: when set, emits `--alias NAME`.

These performance and memory controls apply to both single and router modes
and are omitted from the command when left at their blank/unchecked defaults:

- **Threads**: when set, emits `--threads N`.
- **KV cache K/V**: when set, emit `--cache-type-k TYPE` / `--cache-type-v TYPE`
  (`q8_0`, `q4_0`, etc.) to quantize the KV cache and cut VRAM. Flash attention
  must be on for quantized KV cache.
- **Batch / U-batch**: when set, emit `--batch-size N` / `--ubatch-size N`.
- **CPU MoE (all)**: emits `--cpu-moe`, keeping all Mixture-of-Experts weights on
  the CPU. Takes precedence over CPU MoE layers when both are set.
- **CPU MoE layers**: when set, emits `--n-cpu-moe N`, keeping the MoE weights of
  the first N layers on the CPU to fit large MoE models in limited VRAM.
- **mlock**: emits `--mlock` to keep the model resident in RAM.
- **No mmap**: emits `--no-mmap` to disable memory-mapping (slower load).

Additional arguments are still available for new or uncommon flags. Enter them
one line at a time; non-empty lines are parsed in order and appended after the
profile controls.

In router mode, keep per-model values such as `ctx-size`, `parallel`,
`n-gpu-layers`, `model`, and `mmproj` in a `--models-preset` INI file instead
of setting one global launcher value for every routed model.

## Download GGUF files locally

The **Download Model (Hugging Face)** panel downloads GGUF files without
leaving the launcher:

1. Paste a repo id or full URL (`unsloth/GLM-4.6-GGUF` or
   `https://huggingface.co/unsloth/GLM-4.6-GGUF`) and choose **List Files**.
2. Enter a quant filter such as `Q4_K_XL` and choose **Select Matching**. This
   selects every GGUF whose path contains the filter (including all multipart
   shards) plus every `mmproj` and `mtp` auxiliary GGUF in the repo. Manual
   multi-select in the list also works.
3. Choose **Download Selected**. Files land in
   `<custom model folder>\<repo-name>` (default `C:\localai\<repo-name>`),
   preserving the repo's subfolder layout. The download runs `hf download` in
   the background and streams its output to the server log; the model list
   rescans when it finishes.

Gated or private repos use the `hf` CLI's stored login (`hf auth login`).

Alternatively, use the Hugging Face `hf` CLI directly to download models
before launching the server. This avoids launch-time downloads and puts files
directly in the router directory:

```powershell
# Preview matching files before downloading.
hf download bartowski/Example-GGUF --include "*Q4_K_M*.gguf" --dry-run

# Download matching GGUF files into the usual local router directory.
hf download bartowski/Example-GGUF --include "*Q4_K_M*.gguf" --local-dir C:\localai

# Download one exact file.
hf download bartowski/Example-GGUF Example-Q4_K_M.gguf --local-dir C:\localai

# Pin a branch, tag, or commit when reproducibility matters.
hf download bartowski/Example-GGUF Example-Q4_K_M.gguf --revision main --local-dir C:\localai
```

Use `hf auth login` for gated or private repositories. Multipart GGUF models
need every matching shard, so prefer an include pattern that selects all parts.
Router mode treats top-level GGUF files as routable models, so keep auxiliary
files such as `mmproj` GGUFs outside the router directory or use a models preset
that explicitly defines the desired models.

## Remove cached Hugging Face models

The **Hugging Face Cache Removal** panel lists cached model repositories with:

```powershell
hf cache ls --cache-dir C:\localai --format quiet
```

Use **Preview Remove** before deletion. It runs:

```powershell
hf cache rm model/owner/repo --cache-dir C:\localai --dry-run
```

Use **Remove** to delete the selected cached model repo. It asks for
confirmation, then runs:

```powershell
hf cache rm model/owner/repo --cache-dir C:\localai --yes
```

This is intended for Hugging Face cache repositories such as
`model/ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF`. Plain folders that directly
contain hand-placed GGUF files are not tracked by `hf cache rm`.

## Unload models from VRAM

In router mode the server loads models on demand and evicts them with its own
LRU logic, which can leave stale models in VRAM or trigger out-of-memory errors
when switching workloads. The **Loaded Models (VRAM)** panel gives a manual
clean slate:

- **Refresh** queries `GET /v1/models` at the configured host and port and lists
  every model whose status is holding VRAM (`loaded`, `loading`, or `sleeping`).
- **Unload Selected** unloads the chosen model.
- **Unload All** unloads every model currently holding VRAM.

Unload posts `{"model": ID}` to `/models/unload`, falling back to
`/v1/models/unload` on newer builds. The server must be running and serving the
router API at the configured host and port.

Settings are stored in:

`%LOCALAPPDATA%\LlamaServerLauncher\settings.json`

To rebuild the executable:

```powershell
.\build.ps1
```

To verify router mode against the installed `llama-server`:

```powershell
python tools\verify_router_server.py
```
