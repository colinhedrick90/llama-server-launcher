from __future__ import annotations

import glob
import json
import os
import shlex
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


DEFAULT_PROFILE = {
    "mode": "single",
    "model": "",
    "models_dir": "",
    "models_preset": "",
    "models_max": 4,
    "models_autoload": True,
    "device": "Vulkan0",
    "gpu_layers": "all",
    "fit": "on",
    "context_size": 65536,
    "parallel": 1,
    "flash_attn": "auto",
    "jinja": True,
    "threads": "",
    "cache_type_k": "",
    "cache_type_v": "",
    "batch_size": "",
    "ubatch_size": "",
    "n_cpu_moe": "",
    "cpu_moe": False,
    "mlock": False,
    "no_mmap": False,
    "sleep_idle_seconds": "",
    "alias": "",
    "host": "127.0.0.1",
    "port": 8085,
    "extra_args": "",
}

# Valid KV cache quantization types accepted by --cache-type-k / --cache-type-v.
CACHE_TYPES = ("f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl")


@dataclass(frozen=True)
class ModelInfo:
    name: str
    path: str
    size_gib: float

    @property
    def label(self) -> str:
        return f"{self.name} ({self.size_gib:.2f} GiB)"


def default_settings_path() -> Path:
    base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return base / "LlamaServerLauncher" / "settings.json"


def huggingface_cache_path() -> Path:
    return Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def default_hf_cache_dir() -> str:
    localai = Path(r"C:\localai")
    if localai.exists():
        return str(localai)
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return hf_home
    return str(Path.home() / ".cache" / "huggingface" / "hub")


def find_llama_server() -> str:
    found = shutil.which("llama-server")
    return str(Path(found).resolve()) if found else ""


def discover_models(roots: Iterable[Path | str]) -> list[ModelInfo]:
    found: dict[str, ModelInfo] = {}
    for raw_root in roots:
        root = Path(raw_root).expanduser()
        if not root.is_dir():
            continue
        for path in root.rglob("*.gguf"):
            if not path.is_file() or path.name.casefold().startswith("mmproj"):
                continue
            resolved = str(path.resolve())
            found[resolved.casefold()] = ModelInfo(
                name=path.name,
                path=resolved,
                size_gib=path.stat().st_size / (1024**3),
            )
    return sorted(found.values(), key=lambda item: (item.name.casefold(), item.path.casefold()))


def parse_devices(output: str) -> list[str]:
    devices = []
    for line in output.splitlines():
        stripped = line.strip()
        if ":" in stripped and not stripped.lower().startswith("available devices"):
            devices.append(stripped.split(":", 1)[0].strip())
    return devices


def list_devices(server_path: str) -> list[str]:
    if not server_path or not Path(server_path).is_file():
        return []
    result = subprocess.run(
        [server_path, "--list-devices"],
        capture_output=True,
        text=True,
        timeout=15,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return parse_devices(result.stdout + "\n" + result.stderr)


def _strip_matching_quotes(arg: str) -> str:
    return arg[1:-1] if len(arg) >= 2 and arg[0] == arg[-1] == '"' else arg


def parse_extra_args(extra: str) -> list[str]:
    args: list[str] = []
    for line in str(extra).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        args.extend(_strip_matching_quotes(arg) for arg in shlex.split(stripped, posix=False))
    return args


DEFAULT_DOWNLOAD_ROOT = r"C:\localai"
_HF_API_BASE = "https://huggingface.co"
# Auxiliary GGUFs that belong with a model regardless of quant choice:
# multimodal projectors (mmproj-*.gguf) and MTP/draft weights (*mtp*.gguf).
_AUX_FILE_TOKENS = ("mmproj", "mtp")


@dataclass(frozen=True)
class RepoFile:
    path: str
    size_gib: float

    @property
    def label(self) -> str:
        return f"{self.path} ({self.size_gib:.2f} GiB)"


def normalize_repo_id(text: str) -> str:
    repo = str(text).strip().strip("/")
    for prefix in ("https://huggingface.co/", "http://huggingface.co/", "huggingface.co/"):
        if repo.casefold().startswith(prefix.casefold()):
            repo = repo[len(prefix):]
            break
    parts = [part for part in repo.split("/") if part]
    # Trim URL suffixes such as /tree/main or /blob/main/file.gguf.
    if len(parts) > 2 and parts[2] in ("tree", "blob", "resolve"):
        parts = parts[:2]
    return "/".join(parts[:2])


def model_download_dir(repo_id: str, root: str = DEFAULT_DOWNLOAD_ROOT) -> str:
    name = normalize_repo_id(repo_id).rsplit("/", 1)[-1]
    return str(Path(root) / name)


def _hf_token() -> str:
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        return token
    token_path = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
    try:
        return token_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def parse_repo_gguf_files(payload: list) -> list[RepoFile]:
    files = []
    for item in payload:
        if not isinstance(item, dict) or item.get("type") != "file":
            continue
        path = str(item.get("path", ""))
        if not path.casefold().endswith(".gguf"):
            continue
        files.append(RepoFile(path=path, size_gib=float(item.get("size", 0) or 0) / (1024**3)))
    return sorted(files, key=lambda item: item.path.casefold())


def list_repo_gguf_files(repo_id: str, timeout: float = 30.0) -> list[RepoFile]:
    repo = normalize_repo_id(repo_id)
    if "/" not in repo:
        raise ValueError("Enter a repo id such as owner/model-GGUF.")
    url = f"{_HF_API_BASE}/api/models/{repo}/tree/main?recursive=true"
    headers = {"User-Agent": "LlamaServerLauncher"}
    token = _hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload: list = []
    while url:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload.extend(json.loads(response.read().decode("utf-8")))
            link = response.headers.get("Link", "")
        url = ""
        for part in link.split(","):
            if 'rel="next"' in part and "<" in part:
                url = part[part.index("<") + 1 : part.index(">")]
    return parse_repo_gguf_files(payload)


def select_model_files(paths: Iterable[str], pattern: str) -> list[str]:
    pattern_cf = str(pattern).strip().casefold()
    selected = []
    for path in paths:
        name = str(path).rsplit("/", 1)[-1].casefold()
        is_aux = any(token in name for token in _AUX_FILE_TOKENS)
        if is_aux or (pattern_cf and pattern_cf in str(path).casefold()):
            selected.append(str(path))
    return selected


def build_hf_download_command(
    repo_id: str, files: Iterable[str], local_dir: str, hf_path: str = "hf"
) -> list[str]:
    return [
        hf_path,
        "download",
        normalize_repo_id(repo_id),
        *files,
        "--local-dir",
        str(local_dir),
    ]


def run_hf_download(
    repo_id: str,
    files: list[str],
    local_dir: str,
    on_line: Callable[[str], None],
    hf_path: str = "hf",
) -> int:
    if not files:
        raise ValueError("Select at least one file to download.")
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    command = build_hf_download_command(repo_id, files, local_dir, hf_path)
    run_command = command
    if os.name == "nt":
        script = "& " + " ".join(_quote_powershell_arg(arg) for arg in command)
        run_command = ["powershell", "-NoProfile", "-Command", script]
    env = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE", "__PYVENV_LAUNCHER__"):
        env.pop(name, None)
    process = subprocess.Popen(
        run_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert process.stdout is not None
    for line in process.stdout:
        stripped = line.rstrip()
        if stripped:
            on_line(stripped)
    return process.wait()


def parse_hf_cache_model_ids(output: str) -> list[str]:
    ids = []
    for line in output.splitlines():
        item = line.strip()
        if item.startswith("model/"):
            ids.append(item)
    return list(dict.fromkeys(ids))


def build_hf_cache_list_command(cache_dir: str, hf_path: str = "hf") -> list[str]:
    return [hf_path, "cache", "ls", "--cache-dir", str(cache_dir), "--format", "quiet"]


def build_hf_cache_remove_command(
    cache_dir: str, repo_id: str, *, dry_run: bool, hf_path: str = "hf"
) -> list[str]:
    return [
        hf_path,
        "cache",
        "rm",
        str(repo_id),
        "--cache-dir",
        str(cache_dir),
        "--dry-run" if dry_run else "--yes",
    ]


def _quote_powershell_arg(arg: str) -> str:
    return "'" + str(arg).replace("'", "''") + "'"


def _run_cli_command(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    run_command = command
    if os.name == "nt":
        script = "& " + " ".join(_quote_powershell_arg(arg) for arg in command)
        run_command = ["powershell", "-NoProfile", "-Command", script]
    env = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE", "__PYVENV_LAUNCHER__"):
        env.pop(name, None)
    return subprocess.run(
        run_command,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def list_hf_cache_models(cache_dir: str, hf_path: str = "hf") -> list[str]:
    result = _run_cli_command(build_hf_cache_list_command(cache_dir, hf_path), timeout=30)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "hf cache ls failed").strip())
    return parse_hf_cache_model_ids(result.stdout)


def run_hf_cache_remove(
    cache_dir: str, repo_id: str, *, dry_run: bool, hf_path: str = "hf"
) -> str:
    result = _run_cli_command(
        build_hf_cache_remove_command(cache_dir, repo_id, dry_run=dry_run, hf_path=hf_path),
        timeout=120,
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode != 0:
        raise RuntimeError(output or "hf cache rm failed")
    return output


def validate_profile(profile: dict) -> list[str]:
    errors = []
    if profile.get("mode", "single") == "router":
        models_dir = str(profile.get("models_dir", "")).strip()
        if not models_dir:
            errors.append("Select a models directory.")
        elif not Path(models_dir).is_dir():
            errors.append("The selected models directory does not exist.")
        models_preset = str(profile.get("models_preset", "")).strip()
        if models_preset and not Path(models_preset).is_file():
            errors.append("The selected models preset does not exist.")
        try:
            if int(profile.get("models_max", -1)) < 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append("Maximum loaded models must be zero or a positive integer.")
    else:
        if not str(profile.get("model", "")).strip():
            errors.append("Select a GGUF model.")
        elif not Path(str(profile["model"])).is_file():
            errors.append("The selected GGUF model does not exist.")
    try:
        port = int(profile.get("port", 0))
        if not 1 <= port <= 65535:
            raise ValueError
    except (TypeError, ValueError):
        errors.append("Port must be between 1 and 65535.")
    for key, label, minimum in (
        ("context_size", "Context size", 1),
        ("parallel", "Parallel slots", 1),
    ):
        try:
            if int(profile.get(key, 0)) < minimum:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"{label} must be a positive integer.")
    if str(profile.get("flash_attn", "auto")).strip().lower() not in {"auto", "on", "off"}:
        errors.append("Flash attention must be auto, on, or off.")
    threads = str(profile.get("threads", "")).strip()
    if threads:
        try:
            int(threads)
        except (TypeError, ValueError):
            errors.append("Threads must be blank or an integer.")
    for key, label in (("batch_size", "Batch size"), ("ubatch_size", "Micro-batch size")):
        value = str(profile.get(key, "")).strip()
        if value:
            try:
                if int(value) < 1:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(f"{label} must be blank or a positive integer.")
    n_cpu_moe = str(profile.get("n_cpu_moe", "")).strip()
    if n_cpu_moe:
        try:
            if int(n_cpu_moe) < 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append("CPU MoE layers must be blank or zero or a positive integer.")
    for key, label in (("cache_type_k", "KV cache type (K)"), ("cache_type_v", "KV cache type (V)")):
        value = str(profile.get(key, "")).strip()
        if value and value not in CACHE_TYPES:
            errors.append(f"{label} must be blank or one of: {', '.join(CACHE_TYPES)}.")
    sleep_idle_seconds = str(profile.get("sleep_idle_seconds", "")).strip()
    if sleep_idle_seconds:
        try:
            if int(sleep_idle_seconds) < -1:
                raise ValueError
        except (TypeError, ValueError):
            errors.append("Sleep idle seconds must be blank or an integer greater than or equal to -1.")
    return errors


def build_server_command(server_path: str, profile: dict) -> list[str]:
    command = [server_path]
    if profile.get("mode", "single") == "router":
        command.extend(["--models-dir", str(profile["models_dir"])])
        models_preset = str(profile.get("models_preset", "")).strip()
        if models_preset:
            command.extend(["--models-preset", models_preset])
        command.extend(["--models-max", str(profile["models_max"])])
        command.append(
            "--models-autoload" if profile.get("models_autoload", True) else "--no-models-autoload"
        )
    else:
        command.extend(
            [
                "--model",
                str(profile["model"]),
                "--device",
                str(profile["device"]),
                "--gpu-layers",
                str(profile["gpu_layers"]),
                "--fit",
                str(profile["fit"]),
                "--ctx-size",
                str(profile["context_size"]),
                "--parallel",
                str(profile["parallel"]),
            ]
        )
    command.extend(
        [
            "--flash-attn",
            str(profile.get("flash_attn", "auto")).strip().lower(),
            "--jinja" if profile.get("jinja", True) else "--no-jinja",
        ]
    )
    threads = str(profile.get("threads", "")).strip()
    if threads:
        command.extend(["--threads", threads])
    for flag, key in (("--cache-type-k", "cache_type_k"), ("--cache-type-v", "cache_type_v")):
        value = str(profile.get(key, "")).strip()
        if value:
            command.extend([flag, value])
    for flag, key in (("--batch-size", "batch_size"), ("--ubatch-size", "ubatch_size")):
        value = str(profile.get(key, "")).strip()
        if value:
            command.extend([flag, value])
    if profile.get("cpu_moe"):
        command.append("--cpu-moe")
    else:
        n_cpu_moe = str(profile.get("n_cpu_moe", "")).strip()
        if n_cpu_moe:
            command.extend(["--n-cpu-moe", n_cpu_moe])
    if profile.get("mlock"):
        command.append("--mlock")
    if profile.get("no_mmap"):
        command.append("--no-mmap")
    sleep_idle_seconds = str(profile.get("sleep_idle_seconds", "")).strip()
    if sleep_idle_seconds:
        command.extend(["--sleep-idle-seconds", sleep_idle_seconds])
    alias = str(profile.get("alias", "")).strip()
    if alias:
        command.extend(["--alias", alias])
    command.extend(
        [
            "--host",
            str(profile["host"]),
            "--port",
            str(profile["port"]),
        ]
    )
    extra = str(profile.get("extra_args", "")).strip()
    if extra:
        command.extend(parse_extra_args(extra))
    return command


def is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, int(port)))
            return True
        except OSError:
            return False


# Router model statuses (lowercase, per /v1/models) that still hold VRAM.
ROUTER_LOADED_STATUSES = {"loaded", "loading", "sleeping"}
# Unload endpoint moved under /v1 on newer builds; try the legacy path first.
_ROUTER_UNLOAD_PATHS = ("/models/unload", "/v1/models/unload")


# Wildcard BIND addresses are not valid CONNECT destinations. The profile's
# `host` is what llama-server binds to; when it is a wildcard the client half
# has to dial loopback instead, or every router query fails.
_WILDCARD_HOSTS = {"0.0.0.0", "::", "[::]", "*"}


def _client_host(host: str) -> str:
    h = (host or "").strip()
    return "127.0.0.1" if h in _WILDCARD_HOSTS or not h else h


def _router_api_key() -> str:
    """
    Router auth key. Without this header the router answers 401 to every query
    once `--api-key` is set, which silently breaks the Loaded Models panel *and*
    model unloading.

    Resolved from the selected profile's extra_args rather than the environment:
    a GUI app launched from Explorer inherits Explorer's environment, which is
    captured at logon and does NOT see a later-added user variable. LLAMA_API_KEY
    is still honoured as an explicit override when present.
    """
    env_key = (os.environ.get("LLAMA_API_KEY") or "").strip()
    if env_key:
        return env_key
    try:
        raw = json.loads(default_settings_path().read_text(encoding="utf-8"))
        profile = raw.get("profiles", {}).get(raw.get("selected_profile", ""), {})
        args = parse_extra_args(profile.get("extra_args", ""))
        for index, arg in enumerate(args):
            if arg == "--api-key" and index + 1 < len(args):
                return args[index + 1].strip()
            if arg.startswith("--api-key="):
                return arg.split("=", 1)[1].strip()
    except Exception:
        pass  # unreadable settings -> behave as if no key is configured
    return ""


def _router_request(host: str, port, path: str, *, data: bytes | None = None, timeout: float = 10.0):
    url = f"http://{_client_host(host)}:{port}{path}"
    headers = {"Content-Type": "application/json"} if data is not None else {}
    key = _router_api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    method = "POST" if data is not None else "GET"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body.strip() else {}


def parse_router_models(payload: dict) -> list[dict]:
    models = []
    for item in payload.get("data", []):
        status = item.get("status")
        value = status.get("value") if isinstance(status, dict) else status
        models.append({"id": str(item.get("id", "")), "status": str(value or "").lower()})
    return models


def list_router_models(host: str, port, timeout: float = 10.0) -> list[dict]:
    return parse_router_models(_router_request(host, port, "/v1/models", timeout=timeout))


def router_loaded_models(host: str, port, timeout: float = 10.0) -> list[str]:
    return [
        model["id"]
        for model in list_router_models(host, port, timeout=timeout)
        if model["status"] in ROUTER_LOADED_STATUSES
    ]


def unload_router_model(host: str, port, model_id: str, timeout: float = 30.0) -> None:
    data = json.dumps({"model": model_id}).encode("utf-8")
    last_error: urllib.error.HTTPError | None = None
    for path in _ROUTER_UNLOAD_PATHS:
        try:
            _router_request(host, port, path, data=data, timeout=timeout)
            return
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 404:
                continue
            raise
    if last_error is not None:
        raise last_error


def dedicated_gpu_memory_gib(pid: int) -> float | None:
    script = (
        "$c=Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' -ErrorAction SilentlyContinue;"
        f"$s=$c.CounterSamples|Where-Object{{$_.InstanceName -match 'pid_{pid}_'}}|"
        "Measure-Object CookedValue -Sum;"
        "if($s.Sum){[math]::Round($s.Sum/1GB,2)}else{0}"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return float(result.stdout.strip() or 0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class SettingsStore:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else default_settings_path()

    @staticmethod
    def defaults() -> dict:
        return {
            "llama_server": find_llama_server(),
            "custom_model_folder": "",
            "hf_cache_dir": default_hf_cache_dir(),
            "selected_profile": "Default",
            "profiles": {"Default": dict(DEFAULT_PROFILE)},
        }

    def load(self) -> dict:
        settings = self.defaults()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            settings.update({key: raw[key] for key in settings if key in raw})
            profiles = raw.get("profiles", {})
            settings["profiles"] = {
                name: {**DEFAULT_PROFILE, **profile}
                for name, profile in profiles.items()
                if isinstance(name, str) and isinstance(profile, dict)
            }
            settings["profiles"].setdefault("Default", dict(DEFAULT_PROFILE))
            if settings["selected_profile"] not in settings["profiles"]:
                settings["selected_profile"] = "Default"
        except (OSError, ValueError, TypeError):
            pass
        return settings

    def save(self, settings: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        temp.replace(self.path)


class ManagedServer:
    def __init__(self, log_callback: Callable[[str], None]):
        self.log_callback = log_callback
        self.process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.running and self.process else None

    def start(self, command: list[str], cwd: str | None = None) -> None:
        if self.running:
            raise RuntimeError("The managed server is already running.")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
        self.process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=flags,
        )
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()

    def _read_output(self) -> None:
        process = self.process
        if not process or not process.stdout:
            return
        for line in process.stdout:
            self.log_callback(line.rstrip())
        code = process.wait()
        self.log_callback(f"[launcher] llama-server exited with code {code}.")

    def stop(self, timeout: float = 5.0) -> None:
        process = self.process
        if not process or process.poll() is not None:
            return
        self.log_callback("[launcher] Stopping llama-server...")
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.log_callback("[launcher] Server did not stop in time; forcing shutdown.")
            process.kill()
            process.wait(timeout=timeout)
        deadline = time.monotonic() + 1
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if self._reader and self._reader is not threading.current_thread():
            self._reader.join(timeout=1)
        if process.stdout:
            process.stdout.close()


# --- FIM context proxy -------------------------------------------------------
#
# The proxy runs as a Windows scheduled task (registered at logon) rather than as
# a child of this app or of a VS Code terminal, so it survives window reloads.
# These helpers drive that task; they never spawn the proxy directly, which would
# leave a second copy fighting for port 8012.

FIM_PROXY_TASK = "llama-fim-context-proxy"
FIM_PROXY_HEALTH_URL = "http://127.0.0.1:8012/health"
FIM_PROXY_CONFIG = r"C:\Users\colin\llama-fim-context-proxy\config.toml"


def _run_powershell(script: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    """Run a PowerShell snippet.

    Separate from _run_cli_command because that helper single-quotes every
    argument, which turns cmdlet parameter names like -TaskName into positional
    strings.
    """
    env = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE", "__PYVENV_LAUNCHER__"):
        env.pop(name, None)
    return subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def fim_proxy_task_state() -> str:
    """Ready / Running / Disabled / Missing."""
    result = _run_powershell(
        f"$t = Get-ScheduledTask -TaskName '{FIM_PROXY_TASK}' -ErrorAction SilentlyContinue; "
        "if ($t) { $t.State } else { 'Missing' }",
        timeout=20,
    )
    state = (result.stdout or "").strip().splitlines()
    return state[-1].strip() if state else "Unknown"


def fim_proxy_health(timeout: float = 2.0) -> dict:
    """Probe /health. Returns {} when the proxy is not answering."""
    try:
        with urllib.request.urlopen(FIM_PROXY_HEALTH_URL, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return {}


def fim_proxy_status() -> dict:
    """Combined task + health view for the UI."""
    health = fim_proxy_health()
    return {
        "task_state": fim_proxy_task_state(),
        "listening": bool(health),
        "backend": health.get("backend", ""),
        "backend_mode": health.get("backend_mode", ""),
        "backend_url": health.get("backend_url", ""),
        "indexed_chunks": health.get("indexed_chunks"),
    }


def fim_proxy_start() -> str:
    if fim_proxy_task_state() == "Missing":
        raise RuntimeError(
            f"Scheduled task '{FIM_PROXY_TASK}' is not registered. "
            "Register it before starting the proxy from here."
        )
    result = _run_powershell(f"Start-ScheduledTask -TaskName '{FIM_PROXY_TASK}'", timeout=30)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "").strip() or "Start failed")
    return "FIM proxy start requested."


def fim_proxy_stop() -> str:
    """Stop the task, then reap any proxy still holding 8012.

    Stop-ScheduledTask does not always terminate an already-detached process, so
    fall back to killing whatever owns the port. The proxy puts its llama-server
    backend in a kill-on-close job object, so the backend dies with it.
    """
    _run_powershell(f"Stop-ScheduledTask -TaskName '{FIM_PROXY_TASK}' -ErrorAction SilentlyContinue")
    result = _run_powershell(
        "$c = Get-NetTCPConnection -State Listen -LocalPort 8012 -ErrorAction SilentlyContinue; "
        "if ($c) { Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue; 'killed' } "
        "else { 'not running' }",
        timeout=30,
    )
    lines = (result.stdout or "").strip().splitlines()
    detail = lines[-1].strip() if lines else "no response"
    return f"FIM proxy stopped ({detail})."


def fim_proxy_restart() -> str:
    fim_proxy_stop()
    time.sleep(1.0)
    fim_proxy_start()
    return "FIM proxy restarted."


# --- fim-ghost VS Code extension config (ghost-text behavior) --------------
# The extension folder is version-suffixed (local.fim-ghost-<ver>), so glob it.
# The extension file-watches this JSON and reloads live — no proxy restart.
FIM_EXT_CONFIG_GLOB = str(
    Path.home() / ".vscode" / "extensions" / "local.fim-ghost*" / "fim.config.json"
)


def fim_ext_config_path() -> str | None:
    matches = sorted(glob.glob(FIM_EXT_CONFIG_GLOB))
    return matches[-1] if matches else None


def fim_ext_config_load() -> dict:
    path = fim_ext_config_path()
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def fim_retrieval_enabled_get() -> bool:
    """Read [retrieval].enabled from the proxy config.toml (default True)."""
    try:
        import tomllib

        with open(FIM_PROXY_CONFIG, "rb") as handle:
            data = tomllib.load(handle)
        return bool(data.get("retrieval", {}).get("enabled", True))
    except (OSError, ValueError):
        return True


def fim_retrieval_enabled_set(value: bool) -> None:
    """Flip [retrieval].enabled in config.toml, preserving comments/formatting.

    Line-targeted (not a full TOML rewrite) so the surrounding comments survive.
    The proxy reads config.toml only at startup, so this needs a proxy restart.
    """
    token = "true" if value else "false"
    with open(FIM_PROXY_CONFIG, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    in_retrieval = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_retrieval = stripped == "[retrieval]"
            continue
        if in_retrieval and stripped.split("=", 1)[0].strip() == "enabled":
            lines[i] = f"enabled = {token}\n"
            break
    else:
        for i, line in enumerate(lines):
            if line.strip() == "[retrieval]":
                lines.insert(i + 1, f"enabled = {token}\n")
                break
        else:
            raise RuntimeError("[retrieval] section not found in config.toml")
    with open(FIM_PROXY_CONFIG, "w", encoding="utf-8") as handle:
        handle.writelines(lines)


def fim_ext_config_save(values: dict) -> str:
    """Merge values into fim.config.json; the extension reloads it live."""
    path = fim_ext_config_path()
    if not path:
        raise RuntimeError(
            "fim-ghost extension config not found under ~/.vscode/extensions"
        )
    current = fim_ext_config_load()
    current.update(values)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(current, handle, indent=2)
        handle.write("\n")
    return path
