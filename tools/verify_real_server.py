import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from launcher_core import (
    DEFAULT_PROFILE,
    ManagedServer,
    build_server_command,
    dedicated_gpu_memory_gib,
    discover_models,
    find_llama_server,
    huggingface_cache_path,
    is_port_available,
)


def main() -> None:
    server_path = find_llama_server()
    models = discover_models([huggingface_cache_path()])
    model = next(item for item in models if "supergemma" in item.name.casefold())
    profile = {
        **DEFAULT_PROFILE,
        "model": model.path,
        "context_size": 4096,
        "parallel": 1,
        "port": 18085,
    }
    if not is_port_available(profile["host"], profile["port"]):
        raise RuntimeError("Integration-test port 18085 is already occupied.")

    recent_logs = []
    managed = ManagedServer(lambda line: recent_logs.append(line))
    try:
        managed.start(build_server_command(server_path, profile))
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline and managed.running:
            if not is_port_available(profile["host"], profile["port"]):
                break
            time.sleep(1)
        if not managed.running:
            raise RuntimeError("Server exited during startup:\n" + "\n".join(recent_logs[-20:]))
        if is_port_available(profile["host"], profile["port"]):
            raise RuntimeError("Server did not begin listening before timeout.")
        print(f"listening pid={managed.pid} vram_gib={dedicated_gpu_memory_gib(managed.pid)}")
    finally:
        managed.stop(timeout=10)
    print(f"stopped running={managed.running}")


if __name__ == "__main__":
    main()
