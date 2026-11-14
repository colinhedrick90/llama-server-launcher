import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from launcher_core import (
    DEFAULT_PROFILE,
    ManagedServer,
    build_server_command,
    default_model_root,
    find_llama_server,
    is_port_available,
    validate_profile,
)


def main() -> None:
    profile = {
        **DEFAULT_PROFILE,
        "mode": "router",
        "models_dir": str(default_model_root()),
        "models_max": 1,
        "models_autoload": False,
        "port": 18086,
    }
    errors = validate_profile(profile)
    if errors:
        raise RuntimeError("\n".join(errors))
    if not is_port_available(profile["host"], profile["port"]):
        raise RuntimeError("Integration-test port 18086 is already occupied.")

    command = build_server_command(find_llama_server(), profile)
    if "--model" in command:
        raise RuntimeError("Router command unexpectedly contains --model.")

    recent_logs = []
    managed = ManagedServer(lambda line: recent_logs.append(line))
    try:
        managed.start(command)
        url = f"http://{profile['host']}:{profile['port']}/v1/models"
        deadline = time.monotonic() + 30
        response_body = ""
        while time.monotonic() < deadline and managed.running:
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    response_body = response.read().decode("utf-8")
                break
            except OSError:
                time.sleep(0.5)
        if not managed.running:
            raise RuntimeError("Router exited during startup:\n" + "\n".join(recent_logs[-20:]))
        if not response_body:
            raise RuntimeError("Router API did not respond before timeout.")
        print(f"router_api={response_body}")
    finally:
        managed.stop(timeout=10)
    print(f"stopped running={managed.running}")


if __name__ == "__main__":
    main()
