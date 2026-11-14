import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from launcher_core import (
    DEFAULT_PROFILE,
    ManagedServer,
    SettingsStore,
    build_hf_cache_list_command,
    build_hf_cache_remove_command,
    build_hf_download_command,
    build_server_command,
    discover_models,
    model_download_dir,
    normalize_repo_id,
    parse_extra_args,
    parse_hf_cache_model_ids,
    parse_devices,
    parse_repo_gguf_files,
    parse_router_models,
    select_model_files,
    validate_profile,
)


class HfDownloadTests(unittest.TestCase):
    def test_normalizes_repo_ids_and_urls(self):
        for raw in (
            "unsloth/GLM-4.6-GGUF",
            " unsloth/GLM-4.6-GGUF ",
            "https://huggingface.co/unsloth/GLM-4.6-GGUF",
            "https://huggingface.co/unsloth/GLM-4.6-GGUF/tree/main/UD-Q4_K_XL",
            "huggingface.co/unsloth/GLM-4.6-GGUF/blob/main/x.gguf",
        ):
            self.assertEqual("unsloth/GLM-4.6-GGUF", normalize_repo_id(raw))

    def test_model_download_dir_uses_repo_name_under_root(self):
        self.assertEqual(
            r"C:\localai\GLM-4.6-GGUF",
            model_download_dir("unsloth/GLM-4.6-GGUF"),
        )
        self.assertEqual(
            str(Path(r"D:\models") / "GLM-4.6-GGUF"),
            model_download_dir("https://huggingface.co/unsloth/GLM-4.6-GGUF", r"D:\models"),
        )

    def test_parses_gguf_files_from_tree_payload(self):
        payload = [
            {"type": "file", "path": "UD-Q4_K_XL/model-UD-Q4_K_XL-00001-of-00002.gguf", "size": 2**30},
            {"type": "file", "path": "mmproj-F16.gguf", "size": 2**29},
            {"type": "file", "path": "README.md", "size": 100},
            {"type": "directory", "path": "UD-Q4_K_XL"},
        ]

        files = parse_repo_gguf_files(payload)

        self.assertEqual(
            ["mmproj-F16.gguf", "UD-Q4_K_XL/model-UD-Q4_K_XL-00001-of-00002.gguf"],
            [item.path for item in files],
        )
        self.assertAlmostEqual(1.0, files[1].size_gib)
        self.assertIn("(0.50 GiB)", files[0].label)

    def test_selects_quant_matches_plus_mmproj_and_mtp(self):
        paths = [
            "UD-Q4_K_XL/model-UD-Q4_K_XL-00001-of-00002.gguf",
            "UD-Q4_K_XL/model-UD-Q4_K_XL-00002-of-00002.gguf",
            "Q8_0/model-Q8_0.gguf",
            "mmproj-F16.gguf",
            "model-mtp-F16.gguf",
        ]

        selected = select_model_files(paths, "q4_k_xl")

        self.assertEqual(
            [
                "UD-Q4_K_XL/model-UD-Q4_K_XL-00001-of-00002.gguf",
                "UD-Q4_K_XL/model-UD-Q4_K_XL-00002-of-00002.gguf",
                "mmproj-F16.gguf",
                "model-mtp-F16.gguf",
            ],
            selected,
        )

    def test_blank_pattern_selects_only_auxiliary_files(self):
        paths = ["Q8_0/model-Q8_0.gguf", "mmproj-F16.gguf"]

        self.assertEqual(["mmproj-F16.gguf"], select_model_files(paths, ""))

    def test_builds_hf_download_command(self):
        command = build_hf_download_command(
            "https://huggingface.co/unsloth/GLM-4.6-GGUF",
            ["UD-Q4_K_XL/a.gguf", "mmproj-F16.gguf"],
            r"C:\localai\GLM-4.6-GGUF",
        )

        self.assertEqual(
            [
                "hf",
                "download",
                "unsloth/GLM-4.6-GGUF",
                "UD-Q4_K_XL/a.gguf",
                "mmproj-F16.gguf",
                "--local-dir",
                r"C:\localai\GLM-4.6-GGUF",
            ],
            command,
        )


class ModelDiscoveryTests(unittest.TestCase):
    def test_discovers_gguf_recursively_and_ignores_other_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            nested = root / "nested"
            nested.mkdir()
            model = nested / "model.gguf"
            model.write_bytes(b"123")
            (nested / "mmproj-model.gguf").write_bytes(b"123")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")

            models = discover_models([root])

            self.assertEqual([str(model.resolve())], [item.path for item in models])
            self.assertEqual("model.gguf", models[0].name)

    def test_deduplicates_overlapping_model_roots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            nested = root / "nested"
            nested.mkdir()
            model = nested / "model.gguf"
            model.write_bytes(b"x")

            models = discover_models([root, nested])

            self.assertEqual(1, len(models))


class CommandTests(unittest.TestCase):
    def test_parses_llama_server_device_output(self):
        output = """Available devices:
  Vulkan0: AMD Radeon AI PRO R9700 (32624 MiB, 29782 MiB free)
"""

        self.assertEqual(["Vulkan0"], parse_devices(output))

    def test_builds_expected_llama_server_command(self):
        profile = dict(DEFAULT_PROFILE)
        profile.update(
            {
                "model": r"C:\models\code model.gguf",
                "device": "Vulkan0",
                "gpu_layers": "all",
                "fit": "on",
                "context_size": 65536,
                "parallel": 1,
                "flash_attn": "on",
                "jinja": True,
                "sleep_idle_seconds": 300,
                "alias": "Code Model",
                "host": "127.0.0.1",
                "port": 8085,
                "extra_args": "--metrics\n--log-prefix",
            }
        )

        command = build_server_command(r"C:\tools\llama-server.exe", profile)

        self.assertEqual(r"C:\tools\llama-server.exe", command[0])
        self.assertEqual(
            [
                "--model",
                r"C:\models\code model.gguf",
                "--device",
                "Vulkan0",
                "--gpu-layers",
                "all",
                "--fit",
                "on",
                "--ctx-size",
                "65536",
                "--parallel",
                "1",
                "--flash-attn",
                "on",
                "--jinja",
                "--sleep-idle-seconds",
                "300",
                "--alias",
                "Code Model",
                "--host",
                "127.0.0.1",
                "--port",
                "8085",
                "--metrics",
                "--log-prefix",
            ],
            command[1:],
        )

    def test_defaults_omit_performance_and_memory_flags(self):
        profile = dict(DEFAULT_PROFILE)
        profile["model"] = r"C:\models\m.gguf"

        command = build_server_command(r"C:\tools\llama-server.exe", profile)

        for flag in (
            "--threads",
            "--cache-type-k",
            "--cache-type-v",
            "--batch-size",
            "--ubatch-size",
            "--cpu-moe",
            "--n-cpu-moe",
            "--mlock",
            "--no-mmap",
        ):
            self.assertNotIn(flag, command)

    def test_builds_command_with_performance_and_memory_flags(self):
        profile = dict(DEFAULT_PROFILE)
        profile.update(
            {
                "model": r"C:\models\m.gguf",
                "threads": "8",
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
                "batch_size": "1024",
                "ubatch_size": "256",
                "n_cpu_moe": "12",
                "mlock": True,
                "no_mmap": True,
            }
        )

        command = build_server_command(r"C:\tools\llama-server.exe", profile)

        self.assertEqual(["--threads", "8"], command[command.index("--threads"):command.index("--threads") + 2])
        self.assertEqual(["--cache-type-k", "q8_0"], command[command.index("--cache-type-k"):command.index("--cache-type-k") + 2])
        self.assertEqual(["--cache-type-v", "q8_0"], command[command.index("--cache-type-v"):command.index("--cache-type-v") + 2])
        self.assertEqual(["--batch-size", "1024"], command[command.index("--batch-size"):command.index("--batch-size") + 2])
        self.assertEqual(["--ubatch-size", "256"], command[command.index("--ubatch-size"):command.index("--ubatch-size") + 2])
        self.assertEqual(["--n-cpu-moe", "12"], command[command.index("--n-cpu-moe"):command.index("--n-cpu-moe") + 2])
        self.assertIn("--mlock", command)
        self.assertIn("--no-mmap", command)

    def test_cpu_moe_all_takes_precedence_over_n_cpu_moe(self):
        profile = dict(DEFAULT_PROFILE)
        profile.update({"model": r"C:\models\m.gguf", "cpu_moe": True, "n_cpu_moe": "12"})

        command = build_server_command(r"C:\tools\llama-server.exe", profile)

        self.assertIn("--cpu-moe", command)
        self.assertNotIn("--n-cpu-moe", command)

    def test_validation_rejects_bad_performance_values(self):
        profile = dict(DEFAULT_PROFILE)
        profile.update(
            {
                "model": __file__,
                "threads": "abc",
                "batch_size": "0",
                "n_cpu_moe": "-1",
                "cache_type_k": "q3_bogus",
            }
        )

        errors = validate_profile(profile)

        self.assertTrue(any("Threads" in e for e in errors))
        self.assertTrue(any("Batch size" in e for e in errors))
        self.assertTrue(any("CPU MoE layers" in e for e in errors))
        self.assertTrue(any("KV cache type (K)" in e for e in errors))

    def test_parses_additional_arguments_line_by_line(self):
        extra = '--chat-template-file "C:\\templates\\tool use.jinja"\n--metrics\n--alias ignored'

        self.assertEqual(
            [
                "--chat-template-file",
                r"C:\templates\tool use.jinja",
                "--metrics",
                "--alias",
                "ignored",
            ],
            parse_extra_args(extra),
        )

    def test_builds_router_command_without_single_model_arguments(self):
        profile = {
            **DEFAULT_PROFILE,
            "mode": "router",
            "models_dir": r"C:\localai",
            "models_preset": r"C:\localai\router.ini",
            "models_max": 2,
            "models_autoload": False,
            "context_size": 32768,
            "parallel": 2,
            "flash_attn": "auto",
            "jinja": False,
            "sleep_idle_seconds": "",
            "alias": "",
            "host": "127.0.0.1",
            "port": 8085,
            "extra_args": "--metrics",
        }

        command = build_server_command(r"C:\tools\llama-server.exe", profile)

        self.assertEqual(
            [
                r"C:\tools\llama-server.exe",
                "--models-dir",
                r"C:\localai",
                "--models-preset",
                r"C:\localai\router.ini",
                "--models-max",
                "2",
                "--no-models-autoload",
                "--flash-attn",
                "auto",
                "--no-jinja",
                "--host",
                "127.0.0.1",
                "--port",
                "8085",
                "--metrics",
            ],
            command,
        )
        for argument in (
            "--model",
            "--device",
            "--gpu-layers",
            "--fit",
            "--ctx-size",
            "--parallel",
        ):
            self.assertNotIn(argument, command)

    def test_builds_router_command_with_autoload_and_no_preset(self):
        profile = {
            **DEFAULT_PROFILE,
            "mode": "router",
            "models_dir": r"C:\localai",
            "models_preset": "",
            "models_max": 0,
            "models_autoload": True,
        }

        command = build_server_command(r"C:\tools\llama-server.exe", profile)

        self.assertIn("--models-autoload", command)
        self.assertNotIn("--models-preset", command)
        self.assertEqual("0", command[command.index("--models-max") + 1])

    def test_validation_rejects_missing_model_and_invalid_port(self):
        profile = dict(DEFAULT_PROFILE)
        profile["model"] = ""
        profile["port"] = 70000

        errors = validate_profile(profile)

        self.assertIn("Select a GGUF model.", errors)
        self.assertIn("Port must be between 1 and 65535.", errors)

    def test_router_validation_requires_directory_and_valid_router_values(self):
        profile = {
            **DEFAULT_PROFILE,
            "mode": "router",
            "model": "",
            "models_dir": "",
            "models_preset": r"C:\missing\router.ini",
            "models_max": -1,
            "flash_attn": "maybe",
            "sleep_idle_seconds": "soon",
        }

        errors = validate_profile(profile)

        self.assertNotIn("Select a GGUF model.", errors)
        self.assertIn("Select a models directory.", errors)
        self.assertIn("The selected models preset does not exist.", errors)
        self.assertIn("Maximum loaded models must be zero or a positive integer.", errors)
        self.assertIn("Flash attention must be auto, on, or off.", errors)
        self.assertIn("Sleep idle seconds must be blank or an integer greater than or equal to -1.", errors)

    def test_router_validation_accepts_existing_directory_without_model(self):
        with tempfile.TemporaryDirectory() as temp:
            profile = {
                **DEFAULT_PROFILE,
                "mode": "router",
                "model": "",
                "models_dir": temp,
                "models_max": 4,
            }

            self.assertEqual([], validate_profile(profile))


class HuggingFaceCacheTests(unittest.TestCase):
    def test_parses_router_models_status_case_insensitively(self):
        payload = {
            "data": [
                {"id": "gpt-oss-20b", "status": {"value": "LOADED"}},
                {"id": "gemma-4-12b-q4", "status": {"value": "unloaded"}},
                {"id": "qwen3-embedding-0.6b", "status": "Sleeping"},
                {"id": "no-status"},
            ]
        }

        models = parse_router_models(payload)

        self.assertEqual(
            [
                {"id": "gpt-oss-20b", "status": "loaded"},
                {"id": "gemma-4-12b-q4", "status": "unloaded"},
                {"id": "qwen3-embedding-0.6b", "status": "sleeping"},
                {"id": "no-status", "status": ""},
            ],
            models,
        )

    def test_parses_quiet_hf_cache_model_ids(self):
        output = """
model/Qwen/Qwen3-Embedding-0.6B-GGUF
dataset/nyu-mll/glue
model/ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF
"""

        self.assertEqual(
            [
                "model/Qwen/Qwen3-Embedding-0.6B-GGUF",
                "model/ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF",
            ],
            parse_hf_cache_model_ids(output),
        )

    def test_builds_hf_cache_commands(self):
        self.assertEqual(
            ["hf", "cache", "ls", "--cache-dir", r"C:\localai", "--format", "quiet"],
            build_hf_cache_list_command(r"C:\localai"),
        )
        self.assertEqual(
            [
                "hf",
                "cache",
                "rm",
                "model/ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF",
                "--cache-dir",
                r"C:\localai",
                "--dry-run",
            ],
            build_hf_cache_remove_command(
                r"C:\localai",
                "model/ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF",
                dry_run=True,
            ),
        )
        self.assertEqual(
            [
                "hf",
                "cache",
                "rm",
                "model/ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF",
                "--cache-dir",
                r"C:\localai",
                "--yes",
            ],
            build_hf_cache_remove_command(
                r"C:\localai",
                "model/ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF",
                dry_run=False,
            ),
        )


class SettingsStoreTests(unittest.TestCase):
    def test_round_trips_multiple_profiles(self):
        with tempfile.TemporaryDirectory() as temp:
            settings_path = Path(temp) / "settings.json"
            store = SettingsStore(settings_path)
            settings = store.defaults()
            settings["profiles"]["Coding"] = {
                **DEFAULT_PROFILE,
                "context_size": 65536,
            }
            settings["selected_profile"] = "Coding"
            store.save(settings)

            loaded = store.load()

            self.assertEqual("Coding", loaded["selected_profile"])
            self.assertEqual(65536, loaded["profiles"]["Coding"]["context_size"])
            self.assertIn("Default", loaded["profiles"])

    def test_load_recovers_from_invalid_json(self):
        with tempfile.TemporaryDirectory() as temp:
            settings_path = Path(temp) / "settings.json"
            settings_path.write_text("{broken", encoding="utf-8")

            loaded = SettingsStore(settings_path).load()

            self.assertEqual("Default", loaded["selected_profile"])
            self.assertEqual(DEFAULT_PROFILE, loaded["profiles"]["Default"])

    def test_load_upgrades_existing_single_model_profile_with_router_defaults(self):
        with tempfile.TemporaryDirectory() as temp:
            settings_path = Path(temp) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "selected_profile": "Default",
                        "profiles": {"Default": {"model": r"C:\models\old.gguf"}},
                    }
                ),
                encoding="utf-8",
            )

            loaded = SettingsStore(settings_path).load()

            profile = loaded["profiles"]["Default"]
            self.assertEqual("single", profile["mode"])
            self.assertEqual("", profile["models_dir"])
            self.assertEqual(4, profile["models_max"])
            self.assertTrue(profile["models_autoload"])


class ManagedServerTests(unittest.TestCase):
    def test_stops_only_the_process_it_started(self):
        logs = []
        server = ManagedServer(logs.append)
        server.start(
            [
                sys.executable,
                "-c",
                "import time; print('ready', flush=True); time.sleep(60)",
            ]
        )
        try:
            time.sleep(0.3)
            self.assertTrue(server.running)
            server.stop(timeout=2)
            self.assertFalse(server.running)
        finally:
            server.stop(timeout=1)


if __name__ == "__main__":
    unittest.main()
