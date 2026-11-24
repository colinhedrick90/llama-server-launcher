from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from launcher_core import (
    CACHE_TYPES,
    DEFAULT_DOWNLOAD_ROOT,
    DEFAULT_PROFILE,
    FIM_PROXY_CONFIG,
    IS_WINDOWS,
    ROUTER_LOADED_STATUSES,
    ManagedServer,
    SettingsStore,
    build_server_command,
    default_hf_cache_dir,
    default_model_root,
    discover_models,
    find_fim_proxy_executable,
    fim_ext_config_load,
    fim_ext_config_save,
    fim_proxy_base_url,
    fim_proxy_health,
    fim_proxy_log_path,
    fim_proxy_restart,
    fim_proxy_start,
    fim_proxy_status,
    fim_proxy_stop,
    fim_retrieval_enabled_get,
    fim_retrieval_enabled_set,
    huggingface_cache_path,
    is_port_available,
    list_hf_cache_models,
    list_devices,
    list_repo_gguf_files,
    list_router_models,
    model_download_dir,
    normalize_repo_id,
    router_loaded_models,
    run_hf_cache_remove,
    run_hf_download,
    select_model_files,
    unload_router_model,
    validate_profile,
)


# "monospace" is a fontconfig alias; Windows Tk has no such family and silently
# falls back to a proportional font, and Consolas does not exist on Linux.
MONO_FONT = ("Consolas" if IS_WINDOWS else "monospace", 9)

# On Linux Tk, "*.*" only matches filenames containing a literal dot and hides
# extensionless binaries like llama-server; "*" matches every file.
ALL_FILES_PATTERN = "*.*" if IS_WINDOWS else "*"

class _Tooltip:
    """Minimal hover tooltip — Tkinter ships none."""

    def __init__(self, widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None) -> None:
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self.tip,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            wraplength=340,
            padx=6,
            pady=4,
        ).pack()

    def _hide(self, _event=None) -> None:
        if self.tip:
            self.tip.destroy()
            self.tip = None


class LauncherApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Llama Server Launcher")
        self.root.geometry("960x800")
        self.root.minsize(800, 700)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.store = SettingsStore()
        self.settings = self.store.load()
        self.models = []
        self.model_by_label = {}
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.server = ManagedServer(self.log_queue.put)

        self.profile_name = tk.StringVar()
        self.server_path = tk.StringVar(value=self.settings["llama_server"])
        self.custom_folder = tk.StringVar(value=self.settings["custom_model_folder"])
        self.mode = tk.StringVar()
        self.model = tk.StringVar()
        self.models_dir = tk.StringVar()
        self.models_preset = tk.StringVar()
        self.models_max = tk.StringVar()
        self.models_autoload = tk.BooleanVar()
        self.device = tk.StringVar()
        self.gpu_layers = tk.StringVar()
        self.fit = tk.StringVar()
        self.context_size = tk.StringVar()
        self.parallel = tk.StringVar()
        self.flash_attn = tk.StringVar()
        self.jinja = tk.BooleanVar()
        self.threads = tk.StringVar()
        self.cache_type_k = tk.StringVar()
        self.cache_type_v = tk.StringVar()
        self.batch_size = tk.StringVar()
        self.ubatch_size = tk.StringVar()
        self.n_cpu_moe = tk.StringVar()
        self.cpu_moe = tk.BooleanVar()
        self.mlock = tk.BooleanVar()
        self.no_mmap = tk.BooleanVar()
        self.sleep_idle_seconds = tk.StringVar()
        self.alias = tk.StringVar()
        self.host = tk.StringVar()
        self.port = tk.StringVar()
        self.hf_cache_dir = tk.StringVar(
            value=self.settings.get("hf_cache_dir") or default_hf_cache_dir()
        )
        self.hf_cache_model = tk.StringVar()
        self.hf_repo = tk.StringVar()
        self.hf_quant = tk.StringVar()
        self.hf_dest = tk.StringVar(value="Destination: (enter a repo id)")
        self.hf_repo_files = []
        self.hf_download_running = False
        self.loaded_model = tk.StringVar()
        self.status = tk.StringVar(value="Stopped")
        self.single_mode_widgets = []
        self.router_mode_widgets = []

        # FIM context proxy tab
        self.fim_status = tk.StringVar(value="Unknown")
        self.fim_process_state = tk.StringVar(value="-")
        self.fim_backend = tk.StringVar(value="-")
        self.fim_chunks = tk.StringVar(value="-")
        self.fim_config = tk.StringVar(
            value=self.settings.get("fim_proxy_config") or FIM_PROXY_CONFIG
        )
        self.fim_endpoint = tk.StringVar(value=fim_proxy_base_url(self.fim_config.get()))
        self.fim_exe = tk.StringVar(
            value=self.settings.get("fim_proxy_executable")
            or find_fim_proxy_executable(self.fim_config.get())
        )
        self.fim_workspace = tk.StringVar(value=self.settings.get("fim_proxy_workspace") or "")

        # fim-ghost extension ghost-text settings (fim.config.json, hot-reloads)
        self.fim_enabled = tk.BooleanVar(value=True)
        self.fim_debounce = tk.StringVar(value="200")
        self.fim_max_prefix = tk.StringVar(value="2000")
        self.fim_max_suffix = tk.StringVar(value="512")
        self.fim_n_predict = tk.StringVar(value="64")
        self.fim_stop = tk.StringVar(value="\\n\\n")
        self.fim_show_status = tk.BooleanVar(value=True)
        self.fim_retrieval_enabled = tk.BooleanVar(value=True)
        self.fim_settings_status = tk.StringVar(value="")
        # fim-ghost sampling params (passed through to llama.cpp /infill)
        self.fim_temperature = tk.StringVar(value="0.1")
        self.fim_top_p = tk.StringVar(value="0.95")
        self.fim_top_k = tk.StringVar(value="40")
        self.fim_freq_pen = tk.StringVar(value="0.0")
        self.fim_pres_pen = tk.StringVar(value="0.0")
        self.fim_repeat_pen = tk.StringVar(value="1.05")

        self._build_ui()
        self._refresh_profiles()
        self.profile_name.set(self.settings["selected_profile"])
        self._load_selected_profile()
        self.refresh_devices()
        self.scan_models()
        self.refresh_fim_status()
        self._poll()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)

        # Every pane is a tab, including the log. The log used to sit below the
        # notebook sharing vertical space with the Server controls, which pushed
        # the lower Server rows off screen on shorter displays.
        notebook = ttk.Notebook(outer)
        notebook.grid(row=0, column=0, columnspan=2, sticky="nsew")
        outer.rowconfigure(0, weight=1)

        # --- Server tab, inside a scrollable canvas -------------------------
        # The Server tab is tall (profile + config + router + download + cache +
        # loaded-models). Scrolling means it degrades gracefully instead of
        # clipping whatever sits at the bottom.
        server_outer = ttk.Frame(notebook)
        notebook.add(server_outer, text="Server")
        server_outer.rowconfigure(0, weight=1)
        server_outer.columnconfigure(0, weight=1)

        canvas = tk.Canvas(server_outer, highlightthickness=0, borderwidth=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        vbar = ttk.Scrollbar(server_outer, orient="vertical", command=canvas.yview)
        vbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=vbar.set)

        server_tab = ttk.Frame(canvas, padding=8)
        server_tab.columnconfigure(1, weight=1)
        window = canvas.create_window((0, 0), window=server_tab, anchor="nw")

        def _sync_scrollregion(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _stretch_inner(event):
            # Keep the inner frame as wide as the canvas so widgets fill it.
            canvas.itemconfigure(window, width=event.width)

        server_tab.bind("<Configure>", _sync_scrollregion)
        canvas.bind("<Configure>", _stretch_inner)

        def _wheel(event):
            # Windows/macOS send <MouseWheel> with a delta; X11 sends Button-4/5.
            if getattr(event, "num", None) == 4:
                delta = -1
            elif getattr(event, "num", None) == 5:
                delta = 1
            else:
                delta = -1 if event.delta > 0 else 1
            canvas.yview_scroll(delta, "units")

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            canvas.bind_all(seq, lambda e: _wheel(e) if self._pointer_over(canvas) else None)

        fim_tab = ttk.Frame(notebook, padding=8)
        fim_tab.columnconfigure(1, weight=1)
        notebook.add(fim_tab, text="FIM Proxy")
        self._build_fim_tab(fim_tab)

        log_tab = ttk.Frame(notebook, padding=8)
        notebook.add(log_tab, text="Log")
        self._build_log_tab(log_tab)

        outer = server_tab  # existing rows below are laid out inside the Server tab

        profiles = ttk.LabelFrame(outer, text="Profile", padding=8)
        profiles.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        profiles.columnconfigure(1, weight=1)
        ttk.Label(profiles, text="Profile").grid(row=0, column=0, sticky="w")
        self.profile_combo = ttk.Combobox(profiles, textvariable=self.profile_name, state="readonly")
        self.profile_combo.grid(row=0, column=1, sticky="ew", padx=6)
        self.profile_combo.bind("<<ComboboxSelected>>", lambda _: self._load_selected_profile())
        ttk.Button(profiles, text="New", command=self.new_profile).grid(row=0, column=2, padx=2)
        ttk.Button(profiles, text="Save", command=self.save_profile).grid(row=0, column=3, padx=2)
        ttk.Button(profiles, text="Delete", command=self.delete_profile).grid(row=0, column=4, padx=2)

        config = ttk.LabelFrame(outer, text="Server Configuration", padding=8)
        config.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        config.columnconfigure(1, weight=1)

        row = 0
        row = self._path_row(config, row, "llama-server", self.server_path, self.browse_server, self.refresh_devices)
        ttk.Label(config, text="Launch mode").grid(row=row, column=0, sticky="w", pady=3)
        self.mode_combo = ttk.Combobox(
            config,
            textvariable=self.mode,
            values=("single", "router"),
            state="readonly",
        )
        self.mode_combo.grid(row=row, column=1, columnspan=3, sticky="ew", padx=6, pady=3)
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _: self._update_mode_state())
        row += 1

        row = self._path_row(
            config,
            row,
            "Custom model folder",
            self.custom_folder,
            self.browse_folder,
            self.scan_models,
            self.single_mode_widgets,
        )
        model_label = ttk.Label(config, text="GGUF model")
        model_label.grid(row=row, column=0, sticky="w", pady=3)
        self.model_combo = ttk.Combobox(config, textvariable=self.model, state="readonly")
        self.model_combo.grid(row=row, column=1, columnspan=2, sticky="ew", padx=6, pady=3)
        model_rescan = ttk.Button(config, text="Rescan", command=self.scan_models)
        model_rescan.grid(row=row, column=3, sticky="ew", pady=3)
        self._track(self.single_mode_widgets, model_label)
        self._track(self.single_mode_widgets, self.model_combo, "readonly")
        self._track(self.single_mode_widgets, model_rescan)
        row += 1

        fields = ttk.Frame(config)
        fields.grid(row=row, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        for column in (1, 3, 5, 7):
            fields.columnconfigure(column, weight=1)
        self.device_combo = self._field(
            fields, 0, "Device", self.device, combo=True, group=self.single_mode_widgets
        )
        self._field(fields, 2, "GPU layers", self.gpu_layers, group=self.single_mode_widgets)
        self._field(
            fields,
            4,
            "Fit",
            self.fit,
            combo=True,
            values=("on", "off"),
            group=self.single_mode_widgets,
        )
        self._field(fields, 6, "Context", self.context_size, group=self.single_mode_widgets)
        self._field(fields, 0, "Parallel", self.parallel, row=1, group=self.single_mode_widgets)
        self._field(fields, 2, "Host", self.host, row=1)
        self._field(fields, 4, "Port", self.port, row=1)
        self._field(
            fields,
            6,
            "Flash attention",
            self.flash_attn,
            combo=True,
            values=("auto", "on", "off"),
            row=1,
        )
        self._field(fields, 0, "Sleep idle", self.sleep_idle_seconds, row=2)
        self._field(fields, 2, "Alias", self.alias, row=2)
        ttk.Checkbutton(fields, text="Jinja", variable=self.jinja).grid(
            row=2, column=4, columnspan=2, sticky="w", padx=(0, 12), pady=2
        )
        self._field(fields, 0, "Threads", self.threads, row=3)
        self._field(fields, 2, "Batch", self.batch_size, row=3)
        self._field(fields, 4, "U-batch", self.ubatch_size, row=3)
        self._field(fields, 6, "CPU MoE layers", self.n_cpu_moe, row=3)
        self._field(
            fields, 0, "KV cache K", self.cache_type_k, combo=True, values=("",) + CACHE_TYPES, row=4
        )
        self._field(
            fields, 2, "KV cache V", self.cache_type_v, combo=True, values=("",) + CACHE_TYPES, row=4
        )
        ttk.Checkbutton(fields, text="CPU MoE (all)", variable=self.cpu_moe).grid(
            row=4, column=4, sticky="w", padx=(0, 12), pady=2
        )
        ttk.Checkbutton(fields, text="mlock", variable=self.mlock).grid(
            row=4, column=5, sticky="w", padx=(0, 12), pady=2
        )
        ttk.Checkbutton(fields, text="No mmap", variable=self.no_mmap).grid(
            row=4, column=6, columnspan=2, sticky="w", padx=(0, 12), pady=2
        )

        row += 1
        router = ttk.LabelFrame(config, text="Router Mode", padding=6)
        router.grid(row=row, column=0, columnspan=4, sticky="ew", pady=(6, 3))
        router.columnconfigure(1, weight=1)
        router_dir_label = ttk.Label(router, text="Models directory")
        router_dir_label.grid(row=0, column=0, sticky="w", pady=3)
        router_dir_entry = ttk.Entry(router, textvariable=self.models_dir)
        router_dir_entry.grid(row=0, column=1, sticky="ew", padx=6, pady=3)
        router_dir_browse = ttk.Button(router, text="Browse", command=self.browse_models_dir)
        router_dir_browse.grid(row=0, column=2, sticky="ew", pady=3)
        preset_label = ttk.Label(router, text="Models preset (optional)")
        preset_label.grid(row=1, column=0, sticky="w", pady=3)
        preset_entry = ttk.Entry(router, textvariable=self.models_preset)
        preset_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=3)
        preset_browse = ttk.Button(router, text="Browse", command=self.browse_models_preset)
        preset_browse.grid(row=1, column=2, sticky="ew", pady=3)
        max_label = ttk.Label(router, text="Maximum loaded models")
        max_label.grid(row=2, column=0, sticky="w", pady=3)
        max_entry = ttk.Entry(router, textvariable=self.models_max, width=12)
        max_entry.grid(row=2, column=1, sticky="w", padx=6, pady=3)
        autoload = ttk.Checkbutton(router, text="Automatically load requested models", variable=self.models_autoload)
        autoload.grid(row=2, column=2, sticky="w", pady=3)
        for widget in (
            router_dir_label,
            router_dir_entry,
            router_dir_browse,
            preset_label,
            preset_entry,
            preset_browse,
            max_label,
            max_entry,
            autoload,
        ):
            self._track(self.router_mode_widgets, widget)

        row += 1
        ttk.Label(config, text="Additional arguments").grid(row=row, column=0, sticky="w", pady=3)
        self.extra_args_text = tk.Text(config, height=4, wrap="none", font=MONO_FONT)
        self.extra_args_text.grid(
            row=row, column=1, columnspan=3, sticky="ew", padx=6, pady=3
        )

        row += 1
        download = ttk.LabelFrame(config, text="Download Model (Hugging Face)", padding=6)
        download.grid(row=row, column=0, columnspan=4, sticky="ew", pady=(6, 3))
        download.columnconfigure(1, weight=1)
        ttk.Label(download, text="Repo id or URL").grid(row=0, column=0, sticky="w", pady=3)
        repo_entry = ttk.Entry(download, textvariable=self.hf_repo)
        repo_entry.grid(row=0, column=1, sticky="ew", padx=6, pady=3)
        repo_entry.bind("<Return>", lambda _: self.list_hf_repo_files())
        ttk.Button(download, text="List Files", command=self.list_hf_repo_files).grid(
            row=0, column=2, columnspan=2, sticky="ew", pady=3
        )
        ttk.Label(download, text="Quant filter").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(download, textvariable=self.hf_quant).grid(
            row=1, column=1, sticky="ew", padx=6, pady=3
        )
        ttk.Button(download, text="Select Matching", command=self.select_matching_hf_files).grid(
            row=1, column=2, sticky="ew", pady=3
        )
        self.hf_download_button = ttk.Button(
            download, text="Download Selected", command=self.download_hf_files
        )
        self.hf_download_button.grid(row=1, column=3, sticky="ew", padx=(6, 0), pady=3)
        files_frame = ttk.Frame(download)
        files_frame.grid(row=2, column=0, columnspan=4, sticky="ew", pady=3)
        files_frame.columnconfigure(0, weight=1)
        self.hf_files_list = tk.Listbox(
            files_frame,
            height=6,
            selectmode="extended",
            exportselection=False,
            font=MONO_FONT,
        )
        self.hf_files_list.grid(row=0, column=0, sticky="ew")
        files_scroll = ttk.Scrollbar(files_frame, command=self.hf_files_list.yview)
        files_scroll.grid(row=0, column=1, sticky="ns")
        self.hf_files_list.configure(yscrollcommand=files_scroll.set)
        ttk.Label(download, textvariable=self.hf_dest).grid(
            row=3, column=0, columnspan=4, sticky="w", pady=(0, 3)
        )
        self.hf_repo.trace_add("write", lambda *_: self._update_hf_dest())

        row += 1
        cache = ttk.LabelFrame(config, text="Hugging Face Cache Removal", padding=6)
        cache.grid(row=row, column=0, columnspan=4, sticky="ew", pady=(6, 3))
        cache.columnconfigure(1, weight=1)
        ttk.Label(cache, text="Cache directory").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(cache, textvariable=self.hf_cache_dir).grid(
            row=0, column=1, sticky="ew", padx=6, pady=3
        )
        ttk.Button(cache, text="Browse", command=self.browse_hf_cache_dir).grid(
            row=0, column=2, sticky="ew", pady=3
        )
        ttk.Button(cache, text="Refresh", command=self.refresh_hf_cache_models).grid(
            row=0, column=3, sticky="ew", padx=(6, 0), pady=3
        )
        ttk.Label(cache, text="Cached model").grid(row=1, column=0, sticky="w", pady=3)
        self.hf_cache_combo = ttk.Combobox(cache, textvariable=self.hf_cache_model, state="readonly")
        self.hf_cache_combo.grid(row=1, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(cache, text="Preview Remove", command=self.preview_remove_hf_cache_model).grid(
            row=1, column=2, sticky="ew", pady=3
        )
        ttk.Button(cache, text="Remove", command=self.remove_hf_cache_model).grid(
            row=1, column=3, sticky="ew", padx=(6, 0), pady=3
        )

        row += 1
        loaded = ttk.LabelFrame(config, text="Loaded Models (VRAM)", padding=6)
        loaded.grid(row=row, column=0, columnspan=4, sticky="ew", pady=(6, 3))
        loaded.columnconfigure(1, weight=1)
        ttk.Label(loaded, text="Loaded model").grid(row=0, column=0, sticky="w", pady=3)
        self.loaded_combo = ttk.Combobox(loaded, textvariable=self.loaded_model, state="readonly")
        self.loaded_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(loaded, text="Refresh", command=self.refresh_loaded_models).grid(
            row=0, column=2, sticky="ew", pady=3
        )
        ttk.Button(loaded, text="Unload Selected", command=self.unload_selected_model).grid(
            row=0, column=3, sticky="ew", padx=(6, 0), pady=3
        )
        ttk.Button(loaded, text="Unload All", command=self.unload_all_models).grid(
            row=0, column=4, sticky="ew", padx=(6, 0), pady=3
        )

        controls = ttk.Frame(outer)
        controls.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Button(controls, text="Start Server", command=self.start_server).pack(side="left")
        ttk.Button(controls, text="Stop Server", command=self.stop_server).pack(side="left", padx=6)
        ttk.Button(controls, text="Restart Server", command=self.restart_server).pack(side="left")
        ttk.Label(controls, textvariable=self.status).pack(side="right", padx=8)

    def _pointer_over(self, widget) -> bool:
        """True when the mouse is inside widget - scopes the global wheel bind."""
        try:
            x, y = widget.winfo_pointerxy()
            return str(widget.winfo_containing(x, y)).startswith(str(widget))
        except (tk.TclError, KeyError):
            return False

    def _build_log_tab(self, parent) -> None:
        """Server log on its own tab.

        Previously this shared the window with the Server controls, so both were
        cramped. On its own tab it gets the full height, and the Server tab stops
        being clipped.
        """
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        bar = ttk.Frame(parent)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Button(bar, text="Clear", command=self.clear_log).pack(side="left")
        ttk.Button(bar, text="Copy All", command=self.copy_log).pack(side="left", padx=6)
        self.log_autoscroll = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Auto-scroll", variable=self.log_autoscroll).pack(side="left", padx=6)

        self.log_text = tk.Text(parent, wrap="none", height=30, state="disabled",
                                font=MONO_FONT)
        self.log_text.grid(row=1, column=0, sticky="nsew")
        vscroll = ttk.Scrollbar(parent, orient="vertical", command=self.log_text.yview)
        vscroll.grid(row=1, column=1, sticky="ns")
        hscroll = ttk.Scrollbar(parent, orient="horizontal", command=self.log_text.xview)
        hscroll.grid(row=2, column=0, sticky="ew")
        self.log_text.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)

    def clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def copy_log(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(self.log_text.get("1.0", "end-1c"))

    def _build_fim_tab(self, parent) -> None:
        """Start/stop the FIM context proxy.

        The launcher starts the proxy itself as a detached process (see
        launcher_core.fim_proxy_start), so it survives editor reloads and closing
        this window. A start is refused while the proxy port is already held.
        """
        info = ttk.LabelFrame(parent, text="llama-fim-context-proxy", padding=8)
        info.grid(row=0, column=0, columnspan=2, sticky="ew")
        info.columnconfigure(1, weight=1)

        rows = (
            ("Process", self.fim_process_state),
            ("Endpoint", self.fim_endpoint),
            ("Backend", self.fim_backend),
            ("Indexed chunks", self.fim_chunks),
        )
        for index, (label, variable) in enumerate(rows):
            ttk.Label(info, text=label).grid(row=index, column=0, sticky="w", pady=2)
            ttk.Label(info, textvariable=variable).grid(row=index, column=1, sticky="w", padx=8, pady=2)
        # What to launch. Path rows like llama-server on the Server tab: type or
        # Browse, Refresh re-detects / re-reads. All three are saved locally.
        row = len(rows)
        row = self._path_row(
            info, row, "Proxy executable", self.fim_exe,
            self.browse_fim_exe, self.refresh_fim_exe,
        )
        row = self._path_row(
            info, row, "Config", self.fim_config,
            self.browse_fim_config, self.refresh_fim_config,
        )
        self._path_row(
            info, row, "Workspace", self.fim_workspace,
            self.browse_fim_workspace, self.refresh_fim_workspace,
        )

        controls = ttk.Frame(parent)
        controls.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(controls, text="Start Proxy", command=self.start_fim_proxy).pack(side="left")
        ttk.Button(controls, text="Stop Proxy", command=self.stop_fim_proxy).pack(side="left", padx=6)
        ttk.Button(controls, text="Restart Proxy", command=self.restart_fim_proxy).pack(side="left")
        ttk.Button(controls, text="Refresh", command=self.refresh_fim_status).pack(side="left", padx=6)
        ttk.Label(controls, textvariable=self.fim_status).pack(side="right", padx=8)

        note = ttk.Label(
            parent,
            text=(
                "Start the proxy, then start your editor and it will survive editor "
                "reloads. Stopping it here also stops its llama-server backend. "
                "Workspace is the folder the proxy indexes; leave it blank to use the "
                f"proxy's own folder. Proxy output is written to {fim_proxy_log_path()}."
            ),
            wraplength=720,
            justify="left",
            foreground="gray40",
        )
        note.grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))

        settings = ttk.LabelFrame(
            parent, text="Ghost-text settings (fim.config.json - hot-reloads live)", padding=8
        )
        settings.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        settings.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            settings, text="Enabled (ghost text on)", variable=self.fim_enabled
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=2)

        def field(r, label, var, width=12):
            ttk.Label(settings, text=label).grid(row=r, column=0, sticky="w", pady=2)
            ttk.Entry(settings, textvariable=var, width=width).grid(
                row=r, column=1, sticky="w", padx=8, pady=2
            )

        field(1, "Debounce (ms)", self.fim_debounce)
        field(2, "Max prefix chars", self.fim_max_prefix)
        field(3, "Max suffix chars", self.fim_max_suffix)
        field(4, "Max predicted tokens", self.fim_n_predict)
        field(5, "Stop tokens (comma-sep, \\n = newline)", self.fim_stop, width=24)
        ttk.Checkbutton(
            settings, text="Show status-bar indicator", variable=self.fim_show_status
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=2)

        retr = ttk.Checkbutton(
            settings,
            text="Retrieval enabled (proxy - Restart Proxy to apply)",
            variable=self.fim_retrieval_enabled,
        )
        retr.grid(row=7, column=0, columnspan=2, sticky="w", pady=2)
        _Tooltip(
            retr,
            "Semantic code retrieval into the FIM context. This is a proxy config "
            "(config.toml), so Save writes it but it only takes effect after "
            "Restart Proxy. Even when on, it fires only when the line immediately "
            "before your cursor is a comment - the comment becomes the query.",
        )

        sampling = ttk.LabelFrame(
            parent, text="Sampling (hover a name for what it does)", padding=8
        )
        sampling.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        sampling.columnconfigure(1, weight=1)

        def sfield(r, name, var, rng, tip):
            lbl = ttk.Label(sampling, text=name)
            lbl.grid(row=r, column=0, sticky="w", pady=2)
            ttk.Entry(sampling, textvariable=var, width=10).grid(
                row=r, column=1, sticky="w", padx=8
            )
            ttk.Label(sampling, text=rng, foreground="gray50").grid(
                row=r, column=2, sticky="w"
            )
            _Tooltip(lbl, tip)

        sfield(0, "temperature", self.fim_temperature, "0.0 - 2.0",
               "Randomness of sampling. Low = deterministic, best for code FIM. 0 = greedy (argmax).")
        sfield(1, "top_p", self.fim_top_p, "0.0 - 1.0",
               "Nucleus sampling: only consider tokens within this cumulative probability. 1.0 = off.")
        sfield(2, "top_k", self.fim_top_k, "0 - 1000 (0 = off)",
               "Sample only from the k most likely tokens. 0 disables the cutoff (llama.cpp).")
        sfield(3, "frequency_penalty", self.fim_freq_pen, "-2.0 - 2.0",
               "Penalize tokens by how OFTEN they've appeared. >0 discourages verbatim repeats. Keep ~0 for code (code repeats legitimately).")
        sfield(4, "presence_penalty", self.fim_pres_pen, "-2.0 - 2.0",
               "Penalize tokens that appeared AT ALL. >0 pushes toward new tokens. Keep ~0 for code.")
        sfield(5, "repeat_penalty", self.fim_repeat_pen, ">= 0 (1.0 = none)",
               "llama.cpp repetition penalty on already-generated tokens. >1 discourages repetition; ~1.05 is a gentle default for code.")

        btns = ttk.Frame(parent)
        btns.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(btns, text="Reload", command=self.load_fim_settings).pack(side="left")
        ttk.Button(btns, text="Save", command=self.save_fim_settings).pack(
            side="left", padx=6
        )
        ttk.Label(btns, textvariable=self.fim_settings_status, foreground="gray40").pack(
            side="right", padx=8
        )

        self.load_fim_settings()

    def browse_fim_config(self) -> None:
        current = Path(self.fim_config.get() or FIM_PROXY_CONFIG).expanduser()
        path = filedialog.askopenfilename(
            initialdir=str(current.parent if current.parent.is_dir() else Path.home()),
            filetypes=[("TOML files", "*.toml"), ("All files", ALL_FILES_PATTERN)],
        )
        if path:
            self.fim_config.set(path)
            self.refresh_fim_config()

    def _save_fim_launch_settings(self) -> None:
        self.settings["fim_proxy_config"] = self.fim_config.get().strip() or FIM_PROXY_CONFIG
        self.settings["fim_proxy_executable"] = self.fim_exe.get().strip()
        self.settings["fim_proxy_workspace"] = self.fim_workspace.get().strip()
        self.store.save(self.settings)

    def refresh_fim_config(self) -> None:
        """Remember the chosen proxy config and re-read the settings kept in it."""
        path = self.fim_config.get().strip() or FIM_PROXY_CONFIG
        self.fim_config.set(path)
        if not self.fim_exe.get().strip():
            self.fim_exe.set(find_fim_proxy_executable(path))
        self._save_fim_launch_settings()
        self.fim_endpoint.set(fim_proxy_base_url(path))
        self.load_fim_settings()
        if not Path(path).expanduser().is_file():
            self.fim_settings_status.set("proxy config not found - check the Config path")
        self.refresh_fim_status()

    def browse_fim_exe(self) -> None:
        current = Path(self.fim_exe.get() or self.fim_config.get() or FIM_PROXY_CONFIG).expanduser()
        path = filedialog.askopenfilename(
            initialdir=str(current.parent if current.parent.is_dir() else Path.home()),
            filetypes=(
                [("Executable", "*.exe"), ("All files", ALL_FILES_PATTERN)]
                if IS_WINDOWS
                else [("All files", ALL_FILES_PATTERN)]
            ),
        )
        if path:
            self.fim_exe.set(path)
            self._save_fim_launch_settings()

    def refresh_fim_exe(self) -> None:
        """Re-detect the proxy executable when the box is blank or stale."""
        current = self.fim_exe.get().strip()
        if not current or not Path(current).expanduser().is_file():
            found = find_fim_proxy_executable(self.fim_config.get())
            self.fim_exe.set(found)
            self.fim_settings_status.set(
                "proxy executable detected" if found
                else "proxy executable not found - Browse to it"
            )
        self._save_fim_launch_settings()

    def browse_fim_workspace(self) -> None:
        current = Path(self.fim_workspace.get() or Path.home()).expanduser()
        path = filedialog.askdirectory(
            initialdir=str(current if current.is_dir() else Path.home())
        )
        if path:
            self.fim_workspace.set(path)
            self._save_fim_launch_settings()

    def refresh_fim_workspace(self) -> None:
        path = self.fim_workspace.get().strip()
        self.fim_workspace.set(path)
        self._save_fim_launch_settings()
        if path and not Path(path).expanduser().is_dir():
            self.fim_settings_status.set("workspace folder not found")

    def load_fim_settings(self) -> None:
        # Retrieval lives in the proxy config.toml, independent of the extension JSON.
        self.fim_retrieval_enabled.set(fim_retrieval_enabled_get(self.fim_config.get()))
        data = fim_ext_config_load()
        if not data:
            self.fim_settings_status.set("config not found")
            return
        self.fim_enabled.set(bool(data.get("enabled", True)))
        self.fim_debounce.set(str(data.get("debounceMs", 200)))
        self.fim_max_prefix.set(str(data.get("maxPrefixChars", 2000)))
        self.fim_max_suffix.set(str(data.get("maxSuffixChars", 512)))
        self.fim_n_predict.set(str(data.get("nPredict", 64)))
        stops = data.get("stop", ["\n\n"])
        if isinstance(stops, list):
            self.fim_stop.set(", ".join(s.replace("\n", "\\n") for s in stops))
        self.fim_show_status.set(bool(data.get("showStatus", True)))
        self.fim_temperature.set(str(data.get("temperature", 0.1)))
        self.fim_top_p.set(str(data.get("topP", 0.95)))
        self.fim_top_k.set(str(data.get("topK", 40)))
        self.fim_freq_pen.set(str(data.get("frequencyPenalty", 0.0)))
        self.fim_pres_pen.set(str(data.get("presencePenalty", 0.0)))
        self.fim_repeat_pen.set(str(data.get("repeatPenalty", 1.05)))
        self.fim_settings_status.set("loaded")

    def save_fim_settings(self) -> None:
        try:
            values = {
                "enabled": bool(self.fim_enabled.get()),
                "debounceMs": int(self.fim_debounce.get()),
                "maxPrefixChars": int(self.fim_max_prefix.get()),
                "maxSuffixChars": int(self.fim_max_suffix.get()),
                "nPredict": int(self.fim_n_predict.get()),
                "stop": [
                    s.strip().replace("\\n", "\n")
                    for s in self.fim_stop.get().split(",")
                    if s.strip()
                ],
                "showStatus": bool(self.fim_show_status.get()),
                "temperature": float(self.fim_temperature.get()),
                "topP": float(self.fim_top_p.get()),
                "topK": int(self.fim_top_k.get()),
                "frequencyPenalty": float(self.fim_freq_pen.get()),
                "presencePenalty": float(self.fim_pres_pen.get()),
                "repeatPenalty": float(self.fim_repeat_pen.get()),
            }
        except ValueError:
            messagebox.showerror(
                "Invalid FIM settings",
                "Debounce/char caps/tokens must be whole numbers; sampling values must be numeric.",
            )
            return
        try:
            path = fim_ext_config_save(values)
        except RuntimeError as exc:
            self.fim_settings_status.set(str(exc))
            return
        # Retrieval toggle -> proxy config.toml (takes effect on next proxy restart).
        retr_changed = False
        try:
            want = bool(self.fim_retrieval_enabled.get())
            config_path = self.fim_config.get()
            retr_changed = fim_retrieval_enabled_get(config_path) != want
            fim_retrieval_enabled_set(want, config_path)
        except (OSError, RuntimeError) as exc:
            self._append_log(f"[fim] retrieval toggle write failed: {exc}")
        if retr_changed:
            self.fim_settings_status.set("saved - Restart Proxy to apply retrieval")
        else:
            self.fim_settings_status.set("saved (live)")
        self._append_log(f"[fim] settings saved -> {path}")

    def refresh_fim_status(self) -> None:
        config = self.fim_config.get()  # Tk variables are read on the UI thread only

        def work():
            try:
                status = fim_proxy_status(config)
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                text = f"Status error: {exc}"
                self.root.after(0, lambda: self.fim_status.set(text))
                return
            self.root.after(0, lambda: self._apply_fim_status(status, config))

        threading.Thread(target=work, daemon=True).start()

    def _apply_fim_status(self, status: dict, config: str | None = None) -> None:
        if config is not None and config != self.fim_config.get():
            return  # answer for a config that has since been changed; a newer check is running
        process_state = status.get("process_state", "Unknown")
        self.fim_process_state.set(process_state)
        if status.get("endpoint"):
            self.fim_endpoint.set(status["endpoint"])
        if status.get("listening"):
            self.fim_status.set("Running")
            self.fim_backend.set(
                f"{status.get('backend', '?')} ({status.get('backend_mode', '?')}) "
                f"{status.get('backend_url', '')}"
            )
            chunks = status.get("indexed_chunks")
            self.fim_chunks.set("-" if chunks is None else f"{chunks:,}")
        else:
            if process_state.startswith("Running"):
                self.fim_status.set("Not responding")
            elif process_state.startswith("Starting"):
                self.fim_status.set("Starting...")
            else:
                self.fim_status.set("Stopped")
            self.fim_backend.set("-")
            self.fim_chunks.set("-")

    def _fim_action(self, action, label: str, wait_until_ready: bool = False) -> None:
        self.fim_status.set(f"{label}...")
        self._save_fim_launch_settings()
        config = self.fim_config.get()

        def work():
            try:
                message = action()
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                text = str(exc)
                self.root.after(0, lambda: messagebox.showerror(f"FIM proxy {label} failed", text))
                self.root.after(0, self.refresh_fim_status)
                return
            self.root.after(0, lambda: self._append_log(f"[fim] {message}"))
            if wait_until_ready:
                # The proxy loads its backend model before it answers /health.
                self.root.after(0, lambda: self.fim_status.set("starting..."))
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline and not fim_proxy_health(config):
                    time.sleep(1.0)
            else:
                time.sleep(0.5)
            self.root.after(0, self.refresh_fim_status)

        threading.Thread(target=work, daemon=True).start()

    def _fim_launch_args(self) -> tuple[str, str, str]:
        return self.fim_exe.get(), self.fim_config.get(), self.fim_workspace.get()

    def start_fim_proxy(self) -> None:
        args = self._fim_launch_args()
        self._fim_action(lambda: fim_proxy_start(*args), "start", wait_until_ready=True)

    def stop_fim_proxy(self) -> None:
        config = self.fim_config.get()
        self._fim_action(lambda: fim_proxy_stop(config), "stop")

    def restart_fim_proxy(self) -> None:
        args = self._fim_launch_args()
        self._fim_action(lambda: fim_proxy_restart(*args), "restart", wait_until_ready=True)

    def _path_row(self, parent, row, label, variable, browse_command, refresh_command, group=None):
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", pady=3)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", padx=6, pady=3)
        browse = ttk.Button(parent, text="Browse", command=browse_command)
        browse.grid(row=row, column=2, sticky="ew", pady=3)
        refresh = ttk.Button(parent, text="Refresh", command=refresh_command)
        refresh.grid(row=row, column=3, sticky="ew", pady=3)
        if group is not None:
            for widget in (label_widget, entry, browse, refresh):
                self._track(group, widget)
        return row + 1

    @staticmethod
    def _track(group, widget, enabled_state="normal"):
        group.append((widget, enabled_state))

    def _field(self, parent, column, label, variable, combo=False, values=(), row=0, group=None):
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=column, sticky="w", padx=(0, 4), pady=2)
        widget = (
            ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=12)
            if combo
            else ttk.Entry(parent, textvariable=variable, width=12)
        )
        widget.grid(row=row, column=column + 1, sticky="ew", padx=(0, 12), pady=2)
        if group is not None:
            self._track(group, label_widget)
            self._track(group, widget, "readonly" if combo else "normal")
        return widget

    def _update_mode_state(self) -> None:
        router_mode = self.mode.get() == "router"
        for widget, enabled_state in self.single_mode_widgets:
            widget.configure(state="disabled" if router_mode else enabled_state)
        for widget, enabled_state in self.router_mode_widgets:
            widget.configure(state=enabled_state if router_mode else "disabled")

    def _refresh_profiles(self) -> None:
        self.profile_combo["values"] = sorted(self.settings["profiles"], key=str.casefold)

    def _profile_from_form(self) -> dict:
        selected_model = self.model_by_label.get(self.model.get(), self.model.get())
        return {
            "mode": self.mode.get(),
            "model": selected_model,
            "models_dir": self.models_dir.get(),
            "models_preset": self.models_preset.get(),
            "models_max": self.models_max.get(),
            "models_autoload": self.models_autoload.get(),
            "device": self.device.get(),
            "gpu_layers": self.gpu_layers.get(),
            "fit": self.fit.get(),
            "context_size": self.context_size.get(),
            "parallel": self.parallel.get(),
            "flash_attn": self.flash_attn.get(),
            "jinja": self.jinja.get(),
            "threads": self.threads.get(),
            "cache_type_k": self.cache_type_k.get(),
            "cache_type_v": self.cache_type_v.get(),
            "batch_size": self.batch_size.get(),
            "ubatch_size": self.ubatch_size.get(),
            "n_cpu_moe": self.n_cpu_moe.get(),
            "cpu_moe": self.cpu_moe.get(),
            "mlock": self.mlock.get(),
            "no_mmap": self.no_mmap.get(),
            "sleep_idle_seconds": self.sleep_idle_seconds.get(),
            "alias": self.alias.get(),
            "host": self.host.get(),
            "port": self.port.get(),
            "extra_args": self._get_extra_args(),
        }

    def _load_selected_profile(self) -> None:
        profile = self.settings["profiles"].get(self.profile_name.get(), DEFAULT_PROFILE)
        self.mode.set(str(profile.get("mode", "single")))
        self.models_dir.set(str(profile.get("models_dir", "")))
        self.models_preset.set(str(profile.get("models_preset", "")))
        self.models_max.set(str(profile.get("models_max", 4)))
        self.models_autoload.set(bool(profile.get("models_autoload", True)))
        self.device.set(str(profile["device"]))
        self.gpu_layers.set(str(profile["gpu_layers"]))
        self.fit.set(str(profile["fit"]))
        self.context_size.set(str(profile["context_size"]))
        self.parallel.set(str(profile["parallel"]))
        self.flash_attn.set(str(profile.get("flash_attn", "auto")))
        self.jinja.set(bool(profile.get("jinja", True)))
        self.threads.set(str(profile.get("threads", "")))
        self.cache_type_k.set(str(profile.get("cache_type_k", "")))
        self.cache_type_v.set(str(profile.get("cache_type_v", "")))
        self.batch_size.set(str(profile.get("batch_size", "")))
        self.ubatch_size.set(str(profile.get("ubatch_size", "")))
        self.n_cpu_moe.set(str(profile.get("n_cpu_moe", "")))
        self.cpu_moe.set(bool(profile.get("cpu_moe", False)))
        self.mlock.set(bool(profile.get("mlock", False)))
        self.no_mmap.set(bool(profile.get("no_mmap", False)))
        self.sleep_idle_seconds.set(str(profile.get("sleep_idle_seconds", "")))
        self.alias.set(str(profile.get("alias", "")))
        self.host.set(str(profile["host"]))
        self.port.set(str(profile["port"]))
        self._set_extra_args(str(profile.get("extra_args", "")))
        self._select_model_path(str(profile["model"]))
        self._update_mode_state()

    def _get_extra_args(self) -> str:
        return self.extra_args_text.get("1.0", "end-1c")

    def _set_extra_args(self, value: str) -> None:
        self.extra_args_text.delete("1.0", "end")
        self.extra_args_text.insert("1.0", value)

    def _select_model_path(self, path: str) -> None:
        for label, model_path in self.model_by_label.items():
            if model_path.casefold() == path.casefold():
                self.model.set(label)
                return
        self.model.set(path)

    def new_profile(self) -> None:
        name = simpledialog.askstring("New profile", "Profile name:", parent=self.root)
        if not name or not name.strip():
            return
        name = name.strip()
        if name in self.settings["profiles"]:
            messagebox.showerror("Profile exists", f'A profile named "{name}" already exists.')
            return
        self.settings["profiles"][name] = self._profile_from_form()
        self.profile_name.set(name)
        self._refresh_profiles()
        self.save_profile()

    def save_profile(self) -> None:
        name = self.profile_name.get() or "Default"
        self.settings["profiles"][name] = self._profile_from_form()
        self.settings["selected_profile"] = name
        self.settings["llama_server"] = self.server_path.get()
        self.settings["custom_model_folder"] = self.custom_folder.get()
        self.settings["hf_cache_dir"] = self.hf_cache_dir.get()
        self.settings["fim_proxy_config"] = self.fim_config.get().strip() or FIM_PROXY_CONFIG
        self.settings["fim_proxy_executable"] = self.fim_exe.get().strip()
        self.settings["fim_proxy_workspace"] = self.fim_workspace.get().strip()
        self.store.save(self.settings)
        self._refresh_profiles()
        self._append_log(f"[launcher] Saved profile: {name}")

    def delete_profile(self) -> None:
        name = self.profile_name.get()
        if name == "Default":
            messagebox.showinfo("Default profile", "The Default profile cannot be deleted.")
            return
        if not messagebox.askyesno("Delete profile", f'Delete profile "{name}"?'):
            return
        self.settings["profiles"].pop(name, None)
        self.profile_name.set("Default")
        self._refresh_profiles()
        self._load_selected_profile()
        self.save_profile()

    def browse_server(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=(
                [("Executable", "*.exe"), ("All files", ALL_FILES_PATTERN)]
                if IS_WINDOWS
                else [("All files", ALL_FILES_PATTERN)]
            )
        )
        if path:
            self.server_path.set(path)
            self.refresh_devices()

    def browse_folder(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.custom_folder.set(path)
            self.scan_models()

    def browse_models_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=self.models_dir.get() or str(default_model_root()))
        if path:
            self.models_dir.set(path)

    def browse_models_preset(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=self.models_dir.get() or str(default_model_root()),
            filetypes=[("INI files", "*.ini"), ("All files", ALL_FILES_PATTERN)],
        )
        if path:
            self.models_preset.set(path)

    def browse_hf_cache_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=self.hf_cache_dir.get() or str(default_model_root()))
        if path:
            self.hf_cache_dir.set(path)
            self.refresh_hf_cache_models()

    def scan_models(self) -> None:
        roots = [huggingface_cache_path()]
        if self.custom_folder.get().strip():
            roots.append(Path(self.custom_folder.get()))
        current_path = self.model_by_label.get(self.model.get(), self.model.get())
        self.models = discover_models(roots)
        counts = {}
        for item in self.models:
            counts[item.label] = counts.get(item.label, 0) + 1
        self.model_by_label = {}
        labels = []
        for item in self.models:
            label = item.label if counts[item.label] == 1 else f"{item.label} - {item.path}"
            labels.append(label)
            self.model_by_label[label] = item.path
        self.model_combo["values"] = labels
        self._select_model_path(current_path)
        self._append_log(f"[launcher] Found {len(labels)} GGUF files.")

    def refresh_hf_cache_models(self) -> None:
        cache_dir = self.hf_cache_dir.get().strip()
        if not cache_dir:
            messagebox.showerror("Cache directory required", "Select a Hugging Face cache directory.")
            return
        self.settings["hf_cache_dir"] = cache_dir
        self.store.save(self.settings)
        self._append_log(f"[launcher] Scanning HF cache: {cache_dir}")

        def worker() -> None:
            try:
                models = list_hf_cache_models(cache_dir)
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                self.root.after(0, lambda text=str(exc): messagebox.showerror("HF cache scan failed", text))
                return
            self.root.after(0, lambda: self._set_hf_cache_models(models))

        threading.Thread(target=worker, daemon=True).start()

    def _set_hf_cache_models(self, models: list[str]) -> None:
        current = self.hf_cache_model.get()
        self.hf_cache_combo["values"] = models
        if current in models:
            self.hf_cache_model.set(current)
        elif models:
            self.hf_cache_model.set(models[0])
        else:
            self.hf_cache_model.set("")
        self._append_log(f"[launcher] Found {len(models)} cached HF model repos.")

    def preview_remove_hf_cache_model(self) -> None:
        self._run_hf_cache_remove(dry_run=True)

    def remove_hf_cache_model(self) -> None:
        repo_id = self.hf_cache_model.get().strip()
        if not repo_id:
            messagebox.showerror("No cached model selected", "Select a cached Hugging Face model.")
            return
        if not messagebox.askyesno(
            "Remove cached model",
            f"Remove {repo_id} from the Hugging Face cache?\n\nThis runs hf cache rm with --yes.",
        ):
            return
        self._run_hf_cache_remove(dry_run=False)

    def _run_hf_cache_remove(self, dry_run: bool) -> None:
        cache_dir = self.hf_cache_dir.get().strip()
        repo_id = self.hf_cache_model.get().strip()
        if not cache_dir:
            messagebox.showerror("Cache directory required", "Select a Hugging Face cache directory.")
            return
        if not repo_id:
            messagebox.showerror("No cached model selected", "Select a cached Hugging Face model.")
            return
        action = "Previewing removal" if dry_run else "Removing cached model"
        self._append_log(f"[launcher] {action}: {repo_id}")

        def worker() -> None:
            try:
                output = run_hf_cache_remove(cache_dir, repo_id, dry_run=dry_run)
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                self.root.after(0, lambda text=str(exc): messagebox.showerror("HF cache removal failed", text))
                return

            def done() -> None:
                if output:
                    for line in output.splitlines():
                        self._append_log(line)
                if not dry_run:
                    self.refresh_hf_cache_models()

            self.root.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _download_root(self) -> str:
        return self.custom_folder.get().strip() or DEFAULT_DOWNLOAD_ROOT

    def _update_hf_dest(self) -> None:
        repo = normalize_repo_id(self.hf_repo.get())
        if "/" in repo:
            self.hf_dest.set(f"Destination: {model_download_dir(repo, self._download_root())}")
        else:
            self.hf_dest.set("Destination: (enter a repo id)")

    def list_hf_repo_files(self) -> None:
        repo = normalize_repo_id(self.hf_repo.get())
        if "/" not in repo:
            messagebox.showerror(
                "Repo id required", "Enter a Hugging Face repo id such as unsloth/model-GGUF."
            )
            return
        self._append_log(f"[launcher] Listing GGUF files in {repo}")

        def worker() -> None:
            try:
                files = list_repo_gguf_files(repo)
            except Exception as exc:
                self.root.after(0, lambda text=str(exc): messagebox.showerror("Listing failed", text))
                return
            self.root.after(0, lambda: self._set_hf_repo_files(files))

        threading.Thread(target=worker, daemon=True).start()

    def _set_hf_repo_files(self, files) -> None:
        self.hf_repo_files = files
        self.hf_files_list.delete(0, "end")
        for item in files:
            self.hf_files_list.insert("end", item.label)
        self._append_log(f"[launcher] Found {len(files)} GGUF files in repo.")
        if not files:
            messagebox.showinfo("No GGUF files", "The repo contains no GGUF files on main.")
            return
        if self.hf_quant.get().strip():
            self.select_matching_hf_files()

    def select_matching_hf_files(self) -> None:
        if not self.hf_repo_files:
            messagebox.showinfo("No files listed", "Use List Files first.")
            return
        matching = set(
            select_model_files(
                [item.path for item in self.hf_repo_files], self.hf_quant.get()
            )
        )
        self.hf_files_list.selection_clear(0, "end")
        for index, item in enumerate(self.hf_repo_files):
            if item.path in matching:
                self.hf_files_list.selection_set(index)
        self._append_log(
            f"[launcher] Selected {len(matching)} file(s) "
            f"(quant filter + mmproj/MTP auxiliaries)."
        )

    def download_hf_files(self) -> None:
        if self.hf_download_running:
            messagebox.showinfo("Download running", "A download is already in progress.")
            return
        repo = normalize_repo_id(self.hf_repo.get())
        if "/" not in repo:
            messagebox.showerror(
                "Repo id required", "Enter a Hugging Face repo id such as unsloth/model-GGUF."
            )
            return
        selected = [self.hf_repo_files[i].path for i in self.hf_files_list.curselection()]
        if not selected:
            messagebox.showerror(
                "No files selected", "Select files to download (or use Select Matching)."
            )
            return
        total_gib = sum(
            item.size_gib for item in self.hf_repo_files if item.path in set(selected)
        )
        local_dir = model_download_dir(repo, self._download_root())
        if not messagebox.askyesno(
            "Download model files",
            f"Download {len(selected)} file(s) ({total_gib:.2f} GiB) from {repo}\n"
            f"into {local_dir}?",
        ):
            return
        self.hf_download_running = True
        self.hf_download_button.configure(state="disabled")
        self._append_log(f"[launcher] Downloading {len(selected)} file(s) to {local_dir}")

        def worker() -> None:
            try:
                code = run_hf_download(repo, selected, local_dir, self.log_queue.put)
            except Exception as exc:
                self.root.after(0, lambda text=str(exc): messagebox.showerror("Download failed", text))
                code = -1
            def done() -> None:
                self.hf_download_running = False
                self.hf_download_button.configure(state="normal")
                if code == 0:
                    self._append_log("[launcher] Download complete.")
                    self.scan_models()
                else:
                    self._append_log(f"[launcher] Download exited with code {code}.")
            self.root.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _router_host_port(self) -> tuple[str, str] | None:
        host = self.host.get().strip() or "127.0.0.1"
        port = self.port.get().strip()
        if not port:
            messagebox.showerror("Port required", "Set the server port before managing loaded models.")
            return None
        return host, port

    def refresh_loaded_models(self) -> None:
        target = self._router_host_port()
        if not target:
            return
        host, port = target
        self._append_log(f"[launcher] Querying loaded models at {host}:{port}")

        def worker() -> None:
            try:
                models = list_router_models(host, port)
            except (OSError, ValueError) as exc:
                self.root.after(0, lambda text=str(exc): messagebox.showerror("Query failed", text))
                return
            self.root.after(0, lambda: self._set_loaded_models(models))

        threading.Thread(target=worker, daemon=True).start()

    def _set_loaded_models(self, models: list[dict]) -> None:
        loaded = [m["id"] for m in models if m["status"] in ROUTER_LOADED_STATUSES]
        current = self.loaded_model.get()
        self.loaded_combo["values"] = loaded
        if current in loaded:
            self.loaded_model.set(current)
        elif loaded:
            self.loaded_model.set(loaded[0])
        else:
            self.loaded_model.set("")
        self._append_log(f"[launcher] {len(loaded)} model(s) holding VRAM.")

    def unload_selected_model(self) -> None:
        model_id = self.loaded_model.get().strip()
        if not model_id:
            messagebox.showerror("No model selected", "Select a loaded model to unload.")
            return
        self._unload_models([model_id])

    def unload_all_models(self) -> None:
        target = self._router_host_port()
        if not target:
            return
        host, port = target
        if not messagebox.askyesno(
            "Unload all models",
            "Unload every model currently holding VRAM for a clean slate?",
        ):
            return
        self._append_log("[launcher] Unloading all loaded models...")

        def worker() -> None:
            try:
                model_ids = router_loaded_models(host, port)
            except (OSError, ValueError) as exc:
                self.root.after(0, lambda text=str(exc): messagebox.showerror("Query failed", text))
                return
            self.root.after(0, lambda: self._unload_models(model_ids))

        threading.Thread(target=worker, daemon=True).start()

    def _unload_models(self, model_ids: list[str]) -> None:
        target = self._router_host_port()
        if not target:
            return
        host, port = target
        if not model_ids:
            self._append_log("[launcher] No loaded models to unload.")
            return

        def worker() -> None:
            errors = []
            for model_id in model_ids:
                try:
                    unload_router_model(host, port, model_id)
                    self.log_queue.put(f"[launcher] Unloaded {model_id}.")
                except (OSError, ValueError) as exc:
                    errors.append(f"{model_id}: {exc}")
            if errors:
                self.root.after(0, lambda: messagebox.showerror("Unload failed", "\n".join(errors)))
            self.root.after(0, self.refresh_loaded_models)

        threading.Thread(target=worker, daemon=True).start()

    def refresh_devices(self) -> None:
        try:
            devices = list_devices(self.server_path.get())
        except (OSError, subprocess.SubprocessError) as exc:
            devices = []
            self._append_log(f"[launcher] Device detection failed: {exc}")
        self.device_combo["values"] = devices
        if devices and self.device.get() not in devices:
            self.device.set(devices[0])

    def start_server(self) -> None:
        if self.server.running:
            messagebox.showinfo("Server running", "The managed server is already running.")
            return
        profile = self._profile_from_form()
        errors = validate_profile(profile)
        server_path = self.server_path.get().strip()
        if not Path(server_path).is_file():
            errors.insert(0, "Select a valid llama-server executable.")
        try:
            port = int(profile["port"])
        except ValueError:
            port = 0
        if not errors and not is_port_available(str(profile["host"]), port):
            errors.append(f"Port {port} is already in use.")
        if errors:
            messagebox.showerror("Cannot start server", "\n".join(errors))
            return
        self.save_profile()
        self._sync_opencode_limits(profile)
        command = build_server_command(server_path, profile)
        self._append_log("[launcher] Starting: " + subprocess.list2cmdline(command))
        try:
            self.server.start(command, cwd=str(Path(server_path).parent))
        except OSError as exc:
            messagebox.showerror("Start failed", str(exc))
            return
        self.status.set(f"Running (PID {self.server.pid})")

    SYNC_SCRIPT = Path.home() / ".config" / "scripts" / "sync_model_limits.py"

    def _sync_opencode_limits(self, profile: dict) -> None:
        """Push preset ctx-sizes into opencode.json before the server starts.

        Best-effort: any failure is logged and startup proceeds regardless.
        """
        preset = str(profile.get("models_preset", "")).strip()
        if profile.get("mode") != "router" or not preset or not Path(preset).is_file():
            return
        if not self.SYNC_SCRIPT.is_file():
            return
        python = shutil.which("python") or shutil.which("py")
        if not python:
            self._append_log("[launcher] opencode sync skipped: no python on PATH")
            return
        try:
            result = subprocess.run(
                [python, str(self.SYNC_SCRIPT), preset],
                capture_output=True,
                text=True,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._append_log(f"[launcher] opencode sync failed: {exc}")
            return
        for line in (result.stdout + result.stderr).splitlines():
            if line.strip():
                self._append_log("[launcher] opencode sync: " + line)

    def stop_server(self) -> None:
        self.server.stop()
        self.status.set("Stopped")

    def restart_server(self) -> None:
        self.stop_server()
        self.start_server()

    def _append_log(self, line: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        # Only follow the tail when auto-scroll is on, so scrolling back to read
        # something is not yanked away by the next line of server output.
        if getattr(self, "log_autoscroll", None) is None or self.log_autoscroll.get():
            self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _poll(self) -> None:
        try:
            while True:
                self._append_log(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        if self.server.running:
            self.status.set(f"Running (PID {self.server.pid})")
        elif self.status.get().startswith("Running"):
            self.status.set("Stopped")
        self.root.after(1500, self._poll)

    def close(self) -> None:
        self.save_profile()
        self.stop_server()
        self.root.destroy()
