"""Tkinter-based GUI launcher for the INDI pipeline.

Layout
------
  ┌─────────────────────────────────────────────────────────────┐
  │  YAML settings file: [path_____________________________] [Browse]  │
  │  Data folder:        [path_____________________________] [Browse]  │
  │                                         [▶  Run]  [■  Stop]       │
  ├─────────────────────────────────────────────────────────────┤
  │  Log output                                                  │
  │  ┌───────────────────────────────────────────────────────┐  │
  │  │ INFO    : 12:00:00 :: Initialising pipeline …         │  │
  │  │ INFO    : 12:00:01 :: Reading data …                  │  │
  │  │ WARNING : 12:00:02 :: Low SNR on slice 2              │  │
  │  └───────────────────────────────────────────────────────┘  │
  ├─────────────────────────────────────────────────────────────┤
  │  Progress  [████████████░░░░░░]   Folder 2 / 5              │
  └─────────────────────────────────────────────────────────────┘

Matplotlib figures (outlier selection, segmentation review, etc.) open as
separate TkAgg windows that share the same underlying event loop so that
all interactive clicks and key-presses work exactly as they do on the CLI.

Usage
-----
Run as a script::

    python -m indi.scripts.gui

or (after installation) via the console entry point::

    indi-gui
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, font, messagebox, ttk
from typing import Callable

# ── Matplotlib must be configured before any pyplot import ─────────────────
import matplotlib

matplotlib.use("TkAgg")  # share the tkinter event loop


# ── Constants ──────────────────────────────────────────────────────────────
_POLL_INTERVAL_MS = 40  # GUI polling interval (ms)
_LOG_COLOURS = {
    "ERROR": "#f14c4c",
    "WARNING": "#cca700",
    "INFO": "#4ec9b0",
    "DEBUG": "#888888",
}

_STEP_LABELS: tuple[str, ...] = (
    "Initial setup",
    "Read data",
    "Phase correction",
    "Image denoising",
    "Image registration",
    "Manual outliers pre",
    "Average images",
    "Heart segmentation",
    "Remove slices",
    "Crop FOV",
    "Manual outliers post",
    "Remove outliers",
    "Record image registration",
    "SNR maps",
    "Complex averaging",
    "Tensor fitting",
    "Uformer denoise",
    "Eigensystem",
    "FA/MD/maps",
    "Cardiac coordinates",
    "LV segments",
    "Tensor orientation maps",
    "HA line profiles",
    "Export results",
    "Cleanup",
)


# ── Logging helper ─────────────────────────────────────────────────────────
class _QueueHandler(logging.Handler):
    """Logging handler that forwards records to a :class:`queue.Queue` and optionally triggers a UI flush."""

    def __init__(
        self,
        log_queue: queue.Queue[logging.LogRecord],
        flush_ui: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._queue = log_queue
        self._flush_ui = flush_ui

    def emit(self, record: logging.LogRecord) -> None:
        self._queue.put(record)
        if self._flush_ui:
            self._flush_ui()


# ── Main application ───────────────────────────────────────────────────────
class INDIApp(tk.Tk):
    """Top-level tkinter window for the INDI processing GUI."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()
        self.title("INDI – Cardiac Diffusion Tensor Processing")
        self.resizable(True, True)
        self.minsize(720, 520)

        self._log_queue: queue.Queue[logging.LogRecord] = queue.Queue()
        self._dialog_queue: queue.Queue[tuple] = queue.Queue()

        self._pipeline_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._step_total = 1
        self._step_progress = 0

        # result storage for cross-thread dialogues
        self._dialog_result: bool | None = None
        self._dialog_event = threading.Event()

        self._build_ui()
        self._poll()  # start the recurring polling loop

    # ------------------------------------------------------------------
    # Thread helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _on_ui_thread() -> bool:
        return threading.current_thread() is threading.main_thread()

    def _run_on_ui(self, func: Callable[[], None]) -> None:
        """Run *func* immediately when already on the UI thread, otherwise schedule via ``after``."""

        if self._on_ui_thread():
            func()
        else:
            self.after(0, func)

    def _advance_progress(self, label: str, steps: int = 1) -> None:
        """Advance the determinate progress bar and update the status label."""

        self._step_progress += steps
        current = self._step_progress
        total = max(self._step_total, 1)

        self._run_on_ui(
            lambda c=current, t=total, lbl=label: (
                self._progress.configure(value=min(c, t)),
                self._status_var.set(f"{lbl} ({c}/{t})"),
            )
        )

        # When running on the main thread, pump events so the UI paints immediately
        self._pump_events()

    def _pump_events(self) -> None:
        """Process pending Tk events and flush logs when running long tasks on the UI thread."""

        if not self._on_ui_thread():
            return
        try:
            self._flush_log_queue()
            self.update_idletasks()
            self.update()
        except tk.TclError:
            # The window may have been closed while processing
            pass

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        PAD = 8

        # ── Top: configuration panel ──────────────────────────────────
        top = ttk.LabelFrame(self, text="Configuration", padding=PAD)
        top.pack(fill="x", padx=PAD, pady=(PAD, 0))
        top.columnconfigure(1, weight=1)

        # YAML file row
        ttk.Label(top, text="YAML settings file:").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=3)
        self._yaml_var = tk.StringVar()
        ttk.Entry(top, textvariable=self._yaml_var).grid(row=0, column=1, sticky="ew", padx=4, pady=3)
        ttk.Button(top, text="Browse…", command=self._browse_yaml).grid(
            row=0, column=2, sticky="w", padx=(4, 0), pady=3
        )

        # Data folder row
        ttk.Label(top, text="Data folder:").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=3)
        self._data_var = tk.StringVar()
        ttk.Entry(top, textvariable=self._data_var).grid(row=1, column=1, sticky="ew", padx=4, pady=3)
        ttk.Button(top, text="Browse…", command=self._browse_data).grid(
            row=1, column=2, sticky="w", padx=(4, 0), pady=3
        )

        # Run / Stop buttons
        btn_frame = ttk.Frame(top)
        btn_frame.grid(row=2, column=0, columnspan=3, sticky="e", pady=(6, 0))

        self._run_btn = ttk.Button(btn_frame, text="▶  Run", command=self._on_run, width=14)
        self._run_btn.pack(side="left", padx=(0, 6))

        self._stop_btn = ttk.Button(
            btn_frame,
            text="■  Stop",
            command=self._on_stop,
            width=14,
            state="disabled",
        )
        self._stop_btn.pack(side="left")

        # ── Middle: log output panel ──────────────────────────────────
        log_frame = ttk.LabelFrame(self, text="Log output", padding=PAD)
        log_frame.pack(fill="both", expand=True, padx=PAD, pady=(PAD, 0))

        mono = font.Font(family="Courier", size=10)
        self._log_text = tk.Text(
            log_frame,
            state="disabled",
            wrap="word",
            font=mono,
            background="#1e1e1e",
            foreground="#d4d4d4",
            insertbackground="white",
        )
        self._log_text.pack(side="left", fill="both", expand=True)

        log_scroll = ttk.Scrollbar(log_frame, command=self._log_text.yview)
        log_scroll.pack(side="right", fill="y")
        self._log_text.configure(yscrollcommand=log_scroll.set)

        # Colour tags for log levels
        for level, colour in _LOG_COLOURS.items():
            self._log_text.tag_configure(level, foreground=colour)

        # ── Bottom: progress bar ──────────────────────────────────────
        bot = ttk.Frame(self, padding=(PAD, PAD // 2))
        bot.pack(fill="x", padx=PAD, pady=(PAD // 2, PAD))
        bot.columnconfigure(1, weight=1)

        ttk.Label(bot, text="Progress:").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self._progress = ttk.Progressbar(bot, mode="determinate", maximum=100)
        self._progress.grid(row=0, column=1, sticky="ew")

        self._status_var = tk.StringVar(value="Idle")
        ttk.Label(bot, textvariable=self._status_var, width=20, anchor="e").grid(
            row=0, column=2, sticky="e", padx=(6, 0)
        )

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------

    def _browse_yaml(self) -> None:
        path = filedialog.askopenfilename(
            title="Select YAML settings file",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if path:
            self._yaml_var.set(path)

    def _browse_data(self) -> None:
        path = filedialog.askdirectory(title="Select data root folder")
        if path:
            self._data_var.set(path)

    # ------------------------------------------------------------------
    # Run / Stop
    # ------------------------------------------------------------------

    def _on_run(self) -> None:
        yaml_path = self._yaml_var.get().strip()
        data_folder = self._data_var.get().strip()

        if not yaml_path:
            self._append_log("ERROR", "Please select a YAML settings file.")
            return
        if not os.path.isfile(yaml_path):
            self._append_log("ERROR", f"YAML file not found: {yaml_path}")
            return
        if not data_folder:
            self._append_log("ERROR", "Please select a data folder.")
            return
        if not os.path.isdir(data_folder):
            self._append_log("ERROR", f"Data folder not found: {data_folder}")
            return

        self._stop_event.clear()
        self._run_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._progress.configure(mode="indeterminate", value=0)
        self._progress.start(10)
        self._status_var.set("Starting…")

        start_on_main_thread = sys.platform == "darwin"

        if start_on_main_thread:
            # macOS requires all NSWindow creation (Matplotlib/Tk) to happen on the main thread
            self._append_log(
                "INFO",
                "macOS detected – running pipeline on the main thread so Matplotlib windows can open.",
            )
            self._pipeline_thread = None
            self.after(10, self._pipeline_worker_main_thread, yaml_path, data_folder)
        else:
            self._pipeline_thread = threading.Thread(
                target=self._pipeline_worker,
                args=(yaml_path, data_folder),
                daemon=True,
            )
            self._pipeline_thread.start()

    def _on_stop(self) -> None:
        self._stop_event.set()
        self._append_log("WARNING", "Stop requested – finishing current step…")
        self._stop_btn.configure(state="disabled")

    def _pipeline_done(self, error: Exception | None = None) -> None:
        """Reset controls after the pipeline thread finishes (main thread)."""
        self._progress.stop()
        self._progress.configure(mode="determinate", value=self._step_total if error is None else 0)
        self._run_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        if error is None:
            self._status_var.set("Finished ✓")
        else:
            self._status_var.set("Error – see log")
            messagebox.showerror(
                "Pipeline error",
                f"An error occurred:\n\n{error}",
            )

    # ------------------------------------------------------------------
    # Cross-thread yes/no dialogue
    # ------------------------------------------------------------------

    def _ask_yes_no_in_main_thread(self, question: str) -> None:
        """Show a yes/no dialogue in the main thread and store result."""
        self._dialog_result = messagebox.askyesno("Confirmation", question)
        self._dialog_event.set()

    def ask_yes_no(self, question: str) -> bool:
        """Thread-safe yes/no dialogue.

        May be called from the pipeline worker thread.  Blocks until the
        user answers.
        """
        self._dialog_event.clear()
        self._dialog_result = None
        self.after(0, self._ask_yes_no_in_main_thread, question)
        self._dialog_event.wait()
        return bool(self._dialog_result)

    # ------------------------------------------------------------------
    # Pipeline worker (background thread)
    # ------------------------------------------------------------------

    def _pipeline_worker(self, yaml_path: str, data_folder: str) -> None:
        error: Exception | None = None
        try:
            self._run_pipeline(yaml_path, data_folder)
        except Exception as exc:
            error = exc
        finally:
            self._run_on_ui(lambda: self._pipeline_done(error))

    def _pipeline_worker_main_thread(self, yaml_path: str, data_folder: str) -> None:
        """Run the pipeline without a worker thread (needed for macOS Matplotlib/Tk windows)."""

        error: Exception | None = None
        try:
            self._run_pipeline(yaml_path, data_folder)
        except Exception as exc:
            error = exc
        self._pipeline_done(error)

    def _run_pipeline(self, yaml_path: str, data_folder: str) -> None:
        """Mirror of ``main.main()`` adapted for GUI execution."""

        import pyautogui

        matplotlib.rcParams["font.size"] = 10
        matplotlib.rcParams["toolbar"] = "None"
        os.environ["TF_USE_LEGACY_KERAS"] = "1"

        # Lazy imports – keep them here so they only load when Run is pressed
        from indi.extensions.complex_averaging import complex_averaging
        from indi.extensions.crop_fov import crop_fov, record_image_registration
        from indi.extensions.extensions import (
            export_results,
            get_cardiac_coordinates_short_axis,
            get_colourmaps,
            get_lv_segments,
            get_snr_maps,
            remove_outliers,
            remove_slices,
        )
        from indi.extensions.folder_loop_initial_setup import folder_loop_initial_setup
        from indi.extensions.get_eigensystem import get_eigensystem
        from indi.extensions.get_fa_md import get_fa_md
        from indi.extensions.get_tensor_orientation_maps import get_tensor_orientation_maps
        from indi.extensions.ha_line_profiles import get_ha_line_profiles_and_distance_maps
        from indi.extensions.heart_segmentation import get_average_images, heart_segmentation
        from indi.extensions.image_denoising import image_denoising
        from indi.extensions.image_registration import image_registration
        from indi.extensions.initial_setup import initial_setup_gui
        from indi.extensions.phase_correction_for_complex_averaging import (
            phase_correction_for_complex_averaging,
        )
        from indi.extensions.read_data.read_and_pre_process_data import read_data
        from indi.extensions.select_outliers import select_outliers
        from indi.extensions.tensor_fittings import dipy_tensor_fit

        # Attach the queue handler so log records appear in the GUI panel and flush immediately
        queue_handler = _QueueHandler(self._log_queue, flush_ui=lambda: self._run_on_ui(self._flush_log_queue))

        root_logger = logging.getLogger()
        if not any(isinstance(h, _QueueHandler) for h in root_logger.handlers):
            root_logger.addHandler(queue_handler)

        colormaps = get_colourmaps()

        dti, settings, logger, log_format, all_to_be_analysed_folders = initial_setup_gui(
            yaml_path, data_folder, extra_handlers=[queue_handler]
        )

        settings["screen_size"] = pyautogui.size()
        n_folders = len(all_to_be_analysed_folders)
        steps_per_folder = len(_STEP_LABELS)

        self._step_total = max(1, n_folders * steps_per_folder)
        self._step_progress = 0

        # Switch progress bar to determinate now we know the total step count
        self._run_on_ui(
            lambda: (
                self._progress.stop(),
                self._progress.configure(mode="determinate", maximum=self._step_total, value=0),
                self._status_var.set(f"Starting… (0/{self._step_total})"),
            )
        )

        # Anonymisation confirmation via GUI dialogue instead of stdin
        if settings["workflow_mode"] == "anon":
            answer = self.ask_yes_no("Archive all DICOM files into an encrypted 7z?")
            if answer:
                logger.info("Archiving DICOMs in an encrypted 7z file!")
            else:
                logger.error("Exiting – no permission to archive DICOM data.")
                return

        failed_folders: list[str] = []

        for folder_idx, current_folder in enumerate(all_to_be_analysed_folders, start=1):
            if self._stop_event.is_set():
                logger.warning("Pipeline stopped by user.")
                break

            step_idx = 0

            def advance(label: str, steps: int = 1) -> None:
                nonlocal step_idx
                step_idx += steps
                self._advance_progress(label, steps)

            self._pump_events()

            try:
                info, settings, logger = folder_loop_initial_setup(current_folder, settings, logger, log_format)
                advance(_STEP_LABELS[0])

                [data, info, slices] = read_data(settings, info, logger)
                advance(_STEP_LABELS[1])

                if settings["workflow_mode"] == "anon":
                    logger.info("Anonymisation-only mode. Stopping here.")
                    advance("Anonymisation-only stop", steps=len(_STEP_LABELS) - step_idx)
                    continue

                if settings["complex_data"]:
                    data = phase_correction_for_complex_averaging(data, logger, settings)
                    advance(_STEP_LABELS[2])
                else:
                    advance(f"{_STEP_LABELS[2]} (skipped)")

                if settings["image_denoising"]:
                    data = image_denoising(data, logger, settings)
                    advance(_STEP_LABELS[3])
                else:
                    advance(f"{_STEP_LABELS[3]} (skipped)")

                data, registration_image_data, ref_images, reg_mask = image_registration(
                    data, slices, info, settings, logger
                )
                advance(_STEP_LABELS[4])

                if settings["workflow_mode"] == "reg":
                    logger.info("Registration-only mode. Stopping here.")
                    advance("Registration-only stop", steps=len(_STEP_LABELS) - step_idx)
                    continue

                if settings["remove_outliers_manually_pre"]:
                    logger.info("Manual removal of outliers – pre segmentation")
                    [data, info, slices] = select_outliers(
                        data,
                        slices,
                        registration_image_data,
                        settings,
                        info,
                        logger,
                        stage="pre",
                        segmentation={},
                        mask=reg_mask,
                        prelim_residuals={},
                    )
                    advance(_STEP_LABELS[5])
                else:
                    logger.info("Manual removal of outliers pre segmentation is False")
                    info["rejected_indices"] = []
                    info["n_images_rejected"] = 0
                    advance(f"{_STEP_LABELS[5]} (skipped)")

                average_images = get_average_images(data, slices, info, logger)
                advance(_STEP_LABELS[6])

                segmentation, mask_3c, prelim_residuals = heart_segmentation(
                    data, average_images, slices, info["n_slices"], colormaps, settings, info, logger
                )
                advance(_STEP_LABELS[7])

                data, slices, segmentation = remove_slices(data, slices, segmentation, logger)
                advance(_STEP_LABELS[8])

                dti, data, mask_3c, reg_mask, segmentation, average_images, info, crop_mask = crop_fov(
                    dti,
                    data,
                    mask_3c,
                    reg_mask,
                    segmentation,
                    slices,
                    average_images,
                    registration_image_data,
                    ref_images,
                    info,
                    logger,
                    settings,
                )
                advance(_STEP_LABELS[9])

                logger.info("Manual removal of outliers – post segmentation")
                [data, info, slices] = select_outliers(
                    data,
                    slices,
                    registration_image_data,
                    settings,
                    info,
                    logger,
                    stage="post",
                    segmentation=segmentation,
                    mask=reg_mask,
                    prelim_residuals=prelim_residuals,
                )
                advance(_STEP_LABELS[10])

                data, info = remove_outliers(data, info, settings)
                advance(_STEP_LABELS[11])

                record_image_registration(registration_image_data, ref_images, mask_3c, slices, settings, logger)
                advance(_STEP_LABELS[12])

                [dti["snr"], noise, snr_b0_lv, info] = get_snr_maps(
                    data, mask_3c, average_images, slices, settings, logger, info
                )
                advance(_STEP_LABELS[13])

                if settings["complex_data"]:
                    data = complex_averaging(data, logger)
                    advance(_STEP_LABELS[14])
                else:
                    advance(f"{_STEP_LABELS[14]} (skipped)")

                (
                    dti["tensor"],
                    dti["s0"],
                    dti["residuals_plot"],
                    dti["residuals_map"],
                    _,
                    info,
                ) = dipy_tensor_fit(
                    slices,
                    data,
                    info,
                    settings,
                    mask_3c,
                    average_images,
                    logger,
                    method=settings["tensor_fit_method"],
                    quick_mode=False,
                )
                advance(_STEP_LABELS[15])

                if settings["uformer_denoise"]:
                    try:
                        from indi.extensions.uformer_denoising import denoise_tensor
                    except ImportError:
                        logger.error("Could not import uformer_denoising module")
                        raise ImportError("uformer_denoising not available – please install torch.")
                    logger.info(
                        "Denoising tensor with uformer: breatholds %s",
                        settings["uformer_breatholds"],
                    )
                    dti["tensor"] = denoise_tensor(dti["tensor"], settings)
                    advance(_STEP_LABELS[16])
                else:
                    logger.info("Uformer tensor denoising is disabled")
                    advance(f"{_STEP_LABELS[16]} (skipped)")

                dti, info = get_eigensystem(dti, slices, info, average_images, settings, mask_3c, logger)
                advance(_STEP_LABELS[17])

                dti["md"], dti["fa"], dti["mode"], dti["frob_norm"], dti["mag_anisotropy"], info = get_fa_md(
                    dti["eigenvalues"], info, mask_3c, slices, logger
                )
                advance(_STEP_LABELS[18])

                local_cardiac_coordinates, lv_centres, phi_matrix = get_cardiac_coordinates_short_axis(
                    mask_3c, segmentation, slices, info["n_slices"], settings, dti, average_images, info
                )
                advance(_STEP_LABELS[19])

                dti["lv_sectors"] = get_lv_segments(segmentation, phi_matrix, mask_3c, lv_centres, slices, logger)
                advance(_STEP_LABELS[20])

                dti["ha"], dti["ta"], dti["e2a"], info = get_tensor_orientation_maps(
                    slices, mask_3c, local_cardiac_coordinates, dti, settings, info, logger
                )
                advance(_STEP_LABELS[21])

                (
                    dti["ha_line_profiles"],
                    dti["wall_thickness"],
                    dti["bullseye"],
                    dti["distance_endo"],
                    dti["distance_epi"],
                    dti["distance_transmural"],
                    dti["ha_line_profiles_2"],
                ) = get_ha_line_profiles_and_distance_maps(
                    dti["ha"],
                    lv_centres,
                    slices,
                    mask_3c,
                    segmentation,
                    settings,
                    info,
                    average_images,
                    logger,
                )
                advance(_STEP_LABELS[22])

                export_results(
                    data,
                    dti,
                    info,
                    settings,
                    mask_3c,
                    slices,
                    average_images,
                    segmentation,
                    colormaps,
                    logger,
                )
                advance(_STEP_LABELS[23])

                logger.info("Cleaning up before the next folder")
                del (
                    average_images,
                    crop_mask,
                    data,
                    info,
                    local_cardiac_coordinates,
                    lv_centres,
                    mask_3c,
                    noise,
                    phi_matrix,
                    ref_images,
                    registration_image_data,
                    segmentation,
                    slices,
                    snr_b0_lv,
                )
                dti = {}
                advance(_STEP_LABELS[24])

                logger.info("=" * 60)
                logger.info("FINISHED folder %d / %d", folder_idx, n_folders)
                logger.info("=" * 60)

                self._pump_events()

            except Exception as exc:
                remaining = len(_STEP_LABELS) - step_idx
                if remaining > 0:
                    advance("Error – fast-forward", steps=remaining)
                failed_folders.append(os.path.dirname(current_folder))
                logger.error("Error in folder: %s", os.path.dirname(current_folder))
                logger.error(exc)
                logger.error("=" * 60)
                logger.error("ERROR – see above")
                logger.error("=" * 60)
                if n_folders == 1:
                    raise

        logger.info("=" * 60)
        logger.info("All folders processed")
        logger.info("=" * 60)

        if failed_folders:
            logger.warning("Failed folders:")
            for f in failed_folders:
                logger.warning("  %s", f)

    # ------------------------------------------------------------------
    # Log panel helpers
    # ------------------------------------------------------------------

    def _append_log(self, level: str, message: str) -> None:
        """Insert *message* into the log Text widget with the appropriate colour tag."""
        tag = level if level in _LOG_COLOURS else "INFO"
        self._log_text.configure(state="normal")
        self._log_text.insert("end", message + "\n", tag)
        self._log_text.configure(state="disabled")
        self._log_text.see("end")

    def _flush_log_queue(self) -> None:
        """Drain the logging queue and display all pending records."""
        _formatter = logging.Formatter("%(levelname)s : %(asctime)s :: %(message)s")
        try:
            while True:
                record = self._log_queue.get_nowait()
                text = _formatter.format(record)
                level = record.levelname if record.levelname in _LOG_COLOURS else "INFO"
                self._append_log(level, text)
        except queue.Empty:
            pass

    # ------------------------------------------------------------------
    # Polling loop (runs in main thread via after())
    # ------------------------------------------------------------------

    def _poll(self) -> None:
        self._flush_log_queue()
        self.after(_POLL_INTERVAL_MS, self._poll)


# ── Entry point ────────────────────────────────────────────────────────────


def main() -> None:
    """Launch the INDI GUI application."""
    app = INDIApp()
    app.mainloop()


if __name__ == "__main__":
    main()
