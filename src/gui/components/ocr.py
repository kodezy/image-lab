import tkinter as tk
from tkinter import ttk
from typing import Any

from src.gui.utils import create_button, create_combobox, create_labeled_frame, create_slider, create_spinbox


class OCRPanel:
    """Panel for OCR configuration and results"""

    def __init__(self, parent: tk.Widget, app: Any) -> None:
        self.parent = parent
        self.app = app

        self._setup_variables()
        self._create_frame()

    def display_results(self, result) -> None:
        """Display OCR results"""
        self.result_text.delete(1.0, tk.END)

        if result and result[0] and "rec_texts" in result[0]:
            texts = result[0]["rec_texts"]
            confidence_scores = result[0].get("rec_scores", [])

            formatted_results = []

            for i, text in enumerate(texts):
                if i < len(confidence_scores):
                    confidence = confidence_scores[i] * 100
                    formatted_results.append(f"{text} ({confidence:.1f}%)")
                else:
                    formatted_results.append(text)

            self.result_text.insert(tk.END, "\n".join(formatted_results))

        else:
            self.result_text.insert(tk.END, "No text detected")

    def refresh(self) -> None:
        """Refresh panel with current configuration"""
        config = self.app.ocr_config

        self.lang_var.set(config.lang)
        self.device_var.set(config.device)
        self.version_var.set(config.ocr_version)

        self.det_thresh_var.set(config.text_det_thresh)
        self.box_thresh_var.set(config.text_det_box_thresh)
        self.score_thresh_var.set(config.text_rec_score_thresh)
        self.unclip_ratio_var.set(config.text_det_unclip_ratio)

        self.batch_size_var.set(config.text_recognition_batch_size)
        self.cpu_threads_var.set(config.cpu_threads)

    def _setup_variables(self) -> None:
        """Setup tkinter variables"""
        config = self.app.ocr_config

        self.lang_var = tk.StringVar(value=config.lang)
        self.device_var = tk.StringVar(value=config.device)
        self.version_var = tk.StringVar(value=config.ocr_version)

        self.det_thresh_var = tk.DoubleVar(value=config.text_det_thresh)
        self.box_thresh_var = tk.DoubleVar(value=config.text_det_box_thresh)
        self.score_thresh_var = tk.DoubleVar(value=config.text_rec_score_thresh)
        self.unclip_ratio_var = tk.DoubleVar(value=config.text_det_unclip_ratio)

        self.batch_size_var = tk.IntVar(value=config.text_recognition_batch_size)
        self.cpu_threads_var = tk.IntVar(value=config.cpu_threads)

    def _create_frame(self) -> None:
        """Create OCR frame"""
        self.frame = ttk.Frame(self.parent)

        self._create_settings_section()
        self._create_action_section()
        self._create_results_section()

    def _create_action_section(self) -> None:
        """Create OCR action section"""
        action_frame = create_labeled_frame(self.frame, "🎯 Actions")
        action_frame.pack(fill=tk.X, pady=5, padx=5)

        button_frame = ttk.Frame(action_frame)
        button_frame.pack(fill=tk.X, pady=5)

        self.ocr_button = create_button(button_frame, "🔍 Run OCR", self.app.run_ocr)
        self.ocr_button.pack(fill=tk.X, pady=2)

    def _create_settings_section(self) -> None:
        """Create OCR settings section"""
        settings_frame = create_labeled_frame(self.frame, "⚙️ Settings")
        settings_frame.pack(fill=tk.X, pady=5, padx=5)

        lang_frame, self.lang_combobox = create_combobox(
            settings_frame,
            "Language",
            self.lang_var,
            ["ch", "en", "pt"],
            self._on_lang_changed,
        )
        lang_frame.pack(fill=tk.X, pady=2)

        device_frame, self.device_combobox = create_combobox(
            settings_frame,
            "Device",
            self.device_var,
            ["cpu", "gpu:0"],
            self._on_device_changed,
        )
        device_frame.pack(fill=tk.X, pady=2)

        version_frame, self.version_combobox = create_combobox(
            settings_frame,
            "OCR Version",
            self.version_var,
            ["PP-OCRv5", "PP-OCRv4", "PP-OCRv3"],
            self._on_version_changed,
        )
        version_frame.pack(fill=tk.X, pady=2)

        ttk.Separator(settings_frame, orient="horizontal").pack(fill=tk.X, pady=10)

        det_thresh_frame, _, _ = create_slider(
            settings_frame,
            "Detection Threshold",
            self.det_thresh_var,
            0.1,
            0.9,
            self._on_det_thresh_changed,
        )
        det_thresh_frame.pack(fill=tk.X, pady=2)

        box_thresh_frame, _, _ = create_slider(
            settings_frame,
            "Box Threshold",
            self.box_thresh_var,
            0.1,
            0.9,
            self._on_box_thresh_changed,
        )
        box_thresh_frame.pack(fill=tk.X, pady=2)

        score_thresh_frame, _, _ = create_slider(
            settings_frame,
            "Score Threshold",
            self.score_thresh_var,
            0.0,
            1.0,
            self._on_score_thresh_changed,
        )
        score_thresh_frame.pack(fill=tk.X, pady=2)

        unclip_frame, _, _ = create_slider(
            settings_frame,
            "Unclip Ratio",
            self.unclip_ratio_var,
            1.0,
            3.0,
            self._on_unclip_ratio_changed,
        )
        unclip_frame.pack(fill=tk.X, pady=2)

        batch_size_frame, self.batch_spinbox = create_spinbox(
            settings_frame,
            "Batch Size",
            self.batch_size_var,
            1,
            32,
            self._on_batch_size_changed,
        )
        batch_size_frame.pack(fill=tk.X, pady=2)

        cpu_threads_frame, self.cpu_threads_spinbox = create_spinbox(
            settings_frame,
            "CPU Threads",
            self.cpu_threads_var,
            1,
            16,
            self._on_cpu_threads_changed,
        )
        cpu_threads_frame.pack(fill=tk.X, pady=2)

    def _on_lang_changed(self) -> None:
        """Handle language change"""
        self.app.ocr_config.lang = self.lang_var.get()
        self._invalidate_ocr_instance()

    def _on_device_changed(self) -> None:
        """Handle device change"""
        self.app.ocr_config.device = self.device_var.get()
        self._invalidate_ocr_instance()

    def _on_version_changed(self) -> None:
        """Handle version change"""
        self.app.ocr_config.ocr_version = self.version_var.get()
        self._invalidate_ocr_instance()

    def _on_det_thresh_changed(self, value) -> None:
        """Handle detection threshold change"""
        self.app.ocr_config.text_det_thresh = round(float(value), 3)
        self._invalidate_ocr_instance()

    def _on_box_thresh_changed(self, value) -> None:
        """Handle box threshold change"""
        self.app.ocr_config.text_det_box_thresh = round(float(value), 3)
        self._invalidate_ocr_instance()

    def _on_score_thresh_changed(self, value) -> None:
        """Handle score threshold change"""
        self.app.ocr_config.text_rec_score_thresh = round(float(value), 3)
        self._invalidate_ocr_instance()

    def _on_unclip_ratio_changed(self, value) -> None:
        """Handle unclip ratio change"""
        self.app.ocr_config.text_det_unclip_ratio = round(float(value), 3)
        self._invalidate_ocr_instance()

    def _on_batch_size_changed(self) -> None:
        """Handle batch size change"""
        self.app.ocr_config.text_recognition_batch_size = self.batch_size_var.get()
        self._invalidate_ocr_instance()

    def _on_cpu_threads_changed(self) -> None:
        """Handle CPU threads change"""
        self.app.ocr_config.cpu_threads = self.cpu_threads_var.get()
        self._invalidate_ocr_instance()

    def _invalidate_ocr_instance(self) -> None:
        """Invalidate current OCR instance to force recreation"""
        self.app.ocr_instance = None

    def _create_results_section(self) -> None:
        """Create OCR results section"""
        results_frame = create_labeled_frame(self.frame, "📄 Results")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)

        text_frame = ttk.Frame(results_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.result_text = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 10), height=8)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)

        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
