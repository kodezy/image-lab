import tkinter as tk
from tkinter import ttk
from typing import Any

from src.gui.utils import (
    create_button,
    create_combobox,
    create_labeled_frame,
    create_scrollable_frame,
    create_slider,
    create_spinbox,
)


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
        ocr_config = self.app.ocr_config

        self.ocr_type_var.set(ocr_config.ocr_type)

        if ocr_config.ocr_type == "paddleocr":
            paddle_config = ocr_config.paddleocr_config

            self.paddle_lang_var.set(paddle_config.lang)
            self.device_var.set(paddle_config.device)
            self.version_var.set(paddle_config.ocr_version)

            self.det_thresh_var.set(paddle_config.text_det_thresh)
            self.box_thresh_var.set(paddle_config.text_det_box_thresh)
            self.score_thresh_var.set(paddle_config.text_rec_score_thresh)
            self.unclip_ratio_var.set(paddle_config.text_det_unclip_ratio)

            self.batch_size_var.set(paddle_config.text_recognition_batch_size)
            self.cpu_threads_var.set(paddle_config.cpu_threads)

            self.enable_hpi_var.set(paddle_config.enable_hpi)
            self.mkldnn_cache_var.set(paddle_config.mkldnn_cache_capacity)
            self.precision_var.set(paddle_config.precision)
            self.det_limit_side_len_var.set(paddle_config.text_det_limit_side_len)
            self.det_limit_type_var.set(paddle_config.text_det_limit_type)
            self.use_tensorrt_var.set(paddle_config.use_tensorrt)
            self.enable_mkldnn_var.set(paddle_config.enable_mkldnn)
            self.textline_orientation_batch_var.set(paddle_config.textline_orientation_batch_size)

        else:
            tesseract_config = ocr_config.tesseract_config

            self.tesseract_lang_var.set(tesseract_config.lang)
            self.psm_var.set(tesseract_config.psm)
            self.oem_var.set(tesseract_config.oem)
            self.tesseract_config_var.set(tesseract_config.config)

        self._update_visible_settings()

    def _setup_variables(self) -> None:
        """Setup tkinter variables"""
        ocr_config = self.app.ocr_config

        self.ocr_type_var = tk.StringVar(value=ocr_config.ocr_type)

        paddle_config = ocr_config.paddleocr_config
        self.paddle_lang_var = tk.StringVar(value=paddle_config.lang)
        self.device_var = tk.StringVar(value=paddle_config.device)
        self.version_var = tk.StringVar(value=paddle_config.ocr_version)

        self.det_thresh_var = tk.DoubleVar(value=paddle_config.text_det_thresh)
        self.box_thresh_var = tk.DoubleVar(value=paddle_config.text_det_box_thresh)
        self.score_thresh_var = tk.DoubleVar(value=paddle_config.text_rec_score_thresh)
        self.unclip_ratio_var = tk.DoubleVar(value=paddle_config.text_det_unclip_ratio)

        self.batch_size_var = tk.IntVar(value=paddle_config.text_recognition_batch_size)
        self.cpu_threads_var = tk.IntVar(value=paddle_config.cpu_threads)

        self.enable_hpi_var = tk.BooleanVar(value=paddle_config.enable_hpi)
        self.mkldnn_cache_var = tk.IntVar(value=paddle_config.mkldnn_cache_capacity)
        self.precision_var = tk.StringVar(value=paddle_config.precision)
        self.det_limit_side_len_var = tk.IntVar(value=paddle_config.text_det_limit_side_len)
        self.det_limit_type_var = tk.StringVar(value=paddle_config.text_det_limit_type)
        self.use_tensorrt_var = tk.BooleanVar(value=paddle_config.use_tensorrt)
        self.enable_mkldnn_var = tk.BooleanVar(value=paddle_config.enable_mkldnn)
        self.textline_orientation_batch_var = tk.IntVar(value=paddle_config.textline_orientation_batch_size)

        tesseract_config = ocr_config.tesseract_config
        self.tesseract_lang_var = tk.StringVar(value=tesseract_config.lang)
        self.psm_var = tk.IntVar(value=tesseract_config.psm)
        self.oem_var = tk.IntVar(value=tesseract_config.oem)
        self.tesseract_config_var = tk.StringVar(value=tesseract_config.config)

    def _create_frame(self) -> None:
        """Create OCR frame"""
        self.frame = ttk.Frame(self.parent)

        self._create_settings_section()
        self._create_action_section()
        self._create_results_section()

    def _create_action_section(self) -> None:
        """Create OCR action section"""
        action_frame = create_labeled_frame(self.frame, "🎯 Actions")
        action_frame.pack(fill=tk.X, pady=(2, 5), padx=5)

        button_frame = ttk.Frame(action_frame)
        button_frame.pack(fill=tk.X, pady=5)

        self.ocr_button = create_button(button_frame, "🔍 Run OCR", self.app.run_ocr)
        self.ocr_button.pack(fill=tk.X, pady=2)

    def _create_settings_section(self) -> None:
        """Create OCR settings section"""
        settings_frame = create_labeled_frame(self.frame, "⚙️ Settings")
        settings_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 2), padx=5)

        self.settings_canvas, scrollable_frame, _ = create_scrollable_frame(settings_frame)
        self.settings_canvas.pack(fill=tk.BOTH, expand=True)
        self.scrollable_frame = scrollable_frame

        canvas_items = self.settings_canvas.find_all()
        self.canvas_window_id: int | None = canvas_items[0] if canvas_items else None

        ocr_type_frame, self.ocr_type_combobox = create_combobox(
            scrollable_frame,
            "OCR Engine",
            self.ocr_type_var,
            ["paddleocr", "tesseract"],
            self._on_ocr_type_changed,
        )
        ocr_type_frame.pack(fill=tk.X, pady=(0, 5))

        self.paddleocr_frame = ttk.Frame(scrollable_frame)
        self._create_paddleocr_settings(self.paddleocr_frame)

        self.tesseract_frame = ttk.Frame(scrollable_frame)
        self._create_tesseract_settings(self.tesseract_frame)

        self._update_visible_settings()

    def _create_paddleocr_settings(self, parent: ttk.Frame) -> None:
        """Create PaddleOCR specific settings"""
        lang_frame, self.lang_combobox = create_combobox(
            parent,
            "Language",
            self.paddle_lang_var,
            ["ch", "en", "pt"],
            self._on_paddle_lang_changed,
        )
        lang_frame.pack(fill=tk.X, pady=(0, 5))

        device_frame, self.device_combobox = create_combobox(
            parent,
            "Device",
            self.device_var,
            ["cpu", "gpu:0", "npu:0", "xpu:0", "mlu:0", "dcu:0"],
            self._on_device_changed,
        )
        device_frame.pack(fill=tk.X, pady=(0, 5))

        version_frame, self.version_combobox = create_combobox(
            parent,
            "OCR Version",
            self.version_var,
            ["PP-OCRv5", "PP-OCRv4", "PP-OCRv3"],
            self._on_version_changed,
        )
        version_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=(5, 10))

        det_thresh_frame, _, _ = create_slider(
            parent,
            "Detection Threshold",
            self.det_thresh_var,
            0.1,
            0.9,
            self._on_det_thresh_changed,
        )
        det_thresh_frame.pack(fill=tk.X, pady=(0, 5))

        box_thresh_frame, _, _ = create_slider(
            parent,
            "Box Threshold",
            self.box_thresh_var,
            0.1,
            0.9,
            self._on_box_thresh_changed,
        )
        box_thresh_frame.pack(fill=tk.X, pady=(0, 5))

        score_thresh_frame, _, _ = create_slider(
            parent,
            "Score Threshold",
            self.score_thresh_var,
            0.0,
            1.0,
            self._on_score_thresh_changed,
        )
        score_thresh_frame.pack(fill=tk.X, pady=(0, 5))

        unclip_frame, _, _ = create_slider(
            parent,
            "Unclip Ratio",
            self.unclip_ratio_var,
            1.0,
            3.0,
            self._on_unclip_ratio_changed,
        )
        unclip_frame.pack(fill=tk.X, pady=(0, 5))

        det_limit_frame, self.det_limit_spinbox = create_spinbox(
            parent,
            "Detection Limit Side Length",
            self.det_limit_side_len_var,
            64,
            4096,
            self._on_det_limit_side_len_changed,
        )
        det_limit_frame.pack(fill=tk.X, pady=(0, 5))

        det_limit_type_frame, self.det_limit_type_combobox = create_combobox(
            parent,
            "Detection Limit Type",
            self.det_limit_type_var,
            ["min", "max"],
            self._on_det_limit_type_changed,
        )
        det_limit_type_frame.pack(fill=tk.X, pady=(0, 5))

        batch_size_frame, self.batch_spinbox = create_spinbox(
            parent,
            "Batch Size",
            self.batch_size_var,
            1,
            32,
            self._on_batch_size_changed,
        )
        batch_size_frame.pack(fill=tk.X, pady=(0, 5))

        cpu_threads_frame, self.cpu_threads_spinbox = create_spinbox(
            parent,
            "CPU Threads",
            self.cpu_threads_var,
            1,
            16,
            self._on_cpu_threads_changed,
        )
        cpu_threads_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=(5, 10))

        enable_hpi_frame = ttk.Frame(parent)
        enable_hpi_frame.pack(fill=tk.X, pady=(0, 5))
        self.enable_hpi_checkbox = ttk.Checkbutton(
            enable_hpi_frame,
            text="Enable High Performance Inference",
            variable=self.enable_hpi_var,
            command=self._on_enable_hpi_changed,
        )
        self.enable_hpi_checkbox.pack(anchor=tk.W)

        mkldnn_cache_frame, self.mkldnn_cache_spinbox = create_spinbox(
            parent,
            "MKL-DNN Cache Capacity",
            self.mkldnn_cache_var,
            1,
            100,
            self._on_mkldnn_cache_changed,
        )
        mkldnn_cache_frame.pack(fill=tk.X, pady=(0, 5))

        precision_frame, self.precision_combobox = create_combobox(
            parent,
            "Precision",
            self.precision_var,
            ["fp32", "fp16"],
            self._on_precision_changed,
        )
        precision_frame.pack(fill=tk.X, pady=(0, 5))

        use_tensorrt_frame = ttk.Frame(parent)
        use_tensorrt_frame.pack(fill=tk.X, pady=(0, 5))
        self.use_tensorrt_checkbox = ttk.Checkbutton(
            use_tensorrt_frame,
            text="Enable TensorRT Acceleration",
            variable=self.use_tensorrt_var,
            command=self._on_use_tensorrt_changed,
        )
        self.use_tensorrt_checkbox.pack(anchor=tk.W)

        enable_mkldnn_frame = ttk.Frame(parent)
        enable_mkldnn_frame.pack(fill=tk.X, pady=(0, 5))
        self.enable_mkldnn_checkbox = ttk.Checkbutton(
            enable_mkldnn_frame,
            text="Enable MKL-DNN Acceleration",
            variable=self.enable_mkldnn_var,
            command=self._on_enable_mkldnn_changed,
        )
        self.enable_mkldnn_checkbox.pack(anchor=tk.W)

        textline_orientation_batch_frame, self.textline_orientation_batch_spinbox = create_spinbox(
            parent,
            "Textline Orientation Batch Size",
            self.textline_orientation_batch_var,
            1,
            32,
            self._on_textline_orientation_batch_changed,
        )
        textline_orientation_batch_frame.pack(fill=tk.X, pady=(0, 5))

    def _create_tesseract_settings(self, parent: ttk.Frame) -> None:
        """Create Tesseract specific settings"""
        lang_frame, self.tesseract_lang_combobox = create_combobox(
            parent,
            "Language",
            self.tesseract_lang_var,
            ["eng", "por", "chi_sim", "spa", "fra", "deu"],
            self._on_tesseract_lang_changed,
        )
        lang_frame.pack(fill=tk.X, pady=(0, 5))

        psm_frame, self.psm_spinbox = create_spinbox(
            parent,
            "Page Segmentation Mode (PSM)",
            self.psm_var,
            0,
            13,
            self._on_psm_changed,
        )
        psm_frame.pack(fill=tk.X, pady=(0, 5))

        oem_frame, self.oem_spinbox = create_spinbox(
            parent,
            "OCR Engine Mode (OEM)",
            self.oem_var,
            0,
            3,
            self._on_oem_changed,
        )
        oem_frame.pack(fill=tk.X, pady=(0, 5))

        config_frame = ttk.Frame(parent)
        config_frame.pack(fill=tk.X)
        config_label = ttk.Label(config_frame, text="Config String:")
        config_label.pack(side=tk.LEFT)

        self.tesseract_config_entry = ttk.Entry(config_frame, textvariable=self.tesseract_config_var, width=20)
        self.tesseract_config_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        self.tesseract_config_entry.bind("<KeyRelease>", self._on_tesseract_config_changed)

    def _update_visible_settings(self) -> None:
        """Update visible settings based on OCR type"""
        ocr_type = self.ocr_type_var.get()

        if ocr_type == "paddleocr":
            self.tesseract_frame.pack_forget()
            self.paddleocr_frame.pack(fill=tk.X, anchor=tk.NW)
        else:
            self.paddleocr_frame.pack_forget()
            self.tesseract_frame.pack(fill=tk.X, anchor=tk.NW)

        def update_layout():
            self.scrollable_frame.update_idletasks()
            self.settings_canvas.update_idletasks()

            if self.canvas_window_id:
                canvas_width = self.settings_canvas.winfo_width()
                if canvas_width > 1:
                    self.settings_canvas.itemconfig(self.canvas_window_id, width=canvas_width)

            bbox = self.settings_canvas.bbox("all")
            if bbox:
                self.settings_canvas.configure(scrollregion=bbox)

            self.settings_canvas.yview_moveto(0)

        update_layout()
        self.settings_canvas.after_idle(update_layout)

    def _on_ocr_type_changed(self, value) -> None:
        """Handle OCR type change"""
        self.app.ocr_config.ocr_type = self.ocr_type_var.get()
        self._update_visible_settings()
        self._invalidate_ocr_instance()

    def _on_paddle_lang_changed(self, value) -> None:
        """Handle PaddleOCR language change"""
        self.app.ocr_config.paddleocr_config.lang = self.paddle_lang_var.get()
        self._invalidate_ocr_instance()

    def _on_device_changed(self, value) -> None:
        """Handle device change"""
        self.app.ocr_config.paddleocr_config.device = self.device_var.get()
        self._invalidate_ocr_instance()

    def _on_version_changed(self, value) -> None:
        """Handle version change"""
        self.app.ocr_config.paddleocr_config.ocr_version = self.version_var.get()
        self._invalidate_ocr_instance()

    def _on_det_thresh_changed(self, value) -> None:
        """Handle detection threshold change"""
        self.app.ocr_config.paddleocr_config.text_det_thresh = round(float(value), 3)
        self._invalidate_ocr_instance()

    def _on_box_thresh_changed(self, value) -> None:
        """Handle box threshold change"""
        self.app.ocr_config.paddleocr_config.text_det_box_thresh = round(float(value), 3)
        self._invalidate_ocr_instance()

    def _on_score_thresh_changed(self, value) -> None:
        """Handle score threshold change"""
        self.app.ocr_config.paddleocr_config.text_rec_score_thresh = round(float(value), 3)
        self._invalidate_ocr_instance()

    def _on_unclip_ratio_changed(self, value) -> None:
        """Handle unclip ratio change"""
        self.app.ocr_config.paddleocr_config.text_det_unclip_ratio = round(float(value), 3)
        self._invalidate_ocr_instance()

    def _on_batch_size_changed(self) -> None:
        """Handle batch size change"""
        self.app.ocr_config.paddleocr_config.text_recognition_batch_size = self.batch_size_var.get()
        self._invalidate_ocr_instance()

    def _on_cpu_threads_changed(self) -> None:
        """Handle CPU threads change"""
        self.app.ocr_config.paddleocr_config.cpu_threads = self.cpu_threads_var.get()
        self._invalidate_ocr_instance()

    def _on_enable_hpi_changed(self) -> None:
        """Handle High Performance Inference checkbox change"""
        self.app.ocr_config.paddleocr_config.enable_hpi = self.enable_hpi_var.get()
        self._invalidate_ocr_instance()

    def _on_mkldnn_cache_changed(self) -> None:
        """Handle MKL-DNN Cache Capacity spinbox change"""
        self.app.ocr_config.paddleocr_config.mkldnn_cache_capacity = self.mkldnn_cache_var.get()
        self._invalidate_ocr_instance()

    def _on_precision_changed(self, value) -> None:
        """Handle Precision combobox change"""
        self.app.ocr_config.paddleocr_config.precision = self.precision_var.get()
        self._invalidate_ocr_instance()

    def _on_use_tensorrt_changed(self) -> None:
        """Handle TensorRT checkbox change"""
        self.app.ocr_config.paddleocr_config.use_tensorrt = self.use_tensorrt_var.get()
        self._invalidate_ocr_instance()

    def _on_enable_mkldnn_changed(self) -> None:
        """Handle MKL-DNN checkbox change"""
        self.app.ocr_config.paddleocr_config.enable_mkldnn = self.enable_mkldnn_var.get()
        self._invalidate_ocr_instance()

    def _on_textline_orientation_batch_changed(self) -> None:
        """Handle Textline Orientation Batch Size spinbox change"""
        self.app.ocr_config.paddleocr_config.textline_orientation_batch_size = self.textline_orientation_batch_var.get()
        self._invalidate_ocr_instance()

    def _on_det_limit_side_len_changed(self) -> None:
        """Handle Detection Limit Side Length spinbox change"""
        self.app.ocr_config.paddleocr_config.text_det_limit_side_len = self.det_limit_side_len_var.get()
        self._invalidate_ocr_instance()

    def _on_det_limit_type_changed(self, value) -> None:
        """Handle Detection Limit Type combobox change"""
        self.app.ocr_config.paddleocr_config.text_det_limit_type = self.det_limit_type_var.get()
        self._invalidate_ocr_instance()

    def _on_tesseract_lang_changed(self, value) -> None:
        """Handle Tesseract language change"""
        self.app.ocr_config.tesseract_config.lang = self.tesseract_lang_var.get()
        self._invalidate_ocr_instance()

    def _on_psm_changed(self) -> None:
        """Handle PSM change"""
        self.app.ocr_config.tesseract_config.psm = self.psm_var.get()
        self._invalidate_ocr_instance()

    def _on_oem_changed(self) -> None:
        """Handle OEM change"""
        self.app.ocr_config.tesseract_config.oem = self.oem_var.get()
        self._invalidate_ocr_instance()

    def _on_tesseract_config_changed(self, event) -> None:
        """Handle Tesseract config string change"""
        self.app.ocr_config.tesseract_config.config = self.tesseract_config_var.get()
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
