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

        elif ocr_config.ocr_type == "tesseract":
            tesseract_config = ocr_config.tesseract_config

            self.tesseract_lang_var.set(tesseract_config.lang)
            self.psm_var.set(tesseract_config.psm)
            self.oem_var.set(tesseract_config.oem)
            self.tesseract_config_var.set(tesseract_config.config)

        elif ocr_config.ocr_type == "rapidocr":
            rapid_config = ocr_config.rapidocr_config

            self.rapidocr_lang_var.set(rapid_config.lang_type)
            self.rapidocr_version_var.set(rapid_config.ocr_version)
            self.rapidocr_model_var.set(rapid_config.model_type)
            self.rapidocr_use_det_var.set(rapid_config.use_det)
            self.rapidocr_use_cls_var.set(rapid_config.use_cls)
            self.rapidocr_use_rec_var.set(rapid_config.use_rec)
            self.rapidocr_text_score_var.set(rapid_config.text_score)
            self.rapidocr_min_height_var.set(rapid_config.min_height)
            self.rapidocr_width_height_ratio_var.set(rapid_config.width_height_ratio)
            self.rapidocr_max_side_len_var.set(rapid_config.max_side_len)
            self.rapidocr_min_side_len_var.set(rapid_config.min_side_len)
            self.rapidocr_limit_side_len_var.set(rapid_config.limit_side_len)
            self.rapidocr_limit_type_var.set(rapid_config.limit_type)
            self.rapidocr_thresh_var.set(rapid_config.thresh)
            self.rapidocr_box_thresh_var.set(rapid_config.box_thresh)
            self.rapidocr_max_candidates_var.set(rapid_config.max_candidates)
            self.rapidocr_unclip_ratio_var.set(rapid_config.unclip_ratio)
            self.rapidocr_use_dilation_var.set(rapid_config.use_dilation)
            self.rapidocr_score_mode_var.set(rapid_config.score_mode)
            self.rapidocr_cls_batch_num_var.set(rapid_config.cls_batch_num)
            self.rapidocr_cls_thresh_var.set(rapid_config.cls_thresh)
            self.rapidocr_rec_batch_num_var.set(rapid_config.rec_batch_num)

        else:
            easyocr_config = ocr_config.easyocr_config

            self.easyocr_lang_var.set(",".join(easyocr_config.lang_list))
            self.easyocr_gpu_var.set(bool(easyocr_config.gpu) if isinstance(easyocr_config.gpu, bool) else True)
            self.easyocr_gpu_str_var.set(str(easyocr_config.gpu) if isinstance(easyocr_config.gpu, str) else "")
            self.easyocr_decoder_var.set(easyocr_config.decoder)
            self.easyocr_beam_width_var.set(easyocr_config.beam_width)
            self.easyocr_batch_size_var.set(easyocr_config.batch_size)
            self.easyocr_workers_var.set(easyocr_config.workers)
            self.easyocr_paragraph_var.set(easyocr_config.paragraph)
            self.easyocr_min_size_var.set(easyocr_config.min_size)
            self.easyocr_text_threshold_var.set(easyocr_config.text_threshold)
            self.easyocr_low_text_var.set(easyocr_config.low_text)
            self.easyocr_link_threshold_var.set(easyocr_config.link_threshold)
            self.easyocr_canvas_size_var.set(easyocr_config.canvas_size)
            self.easyocr_mag_ratio_var.set(easyocr_config.mag_ratio)
            self.easyocr_contrast_ths_var.set(easyocr_config.contrast_ths)
            self.easyocr_adjust_contrast_var.set(easyocr_config.adjust_contrast)

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

        rapidocr_config = ocr_config.rapidocr_config
        self.rapidocr_lang_var = tk.StringVar(value=rapidocr_config.lang_type)
        self.rapidocr_version_var = tk.StringVar(value=rapidocr_config.ocr_version)
        self.rapidocr_model_var = tk.StringVar(value=rapidocr_config.model_type)
        self.rapidocr_use_det_var = tk.BooleanVar(value=rapidocr_config.use_det)
        self.rapidocr_use_cls_var = tk.BooleanVar(value=rapidocr_config.use_cls)
        self.rapidocr_use_rec_var = tk.BooleanVar(value=rapidocr_config.use_rec)
        self.rapidocr_text_score_var = tk.DoubleVar(value=rapidocr_config.text_score)
        self.rapidocr_min_height_var = tk.IntVar(value=rapidocr_config.min_height)
        self.rapidocr_width_height_ratio_var = tk.DoubleVar(value=rapidocr_config.width_height_ratio)
        self.rapidocr_max_side_len_var = tk.IntVar(value=rapidocr_config.max_side_len)
        self.rapidocr_min_side_len_var = tk.IntVar(value=rapidocr_config.min_side_len)
        self.rapidocr_limit_side_len_var = tk.IntVar(value=rapidocr_config.limit_side_len)
        self.rapidocr_limit_type_var = tk.StringVar(value=rapidocr_config.limit_type)
        self.rapidocr_thresh_var = tk.DoubleVar(value=rapidocr_config.thresh)
        self.rapidocr_box_thresh_var = tk.DoubleVar(value=rapidocr_config.box_thresh)
        self.rapidocr_max_candidates_var = tk.IntVar(value=rapidocr_config.max_candidates)
        self.rapidocr_unclip_ratio_var = tk.DoubleVar(value=rapidocr_config.unclip_ratio)
        self.rapidocr_use_dilation_var = tk.BooleanVar(value=rapidocr_config.use_dilation)
        self.rapidocr_score_mode_var = tk.StringVar(value=rapidocr_config.score_mode)
        self.rapidocr_cls_batch_num_var = tk.IntVar(value=rapidocr_config.cls_batch_num)
        self.rapidocr_cls_thresh_var = tk.DoubleVar(value=rapidocr_config.cls_thresh)
        self.rapidocr_rec_batch_num_var = tk.IntVar(value=rapidocr_config.rec_batch_num)

        easyocr_config = ocr_config.easyocr_config
        self.easyocr_lang_var = tk.StringVar(value=",".join(easyocr_config.lang_list))
        self.easyocr_gpu_var = tk.BooleanVar(
            value=bool(easyocr_config.gpu) if isinstance(easyocr_config.gpu, bool) else True,
        )
        self.easyocr_gpu_str_var = tk.StringVar(
            value=str(easyocr_config.gpu) if isinstance(easyocr_config.gpu, str) else "",
        )
        self.easyocr_decoder_var = tk.StringVar(value=easyocr_config.decoder)
        self.easyocr_beam_width_var = tk.IntVar(value=easyocr_config.beam_width)
        self.easyocr_batch_size_var = tk.IntVar(value=easyocr_config.batch_size)
        self.easyocr_workers_var = tk.IntVar(value=easyocr_config.workers)
        self.easyocr_paragraph_var = tk.BooleanVar(value=easyocr_config.paragraph)
        self.easyocr_min_size_var = tk.IntVar(value=easyocr_config.min_size)
        self.easyocr_text_threshold_var = tk.DoubleVar(value=easyocr_config.text_threshold)
        self.easyocr_low_text_var = tk.DoubleVar(value=easyocr_config.low_text)
        self.easyocr_link_threshold_var = tk.DoubleVar(value=easyocr_config.link_threshold)
        self.easyocr_canvas_size_var = tk.IntVar(value=easyocr_config.canvas_size)
        self.easyocr_mag_ratio_var = tk.DoubleVar(value=easyocr_config.mag_ratio)
        self.easyocr_contrast_ths_var = tk.DoubleVar(value=easyocr_config.contrast_ths)
        self.easyocr_adjust_contrast_var = tk.DoubleVar(value=easyocr_config.adjust_contrast)

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
            ["paddleocr", "tesseract", "easyocr", "rapidocr"],
            self._on_ocr_type_changed,
        )
        ocr_type_frame.pack(fill=tk.X, pady=(0, 5))

        self.paddleocr_frame = ttk.Frame(scrollable_frame)
        self._create_paddleocr_settings(self.paddleocr_frame)

        self.tesseract_frame = ttk.Frame(scrollable_frame)
        self._create_tesseract_settings(self.tesseract_frame)

        self.easyocr_frame = ttk.Frame(scrollable_frame)
        self._create_easyocr_settings(self.easyocr_frame)

        self.rapidocr_frame = ttk.Frame(scrollable_frame)
        self._create_rapidocr_settings(self.rapidocr_frame)

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

    def _create_easyocr_settings(self, parent: ttk.Frame) -> None:
        """Create EasyOCR specific settings"""
        lang_frame = ttk.Frame(parent)
        lang_frame.pack(fill=tk.X, pady=(0, 5))
        lang_label = ttk.Label(lang_frame, text="Languages (comma-separated):")
        lang_label.pack(side=tk.LEFT)
        self.easyocr_lang_entry = ttk.Entry(lang_frame, textvariable=self.easyocr_lang_var, width=20)
        self.easyocr_lang_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        self.easyocr_lang_entry.bind("<KeyRelease>", self._on_easyocr_lang_changed)

        gpu_frame = ttk.Frame(parent)
        gpu_frame.pack(fill=tk.X, pady=(0, 5))
        self.easyocr_gpu_checkbox = ttk.Checkbutton(
            gpu_frame,
            text="Enable GPU",
            variable=self.easyocr_gpu_var,
            command=self._on_easyocr_gpu_changed,
        )
        self.easyocr_gpu_checkbox.pack(anchor=tk.W)

        decoder_frame, self.easyocr_decoder_combobox = create_combobox(
            parent,
            "Decoder",
            self.easyocr_decoder_var,
            ["greedy", "beamsearch", "wordbeamsearch"],
            self._on_easyocr_decoder_changed,
        )
        decoder_frame.pack(fill=tk.X, pady=(0, 5))

        beam_width_frame, self.easyocr_beam_width_spinbox = create_spinbox(
            parent,
            "Beam Width",
            self.easyocr_beam_width_var,
            1,
            20,
            self._on_easyocr_beam_width_changed,
        )
        beam_width_frame.pack(fill=tk.X, pady=(0, 5))

        batch_size_frame, self.easyocr_batch_size_spinbox = create_spinbox(
            parent,
            "Batch Size",
            self.easyocr_batch_size_var,
            1,
            32,
            self._on_easyocr_batch_size_changed,
        )
        batch_size_frame.pack(fill=tk.X, pady=(0, 5))

        workers_frame, self.easyocr_workers_spinbox = create_spinbox(
            parent,
            "Workers",
            self.easyocr_workers_var,
            0,
            16,
            self._on_easyocr_workers_changed,
        )
        workers_frame.pack(fill=tk.X, pady=(0, 5))

        paragraph_frame = ttk.Frame(parent)
        paragraph_frame.pack(fill=tk.X, pady=(0, 5))
        self.easyocr_paragraph_checkbox = ttk.Checkbutton(
            paragraph_frame,
            text="Paragraph Mode",
            variable=self.easyocr_paragraph_var,
            command=self._on_easyocr_paragraph_changed,
        )
        self.easyocr_paragraph_checkbox.pack(anchor=tk.W)

        min_size_frame, self.easyocr_min_size_spinbox = create_spinbox(
            parent,
            "Min Size",
            self.easyocr_min_size_var,
            1,
            100,
            self._on_easyocr_min_size_changed,
        )
        min_size_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=(5, 10))

        text_threshold_frame, _, _ = create_slider(
            parent,
            "Text Threshold",
            self.easyocr_text_threshold_var,
            0.1,
            0.9,
            self._on_easyocr_text_threshold_changed,
        )
        text_threshold_frame.pack(fill=tk.X, pady=(0, 5))

        low_text_frame, _, _ = create_slider(
            parent,
            "Low Text",
            self.easyocr_low_text_var,
            0.1,
            0.9,
            self._on_easyocr_low_text_changed,
        )
        low_text_frame.pack(fill=tk.X, pady=(0, 5))

        link_threshold_frame, _, _ = create_slider(
            parent,
            "Link Threshold",
            self.easyocr_link_threshold_var,
            0.1,
            0.9,
            self._on_easyocr_link_threshold_changed,
        )
        link_threshold_frame.pack(fill=tk.X, pady=(0, 5))

        canvas_size_frame, self.easyocr_canvas_size_spinbox = create_spinbox(
            parent,
            "Canvas Size",
            self.easyocr_canvas_size_var,
            512,
            4096,
            self._on_easyocr_canvas_size_changed,
        )
        canvas_size_frame.pack(fill=tk.X, pady=(0, 5))

        mag_ratio_frame, _, _ = create_slider(
            parent,
            "Magnification Ratio",
            self.easyocr_mag_ratio_var,
            0.5,
            2.0,
            self._on_easyocr_mag_ratio_changed,
        )
        mag_ratio_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=(5, 10))

        contrast_ths_frame, _, _ = create_slider(
            parent,
            "Contrast Threshold",
            self.easyocr_contrast_ths_var,
            0.0,
            1.0,
            self._on_easyocr_contrast_ths_changed,
        )
        contrast_ths_frame.pack(fill=tk.X, pady=(0, 5))

        adjust_contrast_frame, _, _ = create_slider(
            parent,
            "Adjust Contrast",
            self.easyocr_adjust_contrast_var,
            0.0,
            1.0,
            self._on_easyocr_adjust_contrast_changed,
        )
        adjust_contrast_frame.pack(fill=tk.X, pady=(0, 5))

    def _create_rapidocr_settings(self, parent: ttk.Frame) -> None:
        lang_frame, self.rapidocr_lang_combobox = create_combobox(
            parent,
            "Language",
            self.rapidocr_lang_var,
            ["ch", "en", "multi"],
            self._on_rapidocr_lang_changed,
        )
        lang_frame.pack(fill=tk.X, pady=(0, 5))

        version_frame, self.rapidocr_version_combobox = create_combobox(
            parent,
            "OCR Version",
            self.rapidocr_version_var,
            ["PP-OCRv4", "PP-OCRv5"],
            self._on_rapidocr_version_changed,
        )
        version_frame.pack(fill=tk.X, pady=(0, 5))

        model_frame, self.rapidocr_model_combobox = create_combobox(
            parent,
            "Model Type",
            self.rapidocr_model_var,
            ["mobile", "server"],
            self._on_rapidocr_model_changed,
        )
        model_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=(5, 10))

        use_det_frame = ttk.Frame(parent)
        use_det_frame.pack(fill=tk.X, pady=(0, 5))
        self.rapidocr_use_det_checkbox = ttk.Checkbutton(
            use_det_frame,
            text="Use Text Detection",
            variable=self.rapidocr_use_det_var,
            command=self._on_rapidocr_use_det_changed,
        )
        self.rapidocr_use_det_checkbox.pack(anchor=tk.W)

        use_cls_frame = ttk.Frame(parent)
        use_cls_frame.pack(fill=tk.X, pady=(0, 5))
        self.rapidocr_use_cls_checkbox = ttk.Checkbutton(
            use_cls_frame,
            text="Use Text Direction Classification",
            variable=self.rapidocr_use_cls_var,
            command=self._on_rapidocr_use_cls_changed,
        )
        self.rapidocr_use_cls_checkbox.pack(anchor=tk.W)

        use_rec_frame = ttk.Frame(parent)
        use_rec_frame.pack(fill=tk.X, pady=(0, 5))
        self.rapidocr_use_rec_checkbox = ttk.Checkbutton(
            use_rec_frame,
            text="Use Text Recognition",
            variable=self.rapidocr_use_rec_var,
            command=self._on_rapidocr_use_rec_changed,
        )
        self.rapidocr_use_rec_checkbox.pack(anchor=tk.W)

        text_score_frame, _, _ = create_slider(
            parent,
            "Text Score Threshold",
            self.rapidocr_text_score_var,
            0.0,
            1.0,
            self._on_rapidocr_text_score_changed,
        )
        text_score_frame.pack(fill=tk.X, pady=(0, 5))

        min_height_frame, self.rapidocr_min_height_spinbox = create_spinbox(
            parent,
            "Min Height",
            self.rapidocr_min_height_var,
            10,
            500,
            self._on_rapidocr_min_height_changed,
        )
        min_height_frame.pack(fill=tk.X, pady=(0, 5))

        width_height_ratio_frame, _, _ = create_slider(
            parent,
            "Width/Height Ratio",
            self.rapidocr_width_height_ratio_var,
            1.0,
            20.0,
            self._on_rapidocr_width_height_ratio_changed,
        )
        width_height_ratio_frame.pack(fill=tk.X, pady=(0, 5))

        max_side_len_frame, self.rapidocr_max_side_len_spinbox = create_spinbox(
            parent,
            "Max Side Length",
            self.rapidocr_max_side_len_var,
            500,
            5000,
            self._on_rapidocr_max_side_len_changed,
        )
        max_side_len_frame.pack(fill=tk.X, pady=(0, 5))

        min_side_len_frame, self.rapidocr_min_side_len_spinbox = create_spinbox(
            parent,
            "Min Side Length",
            self.rapidocr_min_side_len_var,
            10,
            500,
            self._on_rapidocr_min_side_len_changed,
        )
        min_side_len_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=(5, 10))

        limit_side_len_frame, self.rapidocr_limit_side_len_spinbox = create_spinbox(
            parent,
            "Detection Limit Side Length",
            self.rapidocr_limit_side_len_var,
            64,
            4096,
            self._on_rapidocr_limit_side_len_changed,
        )
        limit_side_len_frame.pack(fill=tk.X, pady=(0, 5))

        limit_type_frame, self.rapidocr_limit_type_combobox = create_combobox(
            parent,
            "Detection Limit Type",
            self.rapidocr_limit_type_var,
            ["min", "max"],
            self._on_rapidocr_limit_type_changed,
        )
        limit_type_frame.pack(fill=tk.X, pady=(0, 5))

        thresh_frame, _, _ = create_slider(
            parent,
            "Detection Threshold",
            self.rapidocr_thresh_var,
            0.1,
            0.9,
            self._on_rapidocr_thresh_changed,
        )
        thresh_frame.pack(fill=tk.X, pady=(0, 5))

        box_thresh_frame, _, _ = create_slider(
            parent,
            "Box Threshold",
            self.rapidocr_box_thresh_var,
            0.1,
            0.9,
            self._on_rapidocr_box_thresh_changed,
        )
        box_thresh_frame.pack(fill=tk.X, pady=(0, 5))

        unclip_ratio_frame, _, _ = create_slider(
            parent,
            "Unclip Ratio",
            self.rapidocr_unclip_ratio_var,
            1.0,
            3.0,
            self._on_rapidocr_unclip_ratio_changed,
        )
        unclip_ratio_frame.pack(fill=tk.X, pady=(0, 5))

        max_candidates_frame, self.rapidocr_max_candidates_spinbox = create_spinbox(
            parent,
            "Max Candidates",
            self.rapidocr_max_candidates_var,
            100,
            5000,
            self._on_rapidocr_max_candidates_changed,
        )
        max_candidates_frame.pack(fill=tk.X, pady=(0, 5))

        use_dilation_frame = ttk.Frame(parent)
        use_dilation_frame.pack(fill=tk.X, pady=(0, 5))
        self.rapidocr_use_dilation_checkbox = ttk.Checkbutton(
            use_dilation_frame,
            text="Use Dilation",
            variable=self.rapidocr_use_dilation_var,
            command=self._on_rapidocr_use_dilation_changed,
        )
        self.rapidocr_use_dilation_checkbox.pack(anchor=tk.W)

        score_mode_frame, self.rapidocr_score_mode_combobox = create_combobox(
            parent,
            "Score Mode",
            self.rapidocr_score_mode_var,
            ["fast", "slow"],
            self._on_rapidocr_score_mode_changed,
        )
        score_mode_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=(5, 10))

        cls_batch_num_frame, self.rapidocr_cls_batch_num_spinbox = create_spinbox(
            parent,
            "Classification Batch Size",
            self.rapidocr_cls_batch_num_var,
            1,
            32,
            self._on_rapidocr_cls_batch_num_changed,
        )
        cls_batch_num_frame.pack(fill=tk.X, pady=(0, 5))

        cls_thresh_frame, _, _ = create_slider(
            parent,
            "Classification Threshold",
            self.rapidocr_cls_thresh_var,
            0.5,
            1.0,
            self._on_rapidocr_cls_thresh_changed,
        )
        cls_thresh_frame.pack(fill=tk.X, pady=(0, 5))

        rec_batch_num_frame, self.rapidocr_rec_batch_num_spinbox = create_spinbox(
            parent,
            "Recognition Batch Size",
            self.rapidocr_rec_batch_num_var,
            1,
            32,
            self._on_rapidocr_rec_batch_num_changed,
        )
        rec_batch_num_frame.pack(fill=tk.X, pady=(0, 5))

    def _update_visible_settings(self) -> None:
        """Update visible settings based on OCR type"""
        ocr_type = self.ocr_type_var.get()

        self.paddleocr_frame.pack_forget()
        self.tesseract_frame.pack_forget()
        self.easyocr_frame.pack_forget()
        self.rapidocr_frame.pack_forget()

        if ocr_type == "paddleocr":
            self.paddleocr_frame.pack(fill=tk.X, anchor=tk.NW)
        elif ocr_type == "tesseract":
            self.tesseract_frame.pack(fill=tk.X, anchor=tk.NW)
        elif ocr_type == "rapidocr":
            self.rapidocr_frame.pack(fill=tk.X, anchor=tk.NW)
        else:
            self.easyocr_frame.pack(fill=tk.X, anchor=tk.NW)

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

    def _on_easyocr_lang_changed(self, event) -> None:
        """Handle EasyOCR language change"""
        lang_str = self.easyocr_lang_var.get()
        lang_list = [lang.strip() for lang in lang_str.split(",") if lang.strip()]
        self.app.ocr_config.easyocr_config.lang_list = lang_list if lang_list else ["en"]
        self._invalidate_ocr_instance()

    def _on_easyocr_gpu_changed(self) -> None:
        """Handle EasyOCR GPU change"""
        if self.easyocr_gpu_var.get():
            gpu_str = self.easyocr_gpu_str_var.get()
            self.app.ocr_config.easyocr_config.gpu = gpu_str if gpu_str else True
        else:
            self.app.ocr_config.easyocr_config.gpu = False
        self._invalidate_ocr_instance()

    def _on_easyocr_decoder_changed(self, value) -> None:
        """Handle EasyOCR decoder change"""
        self.app.ocr_config.easyocr_config.decoder = self.easyocr_decoder_var.get()
        self._invalidate_ocr_instance()

    def _on_easyocr_beam_width_changed(self) -> None:
        """Handle EasyOCR beam width change"""
        self.app.ocr_config.easyocr_config.beam_width = self.easyocr_beam_width_var.get()
        self._invalidate_ocr_instance()

    def _on_easyocr_batch_size_changed(self) -> None:
        """Handle EasyOCR batch size change"""
        self.app.ocr_config.easyocr_config.batch_size = self.easyocr_batch_size_var.get()
        self._invalidate_ocr_instance()

    def _on_easyocr_workers_changed(self) -> None:
        """Handle EasyOCR workers change"""
        self.app.ocr_config.easyocr_config.workers = self.easyocr_workers_var.get()
        self._invalidate_ocr_instance()

    def _on_easyocr_paragraph_changed(self) -> None:
        """Handle EasyOCR paragraph change"""
        self.app.ocr_config.easyocr_config.paragraph = self.easyocr_paragraph_var.get()
        self._invalidate_ocr_instance()

    def _on_easyocr_min_size_changed(self) -> None:
        """Handle EasyOCR min size change"""
        self.app.ocr_config.easyocr_config.min_size = self.easyocr_min_size_var.get()
        self._invalidate_ocr_instance()

    def _on_easyocr_text_threshold_changed(self, value) -> None:
        """Handle EasyOCR text threshold change"""
        self.app.ocr_config.easyocr_config.text_threshold = round(float(value), 3)
        self._invalidate_ocr_instance()

    def _on_easyocr_low_text_changed(self, value) -> None:
        """Handle EasyOCR low text change"""
        self.app.ocr_config.easyocr_config.low_text = round(float(value), 3)
        self._invalidate_ocr_instance()

    def _on_easyocr_link_threshold_changed(self, value) -> None:
        """Handle EasyOCR link threshold change"""
        self.app.ocr_config.easyocr_config.link_threshold = round(float(value), 3)
        self._invalidate_ocr_instance()

    def _on_easyocr_canvas_size_changed(self) -> None:
        """Handle EasyOCR canvas size change"""
        self.app.ocr_config.easyocr_config.canvas_size = self.easyocr_canvas_size_var.get()
        self._invalidate_ocr_instance()

    def _on_easyocr_mag_ratio_changed(self, value) -> None:
        """Handle EasyOCR magnification ratio change"""
        self.app.ocr_config.easyocr_config.mag_ratio = round(float(value), 3)
        self._invalidate_ocr_instance()

    def _on_easyocr_contrast_ths_changed(self, value) -> None:
        """Handle EasyOCR contrast threshold change"""
        self.app.ocr_config.easyocr_config.contrast_ths = round(float(value), 3)
        self._invalidate_ocr_instance()

    def _on_easyocr_adjust_contrast_changed(self, value) -> None:
        self.app.ocr_config.easyocr_config.adjust_contrast = round(float(value), 3)
        self._invalidate_ocr_instance()

    def _on_rapidocr_lang_changed(self, value) -> None:
        self.app.ocr_config.rapidocr_config.lang_type = self.rapidocr_lang_var.get()
        self._invalidate_ocr_instance()

    def _on_rapidocr_version_changed(self, value) -> None:
        self.app.ocr_config.rapidocr_config.ocr_version = self.rapidocr_version_var.get()
        self._invalidate_ocr_instance()

    def _on_rapidocr_model_changed(self, value) -> None:
        self.app.ocr_config.rapidocr_config.model_type = self.rapidocr_model_var.get()
        self._invalidate_ocr_instance()

    def _on_rapidocr_use_cls_changed(self) -> None:
        self.app.ocr_config.rapidocr_config.use_cls = self.rapidocr_use_cls_var.get()
        self._invalidate_ocr_instance()

    def _on_rapidocr_text_score_changed(self, value) -> None:
        self.app.ocr_config.rapidocr_config.text_score = round(float(value), 3)
        self._invalidate_ocr_instance()

    def _on_rapidocr_limit_side_len_changed(self) -> None:
        self.app.ocr_config.rapidocr_config.limit_side_len = self.rapidocr_limit_side_len_var.get()
        self._invalidate_ocr_instance()

    def _on_rapidocr_limit_type_changed(self, value) -> None:
        self.app.ocr_config.rapidocr_config.limit_type = self.rapidocr_limit_type_var.get()
        self._invalidate_ocr_instance()

    def _on_rapidocr_thresh_changed(self, value) -> None:
        self.app.ocr_config.rapidocr_config.thresh = round(float(value), 3)
        self._invalidate_ocr_instance()

    def _on_rapidocr_box_thresh_changed(self, value) -> None:
        self.app.ocr_config.rapidocr_config.box_thresh = round(float(value), 3)
        self._invalidate_ocr_instance()

    def _on_rapidocr_unclip_ratio_changed(self, value) -> None:
        self.app.ocr_config.rapidocr_config.unclip_ratio = round(float(value), 3)
        self._invalidate_ocr_instance()

    def _on_rapidocr_max_candidates_changed(self) -> None:
        self.app.ocr_config.rapidocr_config.max_candidates = self.rapidocr_max_candidates_var.get()
        self._invalidate_ocr_instance()

    def _on_rapidocr_use_dilation_changed(self) -> None:
        self.app.ocr_config.rapidocr_config.use_dilation = self.rapidocr_use_dilation_var.get()
        self._invalidate_ocr_instance()

    def _on_rapidocr_score_mode_changed(self, value) -> None:
        self.app.ocr_config.rapidocr_config.score_mode = self.rapidocr_score_mode_var.get()
        self._invalidate_ocr_instance()

    def _on_rapidocr_cls_batch_num_changed(self) -> None:
        self.app.ocr_config.rapidocr_config.cls_batch_num = self.rapidocr_cls_batch_num_var.get()
        self._invalidate_ocr_instance()

    def _on_rapidocr_cls_thresh_changed(self, value) -> None:
        self.app.ocr_config.rapidocr_config.cls_thresh = round(float(value), 3)
        self._invalidate_ocr_instance()

    def _on_rapidocr_rec_batch_num_changed(self) -> None:
        self.app.ocr_config.rapidocr_config.rec_batch_num = self.rapidocr_rec_batch_num_var.get()
        self._invalidate_ocr_instance()

    def _on_rapidocr_use_det_changed(self) -> None:
        self.app.ocr_config.rapidocr_config.use_det = self.rapidocr_use_det_var.get()
        self._invalidate_ocr_instance()

    def _on_rapidocr_use_rec_changed(self) -> None:
        self.app.ocr_config.rapidocr_config.use_rec = self.rapidocr_use_rec_var.get()
        self._invalidate_ocr_instance()

    def _on_rapidocr_min_height_changed(self) -> None:
        self.app.ocr_config.rapidocr_config.min_height = self.rapidocr_min_height_var.get()
        self._invalidate_ocr_instance()

    def _on_rapidocr_width_height_ratio_changed(self, value) -> None:
        self.app.ocr_config.rapidocr_config.width_height_ratio = round(float(value), 2)
        self._invalidate_ocr_instance()

    def _on_rapidocr_max_side_len_changed(self) -> None:
        self.app.ocr_config.rapidocr_config.max_side_len = self.rapidocr_max_side_len_var.get()
        self._invalidate_ocr_instance()

    def _on_rapidocr_min_side_len_changed(self) -> None:
        self.app.ocr_config.rapidocr_config.min_side_len = self.rapidocr_min_side_len_var.get()
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
