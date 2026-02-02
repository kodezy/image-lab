from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Config:
    """Base configuration class"""

    def update_from_dict(self, data: dict[str, Any]) -> None:
        """Update config from dictionary"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class CaptureConfig(Config):
    """Image capture configuration"""

    device_enabled: bool = False
    device_id: int = 0


@dataclass
class PaddleOCRConfig(Config):
    """PaddleOCR configuration based on official documentation"""

    ocr_version: Literal["PP-OCRv5", "PP-OCRv4", "PP-OCRv3"] = "PP-OCRv5"
    lang: str = "en"
    device: str = "cpu"

    doc_orientation_classify_model_name: str | None = None
    doc_orientation_classify_model_dir: str | None = None
    doc_unwarping_model_name: str | None = None
    doc_unwarping_model_dir: str | None = None
    text_detection_model_name: str | None = None
    text_detection_model_dir: str | None = None
    textline_orientation_model_name: str | None = None
    textline_orientation_model_dir: str | None = None
    text_recognition_model_name: str | None = None
    text_recognition_model_dir: str | None = None

    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False

    text_det_limit_side_len: int = 960
    text_det_limit_type: Literal["min", "max"] = "max"
    text_det_thresh: float = 0.3
    text_det_box_thresh: float = 0.6
    text_det_unclip_ratio: float = 1.5
    text_det_input_shape: tuple[int, int] | None = None

    text_rec_score_thresh: float = 0.5
    text_recognition_batch_size: int = 6
    text_rec_input_shape: tuple[int, int] | None = None
    textline_orientation_batch_size: int = 1

    enable_hpi: bool = False
    enable_mkldnn: bool = True
    mkldnn_cache_capacity: int = 10
    cpu_threads: int = 8
    use_tensorrt: bool = False
    precision: str = "fp32"


@dataclass
class TesseractConfig(Config):
    """Tesseract OCR configuration"""

    lang: str = "eng"
    psm: int = 3
    oem: Literal[0, 1, 2, 3] = 1
    config: str = ""


@dataclass
class RapidOCRConfig(Config):
    lang_type: Literal["ch", "en", "multi"] = "en"
    ocr_version: Literal["PP-OCRv4", "PP-OCRv5"] = "PP-OCRv4"
    model_type: Literal["mobile", "server"] = "mobile"

    use_det: bool = True
    use_cls: bool = True
    use_rec: bool = True
    text_score: float = 0.5
    min_height: int = 30
    width_height_ratio: float = 8.0
    max_side_len: int = 2000
    min_side_len: int = 30

    limit_side_len: int = 736
    limit_type: Literal["min", "max"] = "min"
    thresh: float = 0.3
    box_thresh: float = 0.5
    max_candidates: int = 1000
    unclip_ratio: float = 1.6
    use_dilation: bool = True
    score_mode: Literal["fast", "slow"] = "fast"

    cls_batch_num: int = 6
    cls_thresh: float = 0.9

    rec_batch_num: int = 6


@dataclass
class EasyOCRConfig(Config):
    """EasyOCR configuration"""

    lang_list: list[str] = field(default_factory=lambda: ["en"])
    gpu: bool | str = True
    model_storage_directory: str | None = None
    download_enabled: bool = True
    user_network_directory: str | None = None
    recog_network: str = "standard"
    detector: bool = True
    recognizer: bool = True

    decoder: str = "greedy"
    beam_width: int = 5
    batch_size: int = 1
    workers: int = 0
    allowlist: str | None = None
    blocklist: str | None = None
    detail: int = 1
    paragraph: bool = False
    min_size: int = 10
    rotation_info: list[int] | None = None

    contrast_ths: float = 0.1
    adjust_contrast: float = 0.5

    text_threshold: float = 0.7
    low_text: float = 0.4
    link_threshold: float = 0.4
    canvas_size: int = 2560
    mag_ratio: float = 1.0

    slope_ths: float = 0.1
    ycenter_ths: float = 0.5
    height_ths: float = 0.5
    width_ths: float = 0.5
    add_margin: float = 0.1
    x_ths: float = 1.0
    y_ths: float = 0.5


@dataclass
class OCRConfig(Config):
    ocr_type: Literal["paddleocr", "tesseract", "easyocr", "rapidocr"] = "paddleocr"
    paddleocr_config: PaddleOCRConfig = field(default_factory=PaddleOCRConfig)
    tesseract_config: TesseractConfig = field(default_factory=TesseractConfig)
    easyocr_config: EasyOCRConfig = field(default_factory=EasyOCRConfig)
    rapidocr_config: RapidOCRConfig = field(default_factory=RapidOCRConfig)

    def update_from_dict(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            if key == "paddleocr_config" and isinstance(value, dict):
                self.paddleocr_config.update_from_dict(value)
            elif key == "tesseract_config" and isinstance(value, dict):
                self.tesseract_config.update_from_dict(value)
            elif key == "easyocr_config" and isinstance(value, dict):
                self.easyocr_config.update_from_dict(value)
            elif key == "rapidocr_config" and isinstance(value, dict):
                self.rapidocr_config.update_from_dict(value)
            elif hasattr(self, key):
                setattr(self, key, value)


@dataclass
class ProcessingConfig(Config):
    """Image processing configuration organized by processing stages"""

    # Basic transformations
    color_space: Literal["Grayscale", "RGB", "BGR", "HSV", "LAB", "YUV", "YCrCb"] = "BGR"
    crop_enabled: bool = False
    bbox: tuple[int, int, int, int] | None = None  # (x1, y1, x2, y2) or None
    resize_enabled: bool = False
    resize_width: int = 1920
    resize_height: int = 1080
    resize_maintain_aspect_ratio: bool = True
    gamma_correction: bool = False
    gamma_value: float = 1.0

    # Noise reduction
    denoise_nl_means: bool = False
    denoise_h: float = 10.0
    denoise_template_window: int = 7
    denoise_search_window: int = 21
    edge_preserving_filter: bool = False
    edge_filter_flags: int = 1
    edge_sigma_s: float = 50.0
    edge_sigma_r: float = 0.4
    noise_reduction_bilateral: bool = False
    bilateral_iterations: int = 1

    # Filters
    bilateral_filter: bool = False
    bilateral_d: int = 9
    bilateral_sigma_color: int = 75
    bilateral_sigma_space: int = 75
    gaussian_blur: bool = False
    gaussian_kernel: int = 3
    gaussian_sigma: float = 0.0
    median_filter: bool = False
    median_kernel: int = 3
    background_subtraction: bool = False
    bg_threshold: int = 50

    # Histogram and contrast
    histogram_equalization: bool = False
    clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    adaptive_hist_eq: bool = False
    adaptive_hist_kernel: int = 8
    multi_otsu: bool = False
    multi_otsu_classes: int = 3
    intensity_normalization: bool = False
    norm_min: int = 0
    norm_max: int = 255
    contrast_stretching: bool = False
    stretch_min_percentile: float = 2.0
    stretch_max_percentile: float = 98.0

    # Line removal
    vertical_line_removal: bool = False
    vertical_kernel_size: int = 3
    horizontal_line_removal: bool = False
    horizontal_kernel_size: int = 3
    hough_lines_removal: bool = False
    hough_threshold: int = 100
    hough_min_line_length: int = 30
    hough_max_line_gap: int = 10

    # Morphological operations
    stroke_width_normalization: bool = False
    stroke_iterations: int = 1
    morphology: bool = False
    morph_kernel_size: int = 2
    morph_open: bool = True
    morph_close: bool = True
    tophat: bool = False
    tophat_kernel_size: int = 3
    blackhat: bool = False
    blackhat_kernel_size: int = 3
    gradient: bool = False
    gradient_kernel_size: int = 3
    morphological_gradient: bool = False
    morphological_gradient_kernel: int = 3

    # Character operations
    character_separation: bool = False
    char_sep_kernel_size: int = 1
    character_dilation: bool = False
    dilation_kernel_size: int = 1
    dilation_iterations: int = 1
    character_erosion: bool = False
    erosion_kernel_size: int = 1
    erosion_iterations: int = 1
    noise_dots_removal: bool = False
    min_contour_area: int = 10

    # Enhancement
    text_enhancement: bool = False
    text_kernel_size: int = 1
    detail_enhancement: bool = False
    detail_sigma_s: float = 10.0
    detail_sigma_r: float = 0.15
    edge_enhancement: bool = False
    edge_strength: float = 1.0
    unsharp_mask: bool = False
    unsharp_strength: float = 1.5
    sharpen: bool = False
    sharpen_strength: float = 0.2

    # Thresholding
    threshold_enabled: bool = False
    threshold_type: Literal["BINARY", "BINARY_INV", "OTSU_BINARY", "ADAPTIVE_MEAN", "ADAPTIVE_GAUSSIAN"] = "BINARY"
    threshold_value: int = 127
    adaptive_block_size: int = 11
    adaptive_c: int = 2

    # Contour filtering
    contour_filtering: bool = False
    contour_area_min: int = 50
    contour_area_max: int = 10000
    connected_components_filtering: bool = False
    cc_min_area: int = 20
    cc_max_area: int = 5000
    aspect_ratio_filtering: bool = False
    min_aspect_ratio: float = 0.1
    max_aspect_ratio: float = 10.0

    # Advanced operations
    distance_transform: bool = False
    distance_transform_type: int = 1
    skeletonize: bool = False
    watershed_markers: bool = False
    local_binary_pattern: bool = False
    lbp_radius: int = 3
    lbp_n_points: int = 24
