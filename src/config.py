from dataclasses import dataclass
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
    device_max_width: int = 1920
    device_max_height: int = 1080


@dataclass
class OCRConfig(Config):
    """PaddleOCR configuration based on official documentation"""

    # Model and version
    ocr_version: Literal["PP-OCRv5", "PP-OCRv4", "PP-OCRv3"] = "PP-OCRv5"
    lang: str = "en"  # ch, en, etc.
    device: str = "cpu"  # cpu, gpu:0, npu:0, xpu:0, mlu:0, dcu:0

    # Module control
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False

    # Text detection
    text_det_limit_side_len: int = 960
    text_det_limit_type: Literal["min", "max"] = "max"
    text_det_thresh: float = 0.3
    text_det_box_thresh: float = 0.6
    text_det_unclip_ratio: float = 1.5

    # Text recognition
    text_rec_score_thresh: float = 0.5
    text_recognition_batch_size: int = 6

    # Performance
    enable_mkldnn: bool = True
    cpu_threads: int = 8
    use_tensorrt: bool = False
    precision: str = "fp32"  # fp32, fp16


@dataclass
class ProcessingConfig(Config):
    """Image processing configuration organized by processing stages"""

    # Basic transformations
    grayscale: bool = False
    crop_enabled: bool = False
    crop_x1: int = 0
    crop_y1: int = 0
    crop_x2: int = 0
    crop_y2: int = 0
    resize_before_process: bool = False
    resize_width: int = 1920
    resize_height: int = 1080
    maintain_aspect_ratio: bool = True
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
    threshold_type: Literal["BINARY", "OTSU_BINARY", "ADAPTIVE_MEAN", "ADAPTIVE_GAUSSIAN"] = "BINARY"
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
