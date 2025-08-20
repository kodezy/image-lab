from paddleocr import PaddleOCR

from src.config import OCRConfig
from src.infra.cache import ocr_cache


@ocr_cache(max_size=16)
def create_ocr(config: OCRConfig | None = None) -> PaddleOCR:
    """Create PaddleOCR instance with custom configurations"""
    if config is None:
        config = OCRConfig()

    return PaddleOCR(
        # Model and version
        ocr_version=config.ocr_version,
        lang=config.lang,
        device=config.device,
        # Model customization
        doc_orientation_classify_model_name=config.doc_orientation_classify_model_name,
        doc_orientation_classify_model_dir=config.doc_orientation_classify_model_dir,
        doc_unwarping_model_name=config.doc_unwarping_model_name,
        doc_unwarping_model_dir=config.doc_unwarping_model_dir,
        text_detection_model_name=config.text_detection_model_name,
        text_detection_model_dir=config.text_detection_model_dir,
        textline_orientation_model_name=config.textline_orientation_model_name,
        textline_orientation_model_dir=config.textline_orientation_model_dir,
        text_recognition_model_name=config.text_recognition_model_name,
        text_recognition_model_dir=config.text_recognition_model_dir,
        # Module control
        use_doc_orientation_classify=config.use_doc_orientation_classify,
        use_doc_unwarping=config.use_doc_unwarping,
        use_textline_orientation=config.use_textline_orientation,
        # Text detection
        text_det_limit_side_len=config.text_det_limit_side_len,
        text_det_limit_type=config.text_det_limit_type,
        text_det_thresh=config.text_det_thresh,
        text_det_box_thresh=config.text_det_box_thresh,
        text_det_unclip_ratio=config.text_det_unclip_ratio,
        text_det_input_shape=config.text_det_input_shape,
        # Text recognition
        text_rec_score_thresh=config.text_rec_score_thresh,
        text_recognition_batch_size=config.text_recognition_batch_size,
        text_rec_input_shape=config.text_rec_input_shape,
        textline_orientation_batch_size=config.textline_orientation_batch_size,
        # Performance
        enable_hpi=config.enable_hpi,
        enable_mkldnn=config.enable_mkldnn,
        mkldnn_cache_capacity=config.mkldnn_cache_capacity,
        cpu_threads=config.cpu_threads,
        use_tensorrt=config.use_tensorrt,
        precision=config.precision,
    )
