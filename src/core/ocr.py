from paddleocr import PaddleOCR

from src.config import OCRConfig
from src.infra.cache import ocr_cache


@ocr_cache(max_size=16)
def create_ocr(config: OCRConfig | None = None) -> PaddleOCR:
    """Create PaddleOCR instance with custom configurations"""
    if config is None:
        config = OCRConfig()

    return PaddleOCR(
        ocr_version=config.ocr_version,
        lang=config.lang,
        device=config.device,
        use_doc_orientation_classify=config.use_doc_orientation_classify,
        use_doc_unwarping=config.use_doc_unwarping,
        use_textline_orientation=config.use_textline_orientation,
        text_det_limit_side_len=config.text_det_limit_side_len,
        text_det_limit_type=config.text_det_limit_type,
        text_det_thresh=config.text_det_thresh,
        text_det_box_thresh=config.text_det_box_thresh,
        text_det_unclip_ratio=config.text_det_unclip_ratio,
        text_rec_score_thresh=config.text_rec_score_thresh,
        text_recognition_batch_size=config.text_recognition_batch_size,
        enable_mkldnn=config.enable_mkldnn,
        cpu_threads=config.cpu_threads,
        use_tensorrt=config.use_tensorrt,
        precision=config.precision,
    )
