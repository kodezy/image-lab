from typing import Any, Protocol

import cv2
import easyocr
import numpy as np
import pytesseract
from paddleocr import PaddleOCR

from src.config import EasyOCRConfig, OCRConfig, PaddleOCRConfig, TesseractConfig
from src.infra.cache import ocr_cache


class OCRProtocol(Protocol):
    def predict(self, image: np.ndarray) -> Any: ...


class PaddleOCRWrapper:
    def __init__(self, config: PaddleOCRConfig) -> None:
        self._ocr = PaddleOCR(
            ocr_version=config.ocr_version,
            lang=config.lang,
            device=config.device,
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
            use_doc_orientation_classify=config.use_doc_orientation_classify,
            use_doc_unwarping=config.use_doc_unwarping,
            use_textline_orientation=config.use_textline_orientation,
            text_det_limit_side_len=config.text_det_limit_side_len,
            text_det_limit_type=config.text_det_limit_type,
            text_det_thresh=config.text_det_thresh,
            text_det_box_thresh=config.text_det_box_thresh,
            text_det_unclip_ratio=config.text_det_unclip_ratio,
            text_det_input_shape=config.text_det_input_shape,
            text_rec_score_thresh=config.text_rec_score_thresh,
            text_recognition_batch_size=config.text_recognition_batch_size,
            text_rec_input_shape=config.text_rec_input_shape,
            textline_orientation_batch_size=config.textline_orientation_batch_size,
            enable_hpi=config.enable_hpi,
            enable_mkldnn=config.enable_mkldnn,
            mkldnn_cache_capacity=config.mkldnn_cache_capacity,
            cpu_threads=config.cpu_threads,
            use_tensorrt=config.use_tensorrt,
            precision=config.precision,
        )

    def predict(self, image: np.ndarray) -> Any:
        return self._ocr.predict(image)


class TesseractOCRWrapper:
    def __init__(self, config: TesseractConfig) -> None:
        self._config = config

    def predict(self, image: np.ndarray) -> Any:
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            raise RuntimeError(
                "Tesseract não está instalado ou não está no PATH.\n"
                "Instale o Tesseract e certifique-se de que está no PATH do sistema.",
            )

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        config_parts = []

        config_parts.append(f"--psm {self._config.psm}")
        config_parts.append(f"--oem {self._config.oem}")

        if self._config.config:
            config_parts.append(self._config.config)

        config_str = " ".join(config_parts) if config_parts else None

        try:
            data = pytesseract.image_to_data(
                gray,
                lang=self._config.lang,
                config=config_str,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exception:
            raise RuntimeError(f"Erro ao executar Tesseract: {exception}") from exception

        texts = []
        scores = []
        boxes = []

        for i in range(len(data["text"])):
            text = data["text"][i].strip()

            if not text:
                continue

            conf = float(data["conf"][i]) if data["conf"][i] != -1 else 0.0
            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            texts.append(text)
            scores.append(conf / 100.0)
            boxes.append([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

        return [
            {
                "rec_texts": texts,
                "rec_scores": scores,
                "det_boxes": boxes,
            },
        ]


class EasyOCRWrapper:
    def __init__(self, config: EasyOCRConfig) -> None:
        self._config = config
        self._reader = easyocr.Reader(
            lang_list=config.lang_list,
            gpu=config.gpu,
            model_storage_directory=config.model_storage_directory,
            download_enabled=config.download_enabled,
            user_network_directory=config.user_network_directory,
            recog_network=config.recog_network,
            detector=config.detector,
            recognizer=config.recognizer,
        )

    def predict(self, image: np.ndarray) -> Any:
        results = self._reader.readtext(
            image,
            decoder=self._config.decoder,
            beamWidth=self._config.beam_width,
            batch_size=self._config.batch_size,
            workers=self._config.workers,
            allowlist=self._config.allowlist,
            blocklist=self._config.blocklist,
            detail=self._config.detail,
            paragraph=self._config.paragraph,
            min_size=self._config.min_size,
            rotation_info=self._config.rotation_info,
            contrast_ths=self._config.contrast_ths,
            adjust_contrast=self._config.adjust_contrast,
            text_threshold=self._config.text_threshold,
            low_text=self._config.low_text,
            link_threshold=self._config.link_threshold,
            canvas_size=self._config.canvas_size,
            mag_ratio=self._config.mag_ratio,
            slope_ths=self._config.slope_ths,
            ycenter_ths=self._config.ycenter_ths,
            height_ths=self._config.height_ths,
            width_ths=self._config.width_ths,
            add_margin=self._config.add_margin,
            x_ths=self._config.x_ths,
            y_ths=self._config.y_ths,
        )

        texts = []
        scores = []
        boxes = []

        for result in results:
            if len(result) >= 3:
                box_coords = result[0]
                text = result[1]
                confidence = result[2]

                texts.append(text)
                scores.append(float(confidence))
                boxes.append(box_coords)

        return [
            {
                "rec_texts": texts,
                "rec_scores": scores,
                "det_boxes": boxes,
            },
        ]


@ocr_cache(max_size=16)
def create_ocr(config: OCRConfig | None = None) -> OCRProtocol:
    if config is None:
        config = OCRConfig()

    if config.ocr_type == "tesseract":
        return TesseractOCRWrapper(config.tesseract_config)

    if config.ocr_type == "easyocr":
        return EasyOCRWrapper(config.easyocr_config)

    return PaddleOCRWrapper(config.paddleocr_config)
