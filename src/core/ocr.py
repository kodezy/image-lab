from typing import Any, Protocol

import cv2
import easyocr
import numpy as np
import pytesseract
from paddleocr import PaddleOCR
from rapidocr import LangDet, LangRec, ModelType, OCRVersion, RapidOCR

from src.config import EasyOCRConfig, OCRConfig, PaddleOCRConfig, RapidOCRConfig, TesseractConfig
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
                "Tesseract is not installed or not in PATH.\nInstall Tesseract and ensure it is in the system PATH.",
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        config_parts = [f"--psm {self._config.psm}", f"--oem {self._config.oem}"]

        if self._config.config:
            config_parts.append(self._config.config)

        config_str = " ".join(config_parts)

        try:
            data = pytesseract.image_to_data(
                gray,
                lang=self._config.lang,
                config=config_str,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exception:
            raise RuntimeError(f"Error executing Tesseract: {exception}") from exception

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


def _to_lang_det(value: str) -> LangDet:
    return {"ch": LangDet.CH, "en": LangDet.EN, "multi": LangDet.MULTI}[value]


def _to_lang_rec(value: str) -> LangRec:
    return {"ch": LangRec.CH, "en": LangRec.EN}[value]


def _to_ocr_version(value: str) -> OCRVersion:
    return {"PP-OCRv4": OCRVersion.PPOCRV4, "PP-OCRv5": OCRVersion.PPOCRV5}[value]


def _to_model_type(value: str) -> ModelType:
    return {"mobile": ModelType.MOBILE, "server": ModelType.SERVER}[value]


class RapidOCRWrapper:
    def __init__(self, config: RapidOCRConfig) -> None:
        self._config = config
        rec_lang = LangRec.CH if config.lang_type == "multi" else _to_lang_rec(config.lang_type)
        self._engine = RapidOCR(
            params={
                "Det.lang_type": _to_lang_det(config.lang_type),
                "Rec.lang_type": rec_lang,
                "Det.ocr_version": _to_ocr_version(config.ocr_version),
                "Rec.ocr_version": _to_ocr_version(config.ocr_version),
                "Det.model_type": _to_model_type(config.model_type),
                "Rec.model_type": _to_model_type(config.model_type),
                "Global.use_det": config.use_det,
                "Global.use_cls": config.use_cls,
                "Global.use_rec": config.use_rec,
                "Global.text_score": config.text_score,
                "Global.min_height": config.min_height,
                "Global.width_height_ratio": config.width_height_ratio,
                "Global.max_side_len": config.max_side_len,
                "Global.min_side_len": config.min_side_len,
                "Det.limit_side_len": config.limit_side_len,
                "Det.limit_type": config.limit_type,
                "Det.thresh": config.thresh,
                "Det.box_thresh": config.box_thresh,
                "Det.max_candidates": config.max_candidates,
                "Det.unclip_ratio": config.unclip_ratio,
                "Det.use_dilation": config.use_dilation,
                "Det.score_mode": config.score_mode,
                "Cls.cls_batch_num": config.cls_batch_num,
                "Cls.cls_thresh": config.cls_thresh,
                "Rec.rec_batch_num": config.rec_batch_num,
            },
        )

    def predict(self, image: np.ndarray) -> Any:
        result = self._engine(image)

        if result is None or not hasattr(result, "txts"):
            return [{"rec_texts": [], "rec_scores": [], "det_boxes": []}]

        texts = list(result.txts) if result.txts else []
        scores = list(result.scores) if result.scores else []
        boxes = [box.tolist() for box in result.boxes] if hasattr(result, "boxes") and result.boxes is not None else []

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

    if config.ocr_type == "rapidocr":
        return RapidOCRWrapper(config.rapidocr_config)

    return PaddleOCRWrapper(config.paddleocr_config)
