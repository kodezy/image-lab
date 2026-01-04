import cv2
import numpy as np
from PIL import ImageGrab

from src.config import CaptureConfig


def capture_image(config: CaptureConfig | None = None) -> np.ndarray:
    """Capture image"""
    if config is None:
        config = CaptureConfig()

    if config.device_enabled:
        image = _capture_device(config)
    else:
        image = _capture_pil()

    return image


def _capture_pil() -> np.ndarray:
    """Capture image using PIL"""
    screenshot = ImageGrab.grab()
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)


def _capture_device(config: CaptureConfig) -> np.ndarray:
    """Capture image from device"""
    cap = None
    try:
        cap = cv2.VideoCapture(config.device_id, cv2.CAP_DSHOW)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open capture device {config.device_id}")

        ret, frame = cap.read()

        if not ret:
            raise RuntimeError("Failed to grab frame from capture device")

        return frame

    finally:
        if cap is not None:
            cap.release()
