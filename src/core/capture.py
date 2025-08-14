import cv2
import numpy as np
import pyautogui
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
    try:
        cap = cv2.VideoCapture(config.device_id, cv2.CAP_DSHOW)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open capture device {config.device_id}")

        screen_width, screen_height = _get_screen_resolution()

        if screen_width > config.device_max_width or screen_height > config.device_max_height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.device_max_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.device_max_height)

        cap.set(cv2.CAP_PROP_FPS, 30)

        ret, frame = cap.read()

        if not ret:
            raise RuntimeError("Failed to grab frame from capture device")

        return frame

    finally:
        if cap is not None:
            cap.release()


def _get_screen_resolution() -> tuple[int, int]:
    """Get screen resolution"""
    return pyautogui.size()
