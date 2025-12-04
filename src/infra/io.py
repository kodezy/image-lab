import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageGrab


def save_image(image: np.ndarray, filename: str) -> bool:
    """Save image to file"""
    try:
        path = Path(filename)

        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        is_success, encoded_img = cv2.imencode(path.suffix or ".png", image)

        if is_success:
            with open(filename, "wb") as file:
                file.write(encoded_img.tobytes())

            return True

        return False

    except Exception:
        return False


def load_image(filename: str) -> np.ndarray | None:
    """Load image from file"""
    try:
        path = Path(filename)

        if not path.exists():
            return None

        with open(filename, "rb") as file:
            file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            return image

    except Exception:
        return None


def load_image_from_clipboard() -> np.ndarray | None:
    """Load image from clipboard"""
    try:
        clipboard_image = ImageGrab.grabclipboard()

        if clipboard_image is None:
            return None

        if not isinstance(clipboard_image, Image.Image):
            return None

        image_array = np.array(clipboard_image)
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    except Exception:
        return None


def save_json(configs: dict[str, Any], filename: str) -> bool:
    """Save JSON file"""
    try:
        data = {key: asdict(config) for key, config in configs.items()}

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return True

    except Exception:
        return False


def load_json(filename: str) -> dict[str, Any] | None:
    """Load JSON file"""
    try:
        with open(filename, encoding="utf-8") as f:
            return json.load(f)

    except Exception:
        return None
