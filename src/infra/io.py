import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageGrab


def save_image(image: np.ndarray, filename: str) -> bool:
    try:
        path = Path(filename)

        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        suffix = path.suffix or ".png"
        is_success, encoded_img = cv2.imencode(suffix, image)

        if not is_success:
            return False

        path.write_bytes(encoded_img.tobytes())
        return True

    except Exception:
        return False


def load_image(filename: str) -> np.ndarray | None:
    try:
        path = Path(filename)

        if not path.exists():
            return None

        file_bytes = path.read_bytes()
        image = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        return image

    except Exception:
        return None


def load_image_from_clipboard() -> np.ndarray | None:
    try:
        clipboard_image = ImageGrab.grabclipboard()

        if clipboard_image is None or not isinstance(clipboard_image, Image.Image):
            return None

        image_array = np.array(clipboard_image)
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    except Exception:
        return None


def save_json(configs: dict[str, Any], filename: str) -> bool:
    try:
        data = {key: asdict(config) for key, config in configs.items()}
        Path(filename).write_text(json.dumps(data, indent=2), encoding="utf-8")
        return True

    except Exception:
        return False


def load_json(filename: str) -> dict[str, Any] | None:
    try:
        return json.loads(Path(filename).read_text(encoding="utf-8"))

    except Exception:
        return None
