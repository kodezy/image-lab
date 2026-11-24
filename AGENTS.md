# Image Lab

Tkinter app for tuning image preprocessing and comparing OCR engines. Python 3.13, `src/` layout, installed as `image-lab`.

## How to work here

- Layers are `gui` → `core` → `infra`, with `src/config.py` shared. Keep Tkinter out of `core` and `infra`.
- Adding a processing option touches three places: the field in `ProcessingConfig`, the step in `src/core/processing.py`, and its widget in `src/gui/components/processing.py`. OCR options follow the same shape through `src/core/ocr.py` and `src/gui/components/ocr.py`.
- `src/gui/components/processing.py` and `ocr.py` are large; read the section you need, not the whole file.

## Commands

Use the existing `.venv` (`source .venv/bin/activate`).

- `pip install -e .` installs the app and its heavy OCR dependencies (Paddle, EasyOCR, ONNX). Avoid re-running it without a reason.
- `image-lab` starts the GUI. It needs a display, and the first run of an OCR engine may download models.
- `ruff check src` and `ruff format --check src` use `line-length=120` from `ruff.toml`. Today `check` reports 36 findings and `format` would change 2 files, so compare against that baseline instead of expecting zero.

There is no test suite. Verify core changes by calling `process_image` or `create_ocr` from a Python snippet; verify GUI changes by running the app.

## Easy to forget

- Images are BGR `numpy` arrays. Convert at the edges only, as `infra/io.py` does.
- `Config.update_from_dict` ignores unknown keys. Renaming or retyping a config field silently breaks saved presets, so keep old names working or say so.
- `process_image` and `create_ocr` are memoized (`src/infra/cache.py`). Steps must depend only on `(image, config)`; any new config field must pickle stably.
- The pipeline order in `process_image` is fixed and documented in `ARCHITECTURE.md`; do not reorder steps casually.
- `pytesseract` needs the Tesseract binary installed on the system, not only the pip package.
- Camera capture uses `cv2.CAP_DSHOW` (Windows backend), although the README lists macOS and Linux. Screen capture goes through PIL and is portable.

## Documents

- [ARCHITECTURE.md](ARCHITECTURE.md): layers, image and OCR contracts, preset format. Read before changing `config.py`, the pipeline, or an OCR wrapper.
- `README.md`: user-facing install and usage only.

**When to update:** change `ARCHITECTURE.md` when a layer boundary, the image format, the OCR result shape, or the preset schema changes. Update the Commands here when tests or CI are added.

## Definition of done

`ruff check` reports nothing new on touched files, and the changed path was exercised: a snippet for `core`, a launch of `image-lab` for GUI work.

## Local instructions

- `.agents/`, `.claude/`, `.cursor/`, `.brantz-lock.json`, `.nexus/`, and `.superpowers/` are local agent tooling, not project code; do not edit or document them.
- `docs/superpowers/plans` and `docs/superpowers/specs` stay on disk; version them only when the user says so.
