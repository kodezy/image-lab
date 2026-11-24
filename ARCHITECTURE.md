# Architecture

Image Lab is a desktop Tkinter app for tuning an image preprocessing pipeline and checking its effect on OCR output. All state is in-process; there is no server, database, or network API.

## Layers

Dependencies point down only: `gui` → `core` → `infra`, and all three read `config`.

| Layer | Path | Owns |
| --- | --- | --- |
| Config | `src/config.py` | Dataclasses for every setting: `CaptureConfig`, `ProcessingConfig`, `OCRConfig` (with one nested config per engine). Also the on-disk preset schema. |
| Core | `src/core/` | Pure image and OCR logic. No Tkinter. `processing.py` (pipeline), `ocr.py` (engine wrappers), `capture.py` (screen grab). |
| Infra | `src/infra/` | `io.py` (image and JSON files, clipboard) and `cache.py` (memoization decorators). |
| GUI | `src/gui/` | `main.py` (`ImageLabGUI`, the entry point `image-lab`) wires the panels in `components/`; `utils.py` has widget factories. |

## Contracts

- **Images are `numpy` arrays in BGR order** (OpenCV convention), optionally with a 4th alpha channel. `load_image` decodes with OpenCV (3 channels, no alpha); clipboard and screen capture convert from PIL's RGB. Transparency exists only in memory: saving to a non-PNG suffix flattens alpha onto white.
- **Pipeline order is fixed** in `process_image`: input normalization → trim borders → resize, then a step list. Crop runs before the steps, unless background blur or mask is enabled, in which case it runs after the steps and the background effect. Each step is `(image, ProcessingConfig) -> image` and reads only its own fields.
- **OCR engines share `OCRProtocol.predict(image)`.** The result is a list whose first item is a dict with `rec_texts` and `rec_scores` (the PaddleOCR shape; confidence in 0–1). The OCR panel reads only those two keys. Tesseract, EasyOCR, and RapidOCR wrappers convert their output into this shape (and add `det_boxes`); the PaddleOCR wrapper passes its native result through. `create_ocr` picks the wrapper from `OCRConfig.ocr_type`.
- **Engine instances are expensive and cached twice.** The GUI keeps one in `ocr_instance` and builds it lazily; the OCR panel and preset loading reset it when settings change. `create_ocr` is also memoized by config. Engine load time is reported separately from run time.
- **`process_image` is memoized** (`image_cache`, 128 entries) on the image bytes and the pickled `ProcessingConfig`, and returns copies. `infra/cache.py` evicts the oldest entry first and holds everything in memory. A config field that does not pickle stably, or a step that depends on state outside `(image, config)`, makes results stale or the cache useless.
- **Presets are JSON** written by `save_json` as `{"capture": …, "ocr": …, "processing": …}`. Loading goes through `update_from_dict`, which ignores unknown keys, so older presets keep loading when fields are added. Renaming or retyping a field silently drops it from existing presets.

## Decisions

- **Four OCR engines (PaddleOCR, Tesseract, EasyOCR, RapidOCR)** live side by side because the tool exists to compare them on the same preprocessed image.
- **Config is plain dataclasses, with no GUI state in them**, so the same object drives the GUI, presets, and the cache key.

## Not covered

No automated tests, CI, or type checker are configured. `ruff` settings are in `ruff.toml`.
