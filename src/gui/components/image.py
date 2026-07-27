import sys
import tkinter as tk
from tkinter import ttk
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.gui.utils import create_button, create_labeled_frame

ZOOM_STEP = 1.1
ZOOM_MIN = 0.1
ZOOM_MAX = 5.0
MAX_BITMAP = 8000
FIT_MARGIN = 50
ZOOM_COALESCE_MS = 16


class ImagePanel:
    def __init__(self, parent: tk.Widget, app: Any) -> None:
        self.parent = parent
        self.app = app

        self.zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self._photo: ImageTk.PhotoImage | None = None
        self._image_id: int | None = None
        self._base_rgb: np.ndarray | None = None
        self._zoom_job: str | None = None
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._drag_origin_x = 0.0
        self._drag_origin_y = 0.0
        self._last_canvas_width = 0
        self._last_canvas_height = 0

        self._create_frame()

    def update_image(self, image: np.ndarray) -> None:
        if image is None:
            self._clear_display()
            return
        try:
            self._set_base_image(image)
            self._update_image_info(image)
            self._rebuild_bitmap()
            self._place_image()
        except Exception as exception:
            self.status_label.config(text=f"Display error: {exception}", foreground="red")

    def zoom_in(self) -> None:
        self._zoom_at_canvas_point(self._canvas_center(), ZOOM_STEP)

    def zoom_out(self) -> None:
        self._zoom_at_canvas_point(self._canvas_center(), 1.0 / ZOOM_STEP)

    def reset_zoom(self) -> None:
        if self._base_rgb is None:
            self.zoom = 1.0
            self.offset_x = 0.0
            self.offset_y = 0.0
            return
        self.zoom = self._clamp_zoom(1.0)
        self._rebuild_bitmap()
        self._center_image()
        self._place_image()

    def fit_to_window(self) -> None:
        if self._base_rgb is None:
            return
        canvas_w, canvas_h = self._canvas_size()
        img_h, img_w = self._base_rgb.shape[:2]
        zoom_w = (canvas_w - FIT_MARGIN) / img_w
        zoom_h = (canvas_h - FIT_MARGIN) / img_h
        self.zoom = self._clamp_zoom(min(zoom_w, zoom_h))
        self._rebuild_bitmap()
        self._center_image()
        self._place_image()

    def _create_frame(self) -> None:
        self.frame = create_labeled_frame(self.parent, "🖼️ Image Display")
        self._create_toolbar()
        self._create_canvas()

    def _create_toolbar(self) -> None:
        toolbar_frame = ttk.Frame(self.frame)
        toolbar_frame.pack(fill=tk.X, pady=(0, 10))
        toolbar_frame.grid_columnconfigure(0, weight=1)
        toolbar_frame.grid_columnconfigure(1, weight=0, minsize=250)

        status_container = ttk.Frame(toolbar_frame)
        status_container.grid(row=0, column=0, sticky="ew")
        zoom_container = ttk.Frame(toolbar_frame)
        zoom_container.grid(row=0, column=1, sticky="e")

        status_frame = ttk.Frame(status_container)
        status_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.status_label = ttk.Label(status_frame, text="No image loaded", foreground="gray")
        self.status_label.pack(side=tk.LEFT)
        self.image_info_label = ttk.Label(status_frame, text="", foreground="blue")
        self.image_info_label.pack(side=tk.LEFT, padx=(20, 0))
        self.cursor_label = ttk.Label(status_frame, text="", foreground="gray")
        self.cursor_label.pack(side=tk.LEFT, padx=(20, 0))

        zoom_frame = ttk.Frame(zoom_container)
        zoom_frame.pack(side=tk.RIGHT)
        self.zoom_label = ttk.Label(zoom_frame, text="100%", width=8)
        self.zoom_label.pack(side=tk.LEFT, padx=5)
        create_button(zoom_frame, "🔍-", self.zoom_out).pack(side=tk.LEFT, padx=2)
        create_button(zoom_frame, "🔍+", self.zoom_in).pack(side=tk.LEFT, padx=2)
        create_button(zoom_frame, "🔄", self.reset_zoom).pack(side=tk.LEFT, padx=2)
        create_button(zoom_frame, "📐", self.fit_to_window).pack(side=tk.LEFT, padx=2)

    def _create_canvas(self) -> None:
        canvas_frame = ttk.Frame(self.frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        bg_color = self._get_canvas_bg_color()
        self.canvas = tk.Canvas(canvas_frame, bg=bg_color, highlightthickness=0, borderwidth=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.canvas.bind("<Button-1>", self._start_drag)
        self.canvas.bind("<B1-Motion>", self._do_drag)
        self.canvas.bind("<ButtonRelease-1>", self._stop_drag)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        self.canvas.bind("<Enter>", lambda _e: self.canvas.configure(cursor="crosshair"))
        self.canvas.bind("<Leave>", self._on_canvas_leave)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

    def _get_canvas_bg_color(self) -> str:
        if sys.platform == "darwin":
            try:
                bg_color = ttk.Style().lookup("TFrame", "background")
                if bg_color:
                    return bg_color
            except Exception:
                pass
        return "white"

    def _set_base_image(self, image: np.ndarray) -> None:
        if len(image.shape) == 2:
            self._base_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            self._base_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _canvas_size(self) -> tuple[int, int]:
        return self.canvas.winfo_width() or 600, self.canvas.winfo_height() or 400

    def _canvas_center(self) -> tuple[float, float]:
        w, h = self._canvas_size()
        return w / 2.0, h / 2.0

    def _max_zoom_for_image(self) -> float:
        if self._base_rgb is None:
            return ZOOM_MAX
        img_h, img_w = self._base_rgb.shape[:2]
        return min(ZOOM_MAX, MAX_BITMAP / max(img_w, img_h))

    def _clamp_zoom(self, zoom: float) -> float:
        return max(ZOOM_MIN, min(self._max_zoom_for_image(), zoom))

    def _rebuild_bitmap(self) -> None:
        if self._base_rgb is None:
            return
        img_h, img_w = self._base_rgb.shape[:2]
        new_w = max(1, int(round(img_w * self.zoom)))
        new_h = max(1, int(round(img_h * self.zoom)))
        interpolation = cv2.INTER_CUBIC if self.zoom > 1.0 else cv2.INTER_AREA
        display = cv2.resize(self._base_rgb, (new_w, new_h), interpolation=interpolation)
        self._photo = ImageTk.PhotoImage(Image.fromarray(display))
        if self._image_id is None:
            self._image_id = self.canvas.create_image(self.offset_x, self.offset_y, image=self._photo, anchor=tk.NW)
        else:
            self.canvas.itemconfig(self._image_id, image=self._photo)
        self._update_zoom_display()

    def _place_image(self) -> None:
        if self._image_id is None:
            return
        self.canvas.coords(self._image_id, self.offset_x, self.offset_y)

    def _center_image(self) -> None:
        if self._base_rgb is None:
            return
        canvas_w, canvas_h = self._canvas_size()
        img_h, img_w = self._base_rgb.shape[:2]
        self.offset_x = (canvas_w - img_w * self.zoom) / 2.0
        self.offset_y = (canvas_h - img_h * self.zoom) / 2.0

    def _zoom_at_canvas_point(self, point: tuple[float, float], factor: float) -> None:
        if self._base_rgb is None:
            return
        cx, cy = point
        img_x = (cx - self.offset_x) / self.zoom
        img_y = (cy - self.offset_y) / self.zoom
        new_zoom = self._clamp_zoom(self.zoom * factor)
        if new_zoom == self.zoom:
            return
        self.zoom = new_zoom
        self.offset_x = cx - img_x * self.zoom
        self.offset_y = cy - img_y * self.zoom
        self._schedule_zoom_rebuild()

    def _schedule_zoom_rebuild(self) -> None:
        if self._zoom_job is not None:
            self.canvas.after_cancel(self._zoom_job)
        self._zoom_job = self.canvas.after(ZOOM_COALESCE_MS, self._apply_zoom_rebuild)

    def _apply_zoom_rebuild(self) -> None:
        self._zoom_job = None
        self._rebuild_bitmap()
        self._place_image()

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        if self._base_rgb is None:
            return
        if sys.platform == "darwin":
            zoom_in = event.delta > 0
        elif getattr(event, "num", None) in (4, 5):
            zoom_in = event.num == 4
        else:
            zoom_in = event.delta > 0
        factor = ZOOM_STEP if zoom_in else 1.0 / ZOOM_STEP
        cx = float(self.canvas.canvasx(event.x))
        cy = float(self.canvas.canvasy(event.y))
        self._zoom_at_canvas_point((cx, cy), factor)

    def _on_mouse_move(self, event: tk.Event) -> None:
        if self._base_rgb is None:
            self.cursor_label.config(text="")
            return
        cx = float(self.canvas.canvasx(event.x))
        cy = float(self.canvas.canvasy(event.y))
        img_x = int((cx - self.offset_x) / self.zoom)
        img_y = int((cy - self.offset_y) / self.zoom)
        img_h, img_w = self._base_rgb.shape[:2]
        if 0 <= img_x < img_w and 0 <= img_y < img_h:
            self.cursor_label.config(text=f"({img_x}, {img_y})")
        else:
            self.cursor_label.config(text="")

    def _on_canvas_leave(self, _event: tk.Event) -> None:
        self.canvas.configure(cursor="")
        self.cursor_label.config(text="")

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        canvas_w, canvas_h = self._canvas_size()
        if canvas_w <= 1 or canvas_h <= 1:
            return
        if self._last_canvas_width == canvas_w and self._last_canvas_height == canvas_h:
            return
        old_w = self._last_canvas_width
        old_h = self._last_canvas_height
        self._last_canvas_width = canvas_w
        self._last_canvas_height = canvas_h
        if self._base_rgb is None or old_w <= 1 or old_h <= 1:
            return
        img_x = ((old_w / 2.0) - self.offset_x) / self.zoom
        img_y = ((old_h / 2.0) - self.offset_y) / self.zoom
        self.offset_x = (canvas_w / 2.0) - img_x * self.zoom
        self.offset_y = (canvas_h / 2.0) - img_y * self.zoom
        self._place_image()

    def _start_drag(self, event: tk.Event) -> None:
        self.canvas.focus_set()
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        self._drag_origin_x = self.offset_x
        self._drag_origin_y = self.offset_y
        self.canvas.configure(cursor="fleur")

    def _do_drag(self, event: tk.Event) -> None:
        self.offset_x = self._drag_origin_x + (event.x - self._drag_start_x)
        self.offset_y = self._drag_origin_y + (event.y - self._drag_start_y)
        self._place_image()

    def _stop_drag(self, _event: tk.Event) -> None:
        self.canvas.configure(cursor="crosshair")

    def _on_double_click(self, _event: tk.Event) -> None:
        self.fit_to_window()

    def _clear_display(self) -> None:
        if self._zoom_job is not None:
            self.canvas.after_cancel(self._zoom_job)
            self._zoom_job = None
        self.canvas.delete("all")
        self._photo = None
        self._image_id = None
        self._base_rgb = None
        self.zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.status_label.config(text="No image loaded", foreground="gray")
        self.image_info_label.config(text="")
        self.cursor_label.config(text="")
        self.zoom_label.config(text="100%")

    def _update_image_info(self, image: np.ndarray) -> None:
        h, w = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        unique_values = np.unique(image)
        is_binary = len(unique_values) == 2 and set(unique_values) == {0, 255}
        if channels == 1:
            info_text = f"{w}×{h}px (Binary)" if is_binary else f"{w}×{h}px (Grayscale)"
        elif is_binary:
            info_text = f"{w}×{h}px (Binary, {channels} channels)"
        else:
            info_text = f"{w}×{h}px ({channels} channels)"
        self.image_info_label.config(text=info_text, foreground="blue")
        self.status_label.config(text="Image loaded", foreground="green")

    def _update_zoom_display(self) -> None:
        self.zoom_label.config(text=f"{int(self.zoom * 100)}%")
