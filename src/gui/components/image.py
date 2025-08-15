import tkinter as tk
from tkinter import ttk
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.gui.utils import create_button, create_labeled_frame


class ImagePanel:
    """Panel for image display with zoom and pan functionality"""

    def __init__(self, parent: tk.Widget, app: Any) -> None:
        self.parent = parent
        self.app = app

        self._initialize_state()
        self._create_frame()

    def update_image(self, image: np.ndarray) -> None:
        """Update displayed image"""
        if image is None:
            self._clear_display()
            return

        try:
            self._update_image_info(image)
            self._display_image(image)

        except Exception as exception:
            self.status_label.config(text=f"Display error: {exception}", foreground="red")

    def zoom_in(self) -> None:
        """Zoom in on image"""
        old_zoom = self.zoom_factor
        self.zoom_factor = min(self.zoom_factor * 1.2, 5.0)

        if self.zoom_factor != old_zoom and self.app.processed_image is not None:
            self._display_image(self.app.processed_image)

    def zoom_out(self) -> None:
        """Zoom out on image"""
        old_zoom = self.zoom_factor
        self.zoom_factor = max(self.zoom_factor / 1.2, 0.3)

        if self.zoom_factor != old_zoom and self.app.processed_image is not None:
            self._display_image(self.app.processed_image)

    def reset_zoom(self) -> None:
        """Reset zoom and pan to defaults"""
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0

        if self.app.processed_image is None:
            return

        # Auto-fit to window on reset
        self.fit_to_window()

    def fit_to_window(self) -> None:
        """Fit image to window size"""
        if self.app.processed_image is None:
            return

        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width() or 600
        canvas_height = self.canvas.winfo_height() or 400

        img_h, img_w = self.app.processed_image.shape[:2]

        # Calculate zoom to fit image in canvas
        margin = 50
        zoom_w = (canvas_width - margin) / img_w
        zoom_h = (canvas_height - margin) / img_h

        self.zoom_factor = min(zoom_w, zoom_h, 1.0)
        self.pan_x = 0
        self.pan_y = 0

        self._display_image(self.app.processed_image)

    def _initialize_state(self) -> None:
        """Initialize image display state"""
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.photo_ref = None
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.image_center_x = 0
        self.image_center_y = 0

    def _create_frame(self) -> None:
        """Create image frame"""
        self.frame = create_labeled_frame(self.parent, "🖼️ Image Display")

        self._create_toolbar()
        self._create_canvas()

    def _create_toolbar(self) -> None:
        """Create image toolbar with controls"""
        toolbar_frame = ttk.Frame(self.frame)
        toolbar_frame.pack(fill=tk.X, pady=(0, 10))

        self._create_status_section(toolbar_frame)
        self._create_zoom_controls(toolbar_frame)

    def _create_canvas(self) -> None:
        """Create image canvas with scrollbars"""
        canvas_frame = ttk.Frame(self.frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg="white", highlightthickness=1, highlightbackground="#ddd")

        h_scrollbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)

        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")

        self._bind_canvas_events()

    def _bind_canvas_events(self) -> None:
        """Bind mouse events to canvas"""
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)

        self.canvas.bind("<Motion>", self._on_mouse_move)

        self.canvas.bind("<Button-1>", self._start_drag)
        self.canvas.bind("<B1-Motion>", self._do_drag)
        self.canvas.bind("<ButtonRelease-1>", self._stop_drag)

        self.canvas.bind("<Double-Button-1>", self._on_double_click)

        self.canvas.bind("<Enter>", self._on_canvas_enter)
        self.canvas.bind("<Leave>", self._on_canvas_leave)

    def _on_mouse_wheel(self, event) -> None:
        """Handle mouse wheel for zooming"""
        if self.app.processed_image is None:
            return

        old_zoom = self.zoom_factor

        if event.delta > 0 or event.num == 4:
            self.zoom_factor = min(self.zoom_factor * 1.1, 5.0)
        else:
            self.zoom_factor = max(self.zoom_factor / 1.1, 0.3)

        if self.zoom_factor != old_zoom:
            self._display_image(self.app.processed_image)

    def _on_mouse_move(self, event) -> None:
        """Handle mouse movement to show cursor position"""
        if self.app.processed_image is None:
            self.cursor_position_label.config(text="", foreground="gray")
            return

        # Get canvas coordinates
        canvas_x = event.x
        canvas_y = event.y

        # Convert to image coordinates
        image_x, image_y = self._canvas_to_image_coordinates(canvas_x, canvas_y)

        # Get image dimensions
        img_height, img_width = self.app.processed_image.shape[:2]

        # Show coordinates only when within image bounds
        if image_x >= 0 and image_y >= 0 and image_x < img_width and image_y < img_height:
            self.cursor_position_label.config(text=f"Cursor: ({image_x}, {image_y})", foreground="purple")
        else:
            self.cursor_position_label.config(text="", foreground="purple")

    def _canvas_to_image_coordinates(self, canvas_x: int, canvas_y: int) -> tuple[int, int]:
        """Convert canvas coordinates to image coordinates"""
        if self.app.processed_image is None:
            return 0, 0

        # Get displayed image dimensions
        display_image = self._prepare_display_image(self.app.processed_image)
        if display_image is None:
            return 0, 0

        img_h, img_w = display_image.shape[:2]

        # Calculate image boundaries on canvas
        image_left = self.image_center_x + self.pan_x - img_w // 2
        image_right = image_left + img_w
        image_top = self.image_center_y + self.pan_y - img_h // 2
        image_bottom = image_top + img_h

        # Check if cursor is over the image
        if image_left <= canvas_x <= image_right and image_top <= canvas_y <= image_bottom:
            # Calculate relative position within displayed image
            relative_x = canvas_x - image_left
            relative_y = canvas_y - image_top

            # Convert to original image coordinates
            image_x = int(relative_x / self.zoom_factor)
            image_y = int(relative_y / self.zoom_factor)

            return image_x, image_y
        else:
            # Cursor outside image
            return -1, -1

    def _on_canvas_enter(self, event) -> None:
        """Handle mouse entering canvas"""
        self.canvas.configure(cursor="crosshair")

    def _on_canvas_leave(self, event) -> None:
        """Handle mouse leaving canvas"""
        self.canvas.configure(cursor="")
        self.cursor_position_label.config(text="", foreground="purple")

    def _start_drag(self, event) -> None:
        """Start dragging operation"""
        self.canvas.focus_set()
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.canvas.configure(cursor="fleur")

    def _do_drag(self, event) -> None:
        """Perform drag operation"""
        if hasattr(self, "drag_start_x"):
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y

            self.pan_x += dx
            self.pan_y += dy

        self.drag_start_x = event.x
        self.drag_start_y = event.y

        if self.app.processed_image is not None:
            self._display_image(self.app.processed_image)

    def _stop_drag(self, event) -> None:
        """Stop dragging operation"""
        self.canvas.configure(cursor="crosshair")

    def _on_double_click(self, event):
        """Handle double click to reset zoom and pan"""
        self.reset_zoom()

    def _create_status_section(self, parent: ttk.Frame) -> None:
        """Create status information section"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.status_label = ttk.Label(status_frame, text="No image loaded", foreground="gray")
        self.status_label.pack(side=tk.LEFT)

        self.image_info_label = ttk.Label(status_frame, text="", foreground="blue")
        self.image_info_label.pack(side=tk.LEFT, padx=(20, 0))

        self.cursor_position_label = ttk.Label(status_frame, text="", foreground="purple")
        self.cursor_position_label.pack(side=tk.LEFT, padx=(20, 0))

    def _create_zoom_controls(self, parent: ttk.Frame) -> None:
        """Create zoom control buttons"""
        zoom_frame = ttk.Frame(parent)
        zoom_frame.pack(side=tk.RIGHT)

        self.zoom_label = ttk.Label(zoom_frame, text="100%", width=8)
        self.zoom_label.pack(side=tk.LEFT, padx=5)

        create_button(zoom_frame, "🔍-", self.zoom_out).pack(side=tk.LEFT, padx=2)
        create_button(zoom_frame, "🔍+", self.zoom_in).pack(side=tk.LEFT, padx=2)
        create_button(zoom_frame, "🔄", self.reset_zoom).pack(side=tk.LEFT, padx=2)
        create_button(zoom_frame, "📐", self.fit_to_window).pack(side=tk.LEFT, padx=2)

    def _clear_display(self) -> None:
        """Clear image display"""
        self.canvas.delete("all")
        self.photo_ref = None
        self.status_label.config(text="No image loaded", foreground="gray")
        self.image_info_label.config(text="")
        self.cursor_position_label.config(text="", foreground="gray")

    def _update_image_info(self, image: np.ndarray) -> None:
        """Update image information display"""
        h, w = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1

        # Detect image type
        unique_values = np.unique(image)
        is_binary = len(unique_values) == 2 and (unique_values == [0, 255]).all()

        if channels == 1:
            if is_binary:
                info_text = f"{w}×{h}px (Binary)"
            else:
                info_text = f"{w}×{h}px (Grayscale)"
        else:
            if is_binary:
                info_text = f"{w}×{h}px (Binary, {channels} channels)"
            else:
                info_text = f"{w}×{h}px ({channels} channels)"

        self.image_info_label.config(text=info_text, foreground="blue")
        self.status_label.config(text="Image loaded", foreground="green")
        self.cursor_position_label.config(text="", foreground="purple")

    def _display_image(self, image: np.ndarray) -> None:
        """Display image on canvas with clean positioning"""
        display_image = self._prepare_display_image(image)
        if display_image is None:
            return

        # Convert to PIL and create PhotoImage
        pil_image = Image.fromarray(display_image)
        self.photo_ref = ImageTk.PhotoImage(pil_image)

        # Clear canvas
        self.canvas.delete("all")

        # Get dimensions
        canvas_width = self.canvas.winfo_width() or 600
        canvas_height = self.canvas.winfo_height() or 400
        img_h, img_w = display_image.shape[:2]

        # Calculate image center on canvas
        self.image_center_x = canvas_width // 2
        self.image_center_y = canvas_height // 2

        # Calculate pan limits
        max_pan_x = max(0, (img_w - canvas_width) // 2)
        max_pan_y = max(0, (img_h - canvas_height) // 2)

        # Apply pan limits
        self.pan_x = max(-max_pan_x, min(max_pan_x, self.pan_x))
        self.pan_y = max(-max_pan_y, min(max_pan_y, self.pan_y))

        # Position image
        image_x = self.image_center_x + self.pan_x
        image_y = self.image_center_y + self.pan_y

        self.canvas.create_image(image_x, image_y, image=self.photo_ref, anchor=tk.CENTER)

        # Update scroll region and zoom display
        self._update_scroll_region(display_image, image_x, image_y)
        self._update_zoom_display()

    def _prepare_display_image(self, image: np.ndarray) -> np.ndarray | None:
        """Prepare image for display with zoom"""
        if len(image.shape) == 2:
            display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.zoom_factor == 1.0:
            return display_image

        orig_h, orig_w = display_image.shape[:2]
        new_w = max(1, int(orig_w * self.zoom_factor))
        new_h = max(1, int(orig_h * self.zoom_factor))

        max_dimension = 8000
        if new_w > max_dimension or new_h > max_dimension:
            scale = min(max_dimension / new_w, max_dimension / new_h)
            new_w = int(new_w * scale)
            new_h = int(new_h * scale)
            self.zoom_factor *= scale

        if self.zoom_factor > 1.0:
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_AREA

        return cv2.resize(display_image, (new_w, new_h), interpolation=interpolation)

    def _update_scroll_region(self, image: np.ndarray, image_x: int, image_y: int) -> None:
        """Update canvas scroll region"""
        img_h, img_w = image.shape[:2]
        margin = 50

        # Scroll region covers the image area with margin
        self.canvas.configure(
            scrollregion=(
                image_x - img_w // 2 - margin,
                image_y - img_h // 2 - margin,
                image_x + img_w // 2 + margin,
                image_y + img_h // 2 + margin,
            )
        )

    def _update_zoom_display(self) -> None:
        """Update zoom percentage display"""
        zoom_percent = int(self.zoom_factor * 100)
        self.zoom_label.config(text=f"{zoom_percent}%")
