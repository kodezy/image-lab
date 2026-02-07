import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.config import CaptureConfig, OCRConfig, ProcessingConfig
from src.core.capture import capture_image
from src.core.ocr import OCRProtocol, create_ocr
from src.core.processing import process_image
from src.gui.components.capture import CapturePanel
from src.gui.components.image import ImagePanel
from src.gui.components.menu import MenuBar
from src.gui.components.ocr import OCRPanel
from src.gui.components.processing import ProcessingPanel
from src.gui.utils import get_padding, show_error, show_success
from src.infra.io import load_image, load_image_from_clipboard, load_json, save_image, save_json


def get_icon_path() -> Path | None:
    project_root = Path(__file__).parent.parent.parent
    icon_path = project_root / "assets" / "app.ico"
    return icon_path if icon_path.exists() else None


class ImageLabGUI:
    """Main GUI application for Image Lab"""

    def __init__(self, window_width: int = 1000, window_height: int = 600) -> None:
        self._window_width = window_width
        self._window_height = window_height

        self.current_image: np.ndarray | None = None
        self.processed_image: np.ndarray | None = None
        self.ocr_instance: OCRProtocol | None = None

        self._initialize_window()
        self._initialize_configs()
        self._setup_ui()

    def run(self) -> None:
        """Start the application"""
        try:
            self.root.mainloop()

        except Exception as exception:
            show_error(f"Application error: {exception}")

    def run_ocr(self) -> None:
        """Run OCR on processed image"""
        if self.processed_image is None:
            show_error("No image to process")
            return

        try:
            if self.ocr_instance is None:
                self.ocr_instance = create_ocr(self.ocr_config)

            ocr_image = self._prepare_image_for_ocr(self.processed_image)
            result = self.ocr_instance.predict(ocr_image)

            self.ocr_panel.display_results(result)

        except Exception as exception:
            show_error(f"OCR failed: {exception}")

    def capture_new_image(self) -> None:
        """Capture new image using current config"""
        try:
            self.current_image = capture_image(self.capture_config)
            self.update_image_display()
            self.image_panel.reset_zoom()

        except Exception as exception:
            show_error(f"Capture failed: {exception}")

    def load_image_file(self, filename: str | None = None) -> None:
        """Load image from file"""
        if filename is None:
            filename = filedialog.askopenfilename(
                title="Load Image",
                filetypes=[
                    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                    ("All files", "*.*"),
                ],
            )

        if not filename:
            return

        try:
            image = load_image(filename)

            if image is not None:
                self.current_image = image
                self.update_image_display()
                self.image_panel.reset_zoom()
            else:
                show_error("Failed to load image")

        except Exception as exception:
            show_error(f"Load failed: {exception}")

    def load_image_from_clipboard(self) -> None:
        """Load image from clipboard"""
        try:
            image = load_image_from_clipboard()

            if image is not None:
                self.current_image = image
                self.update_image_display()
                self.image_panel.reset_zoom()
            else:
                show_error("No image found in clipboard")

        except Exception as exception:
            show_error(f"Clipboard load failed: {exception}")

    def save_image_file(self, filename: str | None = None) -> None:
        """Save processed image to file"""
        if self.processed_image is None:
            show_error("No image to save")
            return

        if filename is None:
            filename = filedialog.asksaveasfilename(
                title="Save Image",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("All files", "*.*"),
                ],
            )

        if not filename:
            return

        try:
            if save_image(self.processed_image, filename):
                show_success(f"Image saved: {filename}")
            else:
                show_error("Failed to save image")

        except Exception as exception:
            show_error(f"Save failed: {exception}")

    def save_config_file(self, filename: str | None = None) -> None:
        """Save configurations to file"""
        if filename is None:
            filename = filedialog.asksaveasfilename(
                title="Save Config",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")],
            )

        if not filename:
            return

        try:
            configs = {
                "capture": self.capture_config,
                "ocr": self.ocr_config,
                "processing": self.processing_config,
            }

            if save_json(configs, filename):
                show_success("Configuration saved")
            else:
                show_error("Failed to save configuration")

        except Exception as exception:
            show_error(f"Save config failed: {exception}")

    def load_config_file(self, filename: str | None = None) -> None:
        """Load configurations from file"""
        if filename is None:
            filename = filedialog.askopenfilename(title="Load Config", filetypes=[("JSON files", "*.json")])

        if not filename:
            return

        try:
            data = load_json(filename)

            if data is None:
                show_error("Failed to load configuration")
                return

            for key, config_data in data.items():
                match key:
                    case "capture":
                        self.capture_config.update_from_dict(config_data)
                    case "ocr":
                        self.ocr_config.update_from_dict(config_data)
                    case "processing":
                        self.processing_config.update_from_dict(config_data)

            self._refresh_panels()
            self.ocr_instance = None
            show_success("Configuration loaded")

        except Exception as exception:
            show_error(f"Load config failed: {exception}")

    def reset_configs(self) -> None:
        """Reset all configurations to defaults"""
        if messagebox.askyesno("Reset Configurations", "Reset all configurations to defaults?"):
            self.capture_config = CaptureConfig()
            self.ocr_config = OCRConfig()
            self.processing_config = ProcessingConfig()

            self._refresh_panels()
            self.ocr_instance = None
            show_success("Configurations reset to defaults")

    def update_image_display(self) -> None:
        """Update image display with current processing"""
        if self.current_image is None:
            return

        try:
            self.processed_image = process_image(self.current_image, self.processing_config)
            self.image_panel.update_image(self.processed_image)
            self.processing_panel.sync_controls_from_image()

        except Exception as exception:
            show_error(f"Processing failed: {exception}")

    def _prepare_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Prepare image for OCR processing"""
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        return image

    def _refresh_panels(self) -> None:
        """Refresh all panels with current configurations"""
        self.capture_panel.refresh()
        self.processing_panel.refresh()
        self.ocr_panel.refresh()

        if self.current_image is not None:
            self.update_image_display()

    def _initialize_window(self) -> None:
        """Initialize main window"""
        self.root = tk.Tk()
        self.root.title("Image Lab")
        self.root.minsize(1000, 600)

        self.root.geometry(f"{self._window_width}x{self._window_height}")
        self._center_window()

        self.root.configure(bg="#f8f9fa")
        self._configure_macos_support()
        self._set_window_icon()

    def _configure_macos_support(self) -> None:
        """Configure macOS-specific settings"""
        if sys.platform == "darwin":
            style = ttk.Style()
            style.theme_use("aqua")

            bg_color = style.lookup("TFrame", "background")
            if bg_color:
                self.root.configure(bg=bg_color)

    def _center_window(self) -> None:
        """Center window on screen"""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x = (screen_width - self._window_width) // 2
        y = ((screen_height - self._window_height) // 2) - 50

        self.root.geometry(f"{self._window_width}x{self._window_height}+{x}+{y}")

    def _set_window_icon(self) -> None:
        """Set window icon"""
        icon_path = get_icon_path()
        if not icon_path:
            return

        try:
            if sys.platform == "darwin":
                pil_image = Image.open(str(icon_path))
                icon = ImageTk.PhotoImage(pil_image)
                self.root.iconphoto(True, icon)  # type: ignore
                self._app_icon_image = icon
            elif str(icon_path).lower().endswith(".ico"):
                self.root.iconbitmap(str(icon_path))
            else:
                pil_image = Image.open(str(icon_path))
                icon = ImageTk.PhotoImage(pil_image)
                self.root.iconphoto(True, icon)
                self._app_icon_image = icon
        except Exception:
            pass

    def _initialize_configs(self) -> None:
        """Initialize configuration objects"""
        self.capture_config = CaptureConfig()
        self.ocr_config = OCRConfig()
        self.processing_config = ProcessingConfig()

    def _setup_ui(self) -> None:
        """Setup user interface components"""
        self._create_menu()
        self._create_main_layout()
        self._create_panels()

    def _create_menu(self) -> None:
        """Create menu bar"""
        self.menu_bar = MenuBar(self.root, self)
        self.root.config(menu=self.menu_bar.menu)

    def _create_main_layout(self) -> None:
        """Create main application layout"""
        padding = get_padding()

        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=padding, pady=padding)

        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

    def _create_panels(self) -> None:
        """Create main panels"""
        self._create_control_panel()
        self._create_image_panel()

    def _create_control_panel(self) -> None:
        """Create left control panel with tabs"""
        padding = get_padding()

        self.control_notebook = ttk.Notebook(self.main_frame, width=400)
        self.control_notebook.grid(row=0, column=0, sticky="nsew", padx=(0, padding))

        self.capture_panel = CapturePanel(self.control_notebook, self)
        self.processing_panel = ProcessingPanel(self.control_notebook, self)
        self.ocr_panel = OCRPanel(self.control_notebook, self)

        self.control_notebook.add(self.capture_panel.frame, text="📷 Capture")
        self.control_notebook.add(self.processing_panel.frame, text="🔧 Processing")
        self.control_notebook.add(self.ocr_panel.frame, text="🔍 OCR")

    def _create_image_panel(self) -> None:
        """Create right image panel"""
        self.image_panel = ImagePanel(self.main_frame, self)
        self.image_panel.frame.grid(row=0, column=1, sticky="nsew")


if __name__ == "__main__":
    app = ImageLabGUI()
    app.run()
