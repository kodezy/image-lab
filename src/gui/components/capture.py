import tkinter as tk
from tkinter import ttk
from typing import Any

from src.gui.utils import create_button, create_labeled_frame, create_scrollable_frame, create_spinbox


class CapturePanel:
    """Panel for image capture configuration and operations"""

    def __init__(self, parent: tk.Widget, app: Any) -> None:
        self.parent = parent
        self.app = app

        self._setup_variables()
        self._create_frame()
        self._update_device_controls_state()

    def refresh(self) -> None:
        """Refresh panel with current configuration"""
        self.device_enabled_var.set(self.app.capture_config.device_enabled)
        self.device_id_var.set(self.app.capture_config.device_id)
        self._update_device_controls_state()

    def _setup_variables(self) -> None:
        """Setup tkinter variables"""
        self.device_enabled_var = tk.BooleanVar(value=self.app.capture_config.device_enabled)
        self.device_id_var = tk.IntVar(value=self.app.capture_config.device_id)

    def _create_frame(self) -> None:
        """Create capture frame"""
        self.frame = ttk.Frame(self.parent)
        _, self.scrollable_frame, _ = create_scrollable_frame(self.frame)

        self._create_source_section()
        self._create_device_section()
        self._create_actions_section()

    def _update_device_controls_state(self) -> None:
        """Update device controls enabled state"""
        state = "normal" if self.device_enabled_var.get() else "disabled"
        self.device_spinbox.configure(state=state)

    def _create_source_section(self) -> None:
        """Create capture source selection"""
        source_frame = create_labeled_frame(self.scrollable_frame, "📷 Capture Source")
        source_frame.pack(fill=tk.X, pady=5, padx=5)

        ttk.Radiobutton(
            source_frame,
            text="Screen",
            variable=self.device_enabled_var,
            value=False,
            command=self._on_source_changed,
        ).pack(anchor=tk.W, pady=2)

        ttk.Radiobutton(
            source_frame,
            text="Device",
            variable=self.device_enabled_var,
            value=True,
            command=self._on_source_changed,
        ).pack(anchor=tk.W, pady=2)

    def _on_source_changed(self) -> None:
        """Handle capture source change"""
        self.app.capture_config.device_enabled = self.device_enabled_var.get()
        self._update_device_controls_state()

    def _create_device_section(self) -> None:
        """Create device configuration section"""
        device_frame = create_labeled_frame(self.scrollable_frame, "⚙️ Device Settings")
        device_frame.pack(fill=tk.X, pady=5, padx=5)

        device_id_frame, self.device_spinbox = create_spinbox(
            device_frame,
            "Device ID",
            self.device_id_var,
            0,
            10,
            command=self._on_device_id_changed,
        )
        device_id_frame.pack(fill=tk.X, pady=2)

    def _on_device_id_changed(self) -> None:
        """Handle device ID change"""
        self.app.capture_config.device_id = self.device_id_var.get()

    def _create_actions_section(self) -> None:
        """Create action buttons section"""
        actions_frame = create_labeled_frame(self.scrollable_frame, "🎯 Actions")
        actions_frame.pack(fill=tk.X, pady=5, padx=5)

        button_frame = ttk.Frame(actions_frame)
        button_frame.pack(fill=tk.X, pady=5)

        create_button(button_frame, "📷 Capture Image", self.app.capture_new_image).pack(fill=tk.X, pady=2)
