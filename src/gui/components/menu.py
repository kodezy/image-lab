import sys
import tkinter as tk
from tkinter import messagebox
from typing import Any

ABOUT_TEXT: str = """Image Lab - Computer Vision Toolkit

A practical toolkit designed to streamline common computer vision operations through an intuitive interface.

Built with Python using modern CV libraries.

Version: 0.1.0"""


def get_modifier_key() -> str:
    """Get platform-appropriate modifier key (Command on macOS, Control on others)"""
    return "Command" if sys.platform == "darwin" else "Ctrl"


def get_modifier_binding() -> str:
    """Get platform-appropriate modifier binding for tkinter"""
    return "Command" if sys.platform == "darwin" else "Control"


def get_shortcuts_text() -> str:
    """Get platform-appropriate keyboard shortcuts text"""
    mod = get_modifier_key()
    return f"""Keyboard Shortcuts:

    File Operations:
    {mod}+N - New Capture
    {mod}+O - Load Image
    {mod}+S - Save Image
    {mod}+L - Load Config
    {mod}+Shift+R - Reset Configs
    {mod}+Shift+S - Save Config

    View Operations:
    {mod}++ - Zoom In
    {mod}+- - Zoom Out
    {mod}+0 - Reset Zoom
    {mod}+F - Fit to Window

    Other:
    F5 - Refresh Image
    Alt+F4 - Exit"""


class MenuBar:
    """Menu bar for the main application"""

    def __init__(self, parent: tk.Tk, app: Any) -> None:
        self.parent = parent
        self.app = app

        self._create_menu()

    def _create_menu(self) -> None:
        """Create menu bar structure"""
        self.menu = tk.Menu(self.parent)

        self._create_file_menu()
        self._create_edit_menu()
        self._create_view_menu()
        self._create_help_menu()

    def _create_file_menu(self) -> None:
        """Create File menu"""
        mod = get_modifier_key()
        file_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="File", menu=file_menu)

        file_menu.add_command(label="New Capture", command=self.app.capture_new_image, accelerator=f"{mod}+N")
        file_menu.add_separator()

        file_menu.add_command(label="Load Image...", command=self.app.load_image_file, accelerator=f"{mod}+O")
        file_menu.add_command(label="Save Image...", command=self.app.save_image_file, accelerator=f"{mod}+S")
        file_menu.add_separator()

        file_menu.add_command(label="Load Config...", command=self.app.load_config_file, accelerator=f"{mod}+L")
        file_menu.add_command(label="Save Config...", command=self.app.save_config_file, accelerator=f"{mod}+Shift+S")
        file_menu.add_separator()

        file_menu.add_command(label="Exit", command=self._exit_app, accelerator="Alt+F4")

        self._bind_file_shortcuts()

    def _exit_app(self) -> None:
        """Exit application with confirmation"""
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            self.parent.quit()

    def _bind_file_shortcuts(self) -> None:
        """Bind keyboard shortcuts for file operations"""
        mod = get_modifier_binding()

        self.parent.bind(f"<{mod}-n>", lambda e: self.app.capture_new_image())
        self.parent.bind(f"<{mod}-o>", lambda e: self.app.load_image_file())
        self.parent.bind(f"<{mod}-s>", lambda e: self.app.save_image_file())
        self.parent.bind(f"<{mod}-l>", lambda e: self.app.load_config_file())
        self.parent.bind(f"<{mod}-Shift-R>", lambda e: self.app.reset_configs())
        self.parent.bind(f"<{mod}-Shift-S>", lambda e: self.app.save_config_file())
        self.parent.bind("<F5>", lambda e: self.app.update_image_display())

        self.parent.bind(f"<{mod}-plus>", lambda e: self._zoom_in())
        self.parent.bind(f"<{mod}-minus>", lambda e: self._zoom_out())
        self.parent.bind(f"<{mod}-0>", lambda e: self._reset_zoom())
        self.parent.bind(f"<{mod}-f>", lambda e: self._fit_to_window())

    def _create_edit_menu(self) -> None:
        """Create Edit menu"""
        mod = get_modifier_key()
        edit_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Edit", menu=edit_menu)

        edit_menu.add_command(label="Reset All Configs", command=self.app.reset_configs, accelerator=f"{mod}+Shift+R")
        edit_menu.add_command(label="Refresh Image", command=self.app.update_image_display, accelerator="F5")

    def _create_view_menu(self) -> None:
        """Create View menu"""
        mod = get_modifier_key()
        view_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="View", menu=view_menu)

        view_menu.add_command(label="Zoom In", command=self._zoom_in, accelerator=f"{mod}++")
        view_menu.add_command(label="Zoom Out", command=self._zoom_out, accelerator=f"{mod}+-")
        view_menu.add_command(label="Reset Zoom", command=self._reset_zoom, accelerator=f"{mod}+0")
        view_menu.add_separator()

        view_menu.add_command(label="Fit to Window", command=self._fit_to_window, accelerator=f"{mod}+F")

    def _zoom_in(self) -> None:
        """Zoom in on image"""
        if hasattr(self.app, "image_panel"):
            self.app.image_panel.zoom_in()

    def _zoom_out(self) -> None:
        """Zoom out on image"""
        if hasattr(self.app, "image_panel"):
            self.app.image_panel.zoom_out()

    def _reset_zoom(self) -> None:
        """Reset image zoom"""
        if hasattr(self.app, "image_panel"):
            self.app.image_panel.reset_zoom()

    def _fit_to_window(self) -> None:
        """Fit image to window"""
        if hasattr(self.app, "image_panel"):
            self.app.image_panel.fit_to_window()

    def _create_help_menu(self) -> None:
        """Create Help menu"""
        help_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Help", menu=help_menu)

        help_menu.add_command(label="About", command=self._show_about)
        help_menu.add_command(label="Keyboard Shortcuts", command=self._show_shortcuts)

    def _show_about(self) -> None:
        messagebox.showinfo("About Image Lab", ABOUT_TEXT)

    def _show_shortcuts(self) -> None:
        """Show keyboard shortcuts dialog"""
        messagebox.showinfo("Keyboard Shortcuts", get_shortcuts_text())
