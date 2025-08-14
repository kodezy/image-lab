import tkinter as tk
from tkinter import messagebox
from typing import Any

ABOUT_TEXT: str = """Image Lab - Computer Vision Toolkit

A practical toolkit designed to streamline common computer vision operations through an intuitive interface.

Built with Python using modern CV libraries.

Version: 0.1.0"""

SHORTCUTS_TEXT: str = """Keyboard Shortcuts:

File Operations:
Ctrl+N - New Capture
Ctrl+O - Load Image
Ctrl+S - Save Image
Ctrl+L - Load Config
Ctrl+Shift+S - Save Config

View Operations:
Ctrl++ - Zoom In
Ctrl+- - Zoom Out
Ctrl+0 - Reset Zoom
Ctrl+F - Fit to Window

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
        file_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="File", menu=file_menu)

        file_menu.add_command(label="New Capture", command=self.app.capture_new_image, accelerator="Ctrl+N")
        file_menu.add_separator()

        file_menu.add_command(label="Load Image...", command=self.app.load_image_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Image...", command=self.app.save_image_file, accelerator="Ctrl+S")
        file_menu.add_separator()

        file_menu.add_command(label="Load Config...", command=self.app.load_config_file, accelerator="Ctrl+L")
        file_menu.add_command(label="Save Config...", command=self.app.save_config_file, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()

        file_menu.add_command(label="Exit", command=self._exit_app, accelerator="Alt+F4")

        self._bind_file_shortcuts()

    def _exit_app(self) -> None:
        """Exit application with confirmation"""
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            self.parent.quit()

    def _bind_file_shortcuts(self) -> None:
        """Bind keyboard shortcuts for file operations"""
        self.parent.bind("<Control-n>", lambda e: self.app.capture_new_image())
        self.parent.bind("<Control-o>", lambda e: self.app.load_image_file())
        self.parent.bind("<Control-s>", lambda e: self.app.save_image_file())
        self.parent.bind("<Control-l>", lambda e: self.app.load_config_file())
        self.parent.bind("<Control-Shift-S>", lambda e: self.app.save_config_file())
        self.parent.bind("<F5>", lambda e: self.app.update_image_display())

        self.parent.bind("<Control-plus>", lambda e: self._zoom_in())
        self.parent.bind("<Control-minus>", lambda e: self._zoom_out())
        self.parent.bind("<Control-0>", lambda e: self._reset_zoom())
        self.parent.bind("<Control-f>", lambda e: self._fit_to_window())

    def _create_edit_menu(self) -> None:
        """Create Edit menu"""
        edit_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Edit", menu=edit_menu)

        edit_menu.add_command(label="Reset All Configs", command=self._reset_configs)
        edit_menu.add_command(label="Refresh Image", command=self.app.update_image_display, accelerator="F5")

    def _reset_configs(self) -> None:
        """Reset configurations with confirmation"""
        if messagebox.askyesno("Reset Configurations", "Reset all configurations to defaults?"):
            self.app.reset_configs()

    def _create_view_menu(self) -> None:
        """Create View menu"""
        view_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="View", menu=view_menu)

        view_menu.add_command(label="Zoom In", command=self._zoom_in, accelerator="Ctrl++")
        view_menu.add_command(label="Zoom Out", command=self._zoom_out, accelerator="Ctrl+-")
        view_menu.add_command(label="Reset Zoom", command=self._reset_zoom, accelerator="Ctrl+0")
        view_menu.add_separator()

        view_menu.add_command(label="Fit to Window", command=self._fit_to_window, accelerator="Ctrl+F")

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
        messagebox.showinfo("Keyboard Shortcuts", SHORTCUTS_TEXT)
