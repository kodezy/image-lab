import sys
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk
from typing import Any


def get_padding() -> int:
    """Get platform-appropriate padding value"""
    return 8 if sys.platform == "darwin" else 10


def show_error(message: str, title: str = "Error") -> None:
    """Show error message dialog"""
    messagebox.showerror(title, message)


def show_success(message: str, title: str = "Success") -> None:
    """Show success message dialog"""
    messagebox.showinfo(title, message)


def show_warning(message: str, title: str = "Warning") -> None:
    """Show warning message dialog"""
    messagebox.showwarning(title, message)


def create_labeled_frame(parent: tk.Widget, text: str, padding: int | None = None) -> ttk.LabelFrame:
    """Create labeled frame with consistent styling"""
    if padding is None:
        padding = get_padding()
    frame = ttk.LabelFrame(parent, text=text, padding=padding)
    return frame


def create_checkbox(
    parent: tk.Widget,
    text: str,
    variable: tk.BooleanVar,
    command: Callable | None = None,
) -> ttk.Checkbutton:
    """Create checkbox with consistent styling"""
    checkbox = ttk.Checkbutton(parent, text=text, variable=variable)

    if command:
        checkbox.configure(command=command)

    return checkbox


def create_slider(
    parent: tk.Widget,
    text: str,
    variable: tk.IntVar | tk.DoubleVar,
    from_: float,
    to: float,
    command: Callable | None = None,
) -> tuple[ttk.Frame, ttk.Scale, ttk.Label]:
    """Create slider with label and value display"""
    frame = ttk.Frame(parent)

    label = ttk.Label(frame, text=f"{text}:")
    label.pack(side=tk.LEFT)

    scale = ttk.Scale(frame, from_=from_, to=to, variable=variable, orient=tk.HORIZONTAL)
    scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))

    if command:
        scale.configure(command=command)

    value_label = ttk.Label(frame, text=str(variable.get()), width=8)
    value_label.pack(side=tk.RIGHT)

    def update_label(*args: Any) -> None:
        value = variable.get()
        if isinstance(variable, tk.IntVar):
            value = int(value)
        value_label.config(text=str(value))

    variable.trace_add("write", update_label)

    return frame, scale, value_label


def create_combobox(
    parent: tk.Widget,
    text: str,
    variable: tk.StringVar,
    values: list[str],
    command: Callable | None = None,
) -> tuple[ttk.Frame, ttk.Combobox]:
    """Create combobox with label"""
    frame = ttk.Frame(parent)

    label = ttk.Label(frame, text=f"{text}:")
    label.pack(side=tk.LEFT)

    combobox = ttk.Combobox(frame, textvariable=variable, values=values, state="readonly", width=15)
    combobox.pack(side=tk.LEFT, padx=(10, 0))

    if command:
        combobox.bind("<<ComboboxSelected>>", lambda e: command(combobox.get()))

    return frame, combobox


def create_spinbox(
    parent: tk.Widget,
    text: str,
    variable: tk.IntVar,
    from_: int,
    to: int,
    command: Callable | None = None,
) -> tuple[ttk.Frame, ttk.Spinbox]:
    """Create spinbox with label"""
    frame = ttk.Frame(parent)

    label = ttk.Label(frame, text=f"{text}:")
    label.pack(side=tk.LEFT)

    spinbox = ttk.Spinbox(frame, from_=from_, to=to, textvariable=variable, width=10)
    spinbox.pack(side=tk.LEFT, padx=(10, 0))

    if command:
        spinbox.configure(command=command)

    return frame, spinbox


def create_button(parent: tk.Widget, text: str, command: Callable, style: str = "") -> ttk.Button:
    """Create button with consistent styling"""
    button = ttk.Button(parent, text=text, command=command)

    if style:
        button.configure(style=style)

    return button


def create_scrollable_frame(parent: tk.Widget) -> tuple[tk.Canvas, ttk.Frame, ttk.Scrollbar]:
    """Create scrollable frame for long content"""

    def configure_scroll_region(event: tk.Event | None = None) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))

    def on_mousewheel(event: tk.Event) -> None:
        if sys.platform == "darwin":
            delta = -event.delta
        else:
            delta = int(-1 * (event.delta / 120))

        canvas.yview_scroll(delta, "units")

    def configure_canvas_window(event: tk.Event | None = None) -> None:
        canvas_width = canvas.winfo_width()
        if canvas_width > 1:
            canvas.itemconfig(canvas_window, width=canvas_width)

    bg_color = _get_canvas_bg_color()
    canvas = tk.Canvas(parent, highlightthickness=0, bg=bg_color)
    scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    canvas.configure(yscrollcommand=scrollbar.set)

    scrollable_frame.bind("<Configure>", configure_scroll_region)

    canvas.bind("<MouseWheel>", on_mousewheel)
    parent.bind("<MouseWheel>", on_mousewheel)

    canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    canvas.bind("<Configure>", configure_canvas_window)

    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    return canvas, scrollable_frame, scrollbar


def _get_canvas_bg_color() -> str:
    """Get appropriate canvas background color based on system theme"""
    if sys.platform == "darwin":
        try:
            style = ttk.Style()
            bg_color = style.lookup("TFrame", "background")

            if bg_color:
                return bg_color

        except Exception as exception:
            pass

    return "white"
