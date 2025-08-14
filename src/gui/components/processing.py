import tkinter as tk
from tkinter import ttk
from typing import Any

from src.gui.utils import create_checkbox, create_combobox, create_labeled_frame, create_scrollable_frame, create_slider


class ProcessingPanel:
    """Panel for image processing configuration"""

    def __init__(self, parent: tk.Widget, app: Any) -> None:
        self.parent = parent
        self.app = app

        self._setup_variables()
        self._create_frame()

    def refresh(self) -> None:
        """Refresh panel with current configuration"""
        config = self.app.processing_config

        self.grayscale_var.set(config.grayscale)

        # Update crop variables
        self.crop_enabled_var.set(config.crop_enabled)
        self.crop_x1_var.set(config.crop_x1)
        self.crop_y1_var.set(config.crop_y1)
        self.crop_x2_var.set(config.crop_x2)
        self.crop_y2_var.set(config.crop_y2)

        self.threshold_enabled_var.set(config.threshold_enabled)
        self.threshold_type_var.set(config.threshold_type)
        self.threshold_value_var.set(config.threshold_value)

        self.bilateral_filter_var.set(config.bilateral_filter)
        self.bilateral_d_var.set(config.bilateral_d)

        self.gaussian_blur_var.set(config.gaussian_blur)
        self.gaussian_kernel_var.set(config.gaussian_kernel)

        self.median_filter_var.set(config.median_filter)
        self.median_kernel_var.set(config.median_kernel)

        self.clahe_var.set(config.clahe)
        self.clahe_clip_var.set(config.clahe_clip_limit)

        self.sharpen_var.set(config.sharpen)
        self.sharpen_strength_var.set(config.sharpen_strength)

        self.edge_enhancement_var.set(config.edge_enhancement)
        self.histogram_eq_var.set(config.histogram_equalization)

        self.morphology_var.set(config.morphology)
        self.morph_kernel_var.set(config.morph_kernel_size)

        self.character_separation_var.set(config.character_separation)
        self.vertical_line_removal_var.set(config.vertical_line_removal)
        self.horizontal_line_removal_var.set(config.horizontal_line_removal)
        self.noise_dots_removal_var.set(config.noise_dots_removal)

        self.adaptive_hist_eq_var.set(config.adaptive_hist_eq)
        self.multi_otsu_var.set(config.multi_otsu)
        self.local_binary_pattern_var.set(config.local_binary_pattern)

        self._update_grayscale_dependent_controls()
        self._update_crop_status()

    def _setup_variables(self) -> None:
        """Setup tkinter variables for all processing options"""
        config = self.app.processing_config

        self.grayscale_var = tk.BooleanVar(value=config.grayscale)

        # Crop variables
        self.crop_enabled_var = tk.BooleanVar(value=config.crop_enabled)
        self.crop_x1_var = tk.IntVar(value=config.crop_x1)
        self.crop_y1_var = tk.IntVar(value=config.crop_y1)
        self.crop_x2_var = tk.IntVar(value=config.crop_x2)
        self.crop_y2_var = tk.IntVar(value=config.crop_y2)

        self.threshold_enabled_var = tk.BooleanVar(value=config.threshold_enabled)
        self.threshold_type_var = tk.StringVar(value=config.threshold_type)
        self.threshold_value_var = tk.IntVar(value=config.threshold_value)

        self.bilateral_filter_var = tk.BooleanVar(value=config.bilateral_filter)
        self.bilateral_d_var = tk.IntVar(value=config.bilateral_d)

        self.gaussian_blur_var = tk.BooleanVar(value=config.gaussian_blur)
        self.gaussian_kernel_var = tk.IntVar(value=config.gaussian_kernel)

        self.median_filter_var = tk.BooleanVar(value=config.median_filter)
        self.median_kernel_var = tk.IntVar(value=config.median_kernel)

        self.clahe_var = tk.BooleanVar(value=config.clahe)
        self.clahe_clip_var = tk.DoubleVar(value=config.clahe_clip_limit)

        self.sharpen_var = tk.BooleanVar(value=config.sharpen)
        self.sharpen_strength_var = tk.DoubleVar(value=config.sharpen_strength)

        self.edge_enhancement_var = tk.BooleanVar(value=config.edge_enhancement)
        self.histogram_eq_var = tk.BooleanVar(value=config.histogram_equalization)

        # Variables for grayscale-dependent operations
        self.adaptive_hist_eq_var = tk.BooleanVar(value=config.adaptive_hist_eq)
        self.multi_otsu_var = tk.BooleanVar(value=config.multi_otsu)
        self.local_binary_pattern_var = tk.BooleanVar(value=config.local_binary_pattern)

        self.morphology_var = tk.BooleanVar(value=config.morphology)
        self.morph_kernel_var = tk.IntVar(value=config.morph_kernel_size)

        self.character_separation_var = tk.BooleanVar(value=config.character_separation)
        self.vertical_line_removal_var = tk.BooleanVar(value=config.vertical_line_removal)
        self.horizontal_line_removal_var = tk.BooleanVar(value=config.horizontal_line_removal)
        self.noise_dots_removal_var = tk.BooleanVar(value=config.noise_dots_removal)

    def _create_frame(self) -> None:
        """Create processing frame"""
        self.frame = ttk.Frame(self.parent)
        _, self.scrollable_frame, _ = create_scrollable_frame(self.frame)

        self._create_format_section()
        self._create_crop_section()
        self._create_threshold_section()
        self._create_noise_filters_section()
        self._create_enhancement_section()
        self._create_morphology_section()
        self._create_misc_section()

        # Initialize control states
        self._update_grayscale_dependent_controls()

    def _create_format_section(self) -> None:
        """Create format section"""
        format_frame = create_labeled_frame(self.scrollable_frame, "📤 Format")
        format_frame.pack(fill=tk.X, pady=5, padx=5)

        create_checkbox(
            format_frame,
            "Grayscale",
            self.grayscale_var,
            self._on_grayscale_changed,
        ).pack(anchor=tk.W, pady=2)

    def _on_grayscale_changed(self) -> None:
        """Handle grayscale change"""
        self.app.processing_config.grayscale = self.grayscale_var.get()
        self._update_grayscale_dependent_controls()
        self.app.update_image_display()

    def _create_crop_section(self) -> None:
        """Create crop section"""
        crop_frame = create_labeled_frame(self.scrollable_frame, "✂️ Crop")
        crop_frame.pack(fill=tk.X, pady=5, padx=5)

        create_checkbox(
            crop_frame,
            "Crop",
            self.crop_enabled_var,
            self._on_crop_enabled_changed,
        ).pack(anchor=tk.W, pady=2)

        coords_frame = ttk.Frame(crop_frame)
        coords_frame.pack(fill=tk.X, pady=5)

        coord_row1 = ttk.Frame(coords_frame)
        coord_row1.pack(fill=tk.X, pady=2)

        ttk.Label(coord_row1, text="X1:").pack(side=tk.LEFT, padx=(0, 5))
        x1_spinbox = ttk.Spinbox(
            coord_row1,
            from_=0,
            to=9999,
            textvariable=self.crop_x1_var,
            width=8,
            command=self._on_crop_x1_changed,
        )
        x1_spinbox.pack(side=tk.LEFT, padx=(0, 15))
        x1_spinbox.bind("<KeyRelease>", lambda e: self._on_crop_x1_changed())

        ttk.Label(coord_row1, text="Y1:").pack(side=tk.LEFT, padx=(0, 5))
        y1_spinbox = ttk.Spinbox(
            coord_row1,
            from_=0,
            to=9999,
            textvariable=self.crop_y1_var,
            width=8,
            command=self._on_crop_y1_changed,
        )

        y1_spinbox.pack(side=tk.LEFT)
        y1_spinbox.bind("<KeyRelease>", lambda e: self._on_crop_y1_changed())

        coord_row2 = ttk.Frame(coords_frame)
        coord_row2.pack(fill=tk.X, pady=2)

        ttk.Label(coord_row2, text="X2:").pack(side=tk.LEFT, padx=(0, 5))
        x2_spinbox = ttk.Spinbox(
            coord_row2,
            from_=0,
            to=9999,
            textvariable=self.crop_x2_var,
            width=8,
            command=self._on_crop_x2_changed,
        )
        x2_spinbox.pack(side=tk.LEFT, padx=(0, 15))
        x2_spinbox.bind("<KeyRelease>", lambda e: self._on_crop_x2_changed())

        ttk.Label(coord_row2, text="Y2:").pack(side=tk.LEFT, padx=(0, 5))
        y2_spinbox = ttk.Spinbox(
            coord_row2,
            from_=0,
            to=9999,
            textvariable=self.crop_y2_var,
            width=8,
            command=self._on_crop_y2_changed,
        )
        y2_spinbox.pack(side=tk.LEFT)
        y2_spinbox.bind("<KeyRelease>", lambda e: self._on_crop_y2_changed())

        # Status label
        self.crop_status_label = ttk.Label(crop_frame, text="Crop: Disabled", foreground="gray")
        self.crop_status_label.pack(anchor=tk.W, pady=2)

    def _on_crop_enabled_changed(self):
        """Handle crop enabled change"""
        self.app.processing_config.crop_enabled = self.crop_enabled_var.get()
        self._update_crop_status()
        self.app.update_image_display()

    def _on_crop_x1_changed(self):
        """Handle crop X1 change"""
        self.app.processing_config.crop_x1 = self.crop_x1_var.get()
        self._update_crop_status()
        self.app.update_image_display()

    def _on_crop_y1_changed(self):
        """Handle crop Y1 change"""
        self.app.processing_config.crop_y1 = self.crop_y1_var.get()
        self._update_crop_status()
        self.app.update_image_display()

    def _on_crop_x2_changed(self):
        """Handle crop X2 change"""
        self.app.processing_config.crop_x2 = self.crop_x2_var.get()
        self._update_crop_status()
        self.app.update_image_display()

    def _on_crop_y2_changed(self):
        """Handle crop Y2 change"""
        self.app.processing_config.crop_y2 = self.crop_y2_var.get()
        self._update_crop_status()
        self.app.update_image_display()

    def _update_crop_status(self):
        """Update crop status label"""
        if not self.crop_enabled_var.get():
            self.crop_status_label.config(text="Crop: Disabled", foreground="gray")
            return

        x1 = self.crop_x1_var.get()
        y1 = self.crop_y1_var.get()
        x2 = self.crop_x2_var.get()
        y2 = self.crop_y2_var.get()

        if x2 > x1 and y2 > y1:
            width = x2 - x1
            height = y2 - y1
            self.crop_status_label.config(
                text=f"Crop: ({x1},{y1}) -> ({x2},{y2}) | {width}x{height}px", foreground="blue"
            )
        else:
            self.crop_status_label.config(text="Crop: Invalid coordinates", foreground="red")

    def _create_threshold_section(self) -> None:
        """Create threshold section"""
        threshold_frame = create_labeled_frame(self.scrollable_frame, "🎯 Threshold")
        threshold_frame.pack(fill=tk.X, pady=5, padx=5)

        self.threshold_enabled_checkbox = create_checkbox(
            threshold_frame,
            "Threshold",
            self.threshold_enabled_var,
            self._on_threshold_enabled_changed,
        )
        self.threshold_enabled_checkbox.pack(anchor=tk.W, pady=2)

        threshold_type_frame, self.threshold_combobox = create_combobox(
            threshold_frame,
            "Type",
            self.threshold_type_var,
            ["BINARY", "OTSU_BINARY", "ADAPTIVE_MEAN", "ADAPTIVE_GAUSSIAN"],
            self._on_threshold_type_changed,
        )
        threshold_type_frame.pack(fill=tk.X, pady=2)

        threshold_value_frame, _, _ = create_slider(
            threshold_frame,
            "Threshold Value",
            self.threshold_value_var,
            0,
            255,
            self._on_threshold_value_changed,
        )
        threshold_value_frame.pack(fill=tk.X, pady=2)

    def _on_threshold_enabled_changed(self) -> None:
        """Handle threshold enabled change"""
        self.app.processing_config.threshold_enabled = self.threshold_enabled_var.get()
        self.app.update_image_display()

    def _on_threshold_type_changed(self) -> None:
        """Handle threshold type change"""
        self.app.processing_config.threshold_type = self.threshold_type_var.get()
        self.app.update_image_display()

    def _on_threshold_value_changed(self, value) -> None:
        """Handle threshold value change"""
        self.app.processing_config.threshold_value = int(float(value))
        self.app.update_image_display()

    def _create_noise_filters_section(self) -> None:
        """Create noise filters section"""
        noise_frame = create_labeled_frame(self.scrollable_frame, "🔧 Noise")
        noise_frame.pack(fill=tk.X, pady=5, padx=5)

        create_checkbox(
            noise_frame,
            "Bilateral Filter",
            self.bilateral_filter_var,
            self._on_bilateral_filter_changed,
        ).pack(anchor=tk.W, pady=2)

        bilateral_d_frame, _, _ = create_slider(
            noise_frame,
            "Bilateral D",
            self.bilateral_d_var,
            1,
            15,
            self._on_bilateral_d_changed,
        )
        bilateral_d_frame.pack(fill=tk.X, pady=2)

        create_checkbox(
            noise_frame,
            "Gaussian Blur",
            self.gaussian_blur_var,
            self._on_gaussian_blur_changed,
        ).pack(anchor=tk.W, pady=2)

        gaussian_kernel_frame, _, _ = create_slider(
            noise_frame,
            "Gaussian Kernel",
            self.gaussian_kernel_var,
            1,
            10,
            self._on_gaussian_kernel_changed,
        )
        gaussian_kernel_frame.pack(fill=tk.X, pady=2)

        create_checkbox(
            noise_frame,
            "Median Filter",
            self.median_filter_var,
            self._on_median_filter_changed,
        ).pack(anchor=tk.W, pady=2)

        median_kernel_frame, _, _ = create_slider(
            noise_frame,
            "Median Kernel",
            self.median_kernel_var,
            1,
            10,
            self._on_median_kernel_changed,
        )
        median_kernel_frame.pack(fill=tk.X, pady=2)

    def _on_bilateral_filter_changed(self) -> None:
        """Handle bilateral filter change"""
        self.app.processing_config.bilateral_filter = self.bilateral_filter_var.get()
        self.app.update_image_display()

    def _on_bilateral_d_changed(self, value) -> None:
        """Handle bilateral d change"""
        self.app.processing_config.bilateral_d = int(float(value))
        self.app.update_image_display()

    def _on_gaussian_blur_changed(self) -> None:
        """Handle gaussian blur change"""
        self.app.processing_config.gaussian_blur = self.gaussian_blur_var.get()
        self.app.update_image_display()

    def _on_gaussian_kernel_changed(self, value) -> None:
        """Handle gaussian kernel change"""
        self.app.processing_config.gaussian_kernel = int(float(value))
        self.app.update_image_display()

    def _on_median_filter_changed(self) -> None:
        """Handle median filter change"""
        self.app.processing_config.median_filter = self.median_filter_var.get()
        self.app.update_image_display()

    def _on_median_kernel_changed(self, value) -> None:
        """Handle median kernel change"""
        self.app.processing_config.median_kernel = int(float(value))
        self.app.update_image_display()

    def _create_enhancement_section(self) -> None:
        """Create enhancement section"""
        enhance_frame = create_labeled_frame(self.scrollable_frame, "✨ Enhancement")
        enhance_frame.pack(fill=tk.X, pady=5, padx=5)

        self.clahe_checkbox = create_checkbox(enhance_frame, "CLAHE", self.clahe_var, self._on_clahe_changed)
        self.clahe_checkbox.pack(anchor=tk.W, pady=2)

        clahe_clip_frame, _, _ = create_slider(
            enhance_frame,
            "CLAHE Clip Limit",
            self.clahe_clip_var,
            0.5,
            10.0,
            self._on_clahe_clip_changed,
        )
        clahe_clip_frame.pack(fill=tk.X, pady=2)

        create_checkbox(enhance_frame, "Sharpen", self.sharpen_var, self._on_sharpen_changed).pack(anchor=tk.W, pady=2)

        sharpen_strength_frame, _, _ = create_slider(
            enhance_frame,
            "Sharpen Strength",
            self.sharpen_strength_var,
            0.0,
            1.0,
            self._on_sharpen_strength_changed,
        )
        sharpen_strength_frame.pack(fill=tk.X, pady=2)

    def _on_clahe_changed(self) -> None:
        """Handle CLAHE change"""
        self.app.processing_config.clahe = self.clahe_var.get()
        self.app.update_image_display()

    def _on_clahe_clip_changed(self, value) -> None:
        """Handle CLAHE clip limit change"""
        self.app.processing_config.clahe_clip_limit = float(value)
        self.app.update_image_display()

    def _on_sharpen_changed(self) -> None:
        """Handle sharpen change"""
        self.app.processing_config.sharpen = self.sharpen_var.get()
        self.app.update_image_display()

    def _on_sharpen_strength_changed(self, value) -> None:
        """Handle sharpen strength change"""
        self.app.processing_config.sharpen_strength = float(value)
        self.app.update_image_display()

    def _create_morphology_section(self) -> None:
        """Create morphology section"""
        morph_frame = create_labeled_frame(self.scrollable_frame, "🔄 Morphology")
        morph_frame.pack(fill=tk.X, pady=5, padx=5)

        create_checkbox(
            morph_frame,
            "Morphology",
            self.morphology_var,
            self._on_morphology_changed,
        ).pack(anchor=tk.W, pady=2)

        morph_kernel_frame, _, _ = create_slider(
            morph_frame,
            "Kernel Size",
            self.morph_kernel_var,
            1,
            10,
            self._on_morph_kernel_changed,
        )
        morph_kernel_frame.pack(fill=tk.X, pady=2)

    def _on_morphology_changed(self) -> None:
        """Handle morphology change"""
        self.app.processing_config.morphology = self.morphology_var.get()
        self.app.update_image_display()

    def _on_morph_kernel_changed(self, value) -> None:
        """Handle morphology kernel change"""
        self.app.processing_config.morph_kernel_size = int(float(value))
        self.app.update_image_display()

    def _create_misc_section(self) -> None:
        """Create misc operations section"""
        misc_frame = create_labeled_frame(self.scrollable_frame, "⚙️ Misc")
        misc_frame.pack(fill=tk.X, pady=5, padx=5)

        self.edge_enhancement_checkbox = create_checkbox(
            misc_frame,
            "Edge Enhancement",
            self.edge_enhancement_var,
            self._on_edge_enhancement_changed,
        )
        self.edge_enhancement_checkbox.pack(anchor=tk.W, pady=2)

        self.histogram_eq_checkbox = create_checkbox(
            misc_frame,
            "Histogram Equalization",
            self.histogram_eq_var,
            self._on_histogram_eq_changed,
        )
        self.histogram_eq_checkbox.pack(anchor=tk.W, pady=2)

        self.adaptive_hist_eq_checkbox = create_checkbox(
            misc_frame,
            "Adaptive Histogram Equalization",
            self.adaptive_hist_eq_var,
            self._on_adaptive_hist_eq_changed,
        )
        self.adaptive_hist_eq_checkbox.pack(anchor=tk.W, pady=2)

        create_checkbox(
            misc_frame,
            "Character Separation",
            self.character_separation_var,
            self._on_character_separation_changed,
        ).pack(anchor=tk.W, pady=2)

        create_checkbox(
            misc_frame,
            "Remove Vertical Lines",
            self.vertical_line_removal_var,
            self._on_vertical_line_removal_changed,
        ).pack(anchor=tk.W, pady=2)

        create_checkbox(
            misc_frame,
            "Remove Horizontal Lines",
            self.horizontal_line_removal_var,
            self._on_horizontal_line_removal_changed,
        ).pack(anchor=tk.W, pady=2)

        create_checkbox(
            misc_frame,
            "Remove Noise Dots",
            self.noise_dots_removal_var,
            self._on_noise_dots_removal_changed,
        ).pack(anchor=tk.W, pady=2)

        self.multi_otsu_checkbox = create_checkbox(
            misc_frame,
            "Multi-OTSU Thresholding",
            self.multi_otsu_var,
            self._on_multi_otsu_changed,
        )
        self.multi_otsu_checkbox.pack(anchor=tk.W, pady=2)

        self.local_binary_pattern_checkbox = create_checkbox(
            misc_frame,
            "Local Binary Pattern",
            self.local_binary_pattern_var,
            self._on_local_binary_pattern_changed,
        )
        self.local_binary_pattern_checkbox.pack(anchor=tk.W, pady=2)

    def _on_edge_enhancement_changed(self) -> None:
        """Handle edge enhancement change"""
        self.app.processing_config.edge_enhancement = self.edge_enhancement_var.get()
        self.app.update_image_display()

    def _on_histogram_eq_changed(self) -> None:
        """Handle histogram equalization change"""
        self.app.processing_config.histogram_equalization = self.histogram_eq_var.get()
        self.app.update_image_display()

    def _on_character_separation_changed(self) -> None:
        """Handle character separation change"""
        self.app.processing_config.character_separation = self.character_separation_var.get()
        self.app.update_image_display()

    def _on_vertical_line_removal_changed(self) -> None:
        """Handle vertical line removal change"""
        self.app.processing_config.vertical_line_removal = self.vertical_line_removal_var.get()
        self.app.update_image_display()

    def _on_horizontal_line_removal_changed(self) -> None:
        """Handle horizontal line removal change"""
        self.app.processing_config.horizontal_line_removal = self.horizontal_line_removal_var.get()
        self.app.update_image_display()

    def _on_noise_dots_removal_changed(self) -> None:
        """Handle noise dots removal change"""
        self.app.processing_config.noise_dots_removal = self.noise_dots_removal_var.get()
        self.app.update_image_display()

    def _on_adaptive_hist_eq_changed(self) -> None:
        """Handle adaptive histogram equalization change"""
        self.app.processing_config.adaptive_hist_eq = self.adaptive_hist_eq_var.get()
        self.app.update_image_display()

    def _on_multi_otsu_changed(self) -> None:
        """Handle multi-OTSU change"""
        self.app.processing_config.multi_otsu = self.multi_otsu_var.get()
        self.app.update_image_display()

    def _on_local_binary_pattern_changed(self) -> None:
        """Handle local binary pattern change"""
        self.app.processing_config.local_binary_pattern = self.local_binary_pattern_var.get()
        self.app.update_image_display()

    def _update_grayscale_dependent_controls(self) -> None:
        """Update state of controls that depend on grayscale"""
        is_grayscale = self.grayscale_var.get()

        # Enable/disable grayscale-dependent controls
        state = tk.NORMAL if is_grayscale else tk.DISABLED

        self.threshold_enabled_checkbox.config(state=state)
        self.clahe_checkbox.config(state=state)
        self.edge_enhancement_checkbox.config(state=state)
        self.histogram_eq_checkbox.config(state=state)
        self.adaptive_hist_eq_checkbox.config(state=state)
        self.multi_otsu_checkbox.config(state=state)
        self.local_binary_pattern_checkbox.config(state=state)
