import tkinter as tk
from tkinter import ttk
from typing import Any

import numpy as np

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

        self.color_space_var.set(config.color_space)
        self.crop_enabled_var.set(config.crop_enabled)

        # Update crop variables
        self.bbox_var.set(self._format_bbox(config.bbox))
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
        self.clahe_clip_limit_var.set(config.clahe_clip_limit)

        self.sharpen_var.set(config.sharpen)
        self.sharpen_strength_var.set(config.sharpen_strength)

        self.edge_enhancement_var.set(config.edge_enhancement)
        self.histogram_equalization_var.set(config.histogram_equalization)

        self.morphology_var.set(config.morphology)
        self.morph_kernel_size_var.set(config.morph_kernel_size)

        self.character_separation_var.set(config.character_separation)
        self.vertical_line_removal_var.set(config.vertical_line_removal)
        self.horizontal_line_removal_var.set(config.horizontal_line_removal)
        self.noise_dots_removal_var.set(config.noise_dots_removal)

        self.adaptive_hist_eq_var.set(config.adaptive_hist_eq)
        self.multi_otsu_var.set(config.multi_otsu)
        self.local_binary_pattern_var.set(config.local_binary_pattern)

        # Basic transformations
        self.resize_enabled_var.set(config.resize_enabled)
        self.resize_width_var.set(config.resize_width)
        self.resize_height_var.set(config.resize_height)
        self.resize_maintain_aspect_var.set(config.resize_maintain_aspect_ratio)
        self.gamma_correction_var.set(config.gamma_correction)
        self.gamma_value_var.set(config.gamma_value)

        self.denoise_nl_means_var.set(config.denoise_nl_means)
        self.denoise_h_var.set(config.denoise_h)
        self.denoise_template_window_var.set(config.denoise_template_window)
        self.denoise_search_window_var.set(config.denoise_search_window)
        self.edge_preserving_filter_var.set(config.edge_preserving_filter)
        self.edge_filter_flags_var.set(config.edge_filter_flags)
        self.edge_sigma_s_var.set(config.edge_sigma_s)
        self.edge_sigma_r_var.set(config.edge_sigma_r)
        self.noise_reduction_bilateral_var.set(config.noise_reduction_bilateral)
        self.bilateral_iterations_var.set(config.bilateral_iterations)

        self.bilateral_sigma_color_var.set(config.bilateral_sigma_color)
        self.bilateral_sigma_space_var.set(config.bilateral_sigma_space)
        self.gaussian_sigma_var.set(config.gaussian_sigma)
        self.background_subtraction_var.set(config.background_subtraction)
        self.bg_threshold_var.set(config.bg_threshold)

        self.clahe_tile_size_var.set(config.clahe_tile_size)
        self.adaptive_hist_kernel_var.set(config.adaptive_hist_kernel)
        self.multi_otsu_classes_var.set(config.multi_otsu_classes)
        self.intensity_normalization_var.set(config.intensity_normalization)
        self.norm_min_var.set(config.norm_min)
        self.norm_max_var.set(config.norm_max)
        self.contrast_stretching_var.set(config.contrast_stretching)
        self.stretch_min_percentile_var.set(config.stretch_min_percentile)
        self.stretch_max_percentile_var.set(config.stretch_max_percentile)

        self.vertical_kernel_size_var.set(config.vertical_kernel_size)
        self.horizontal_kernel_size_var.set(config.horizontal_kernel_size)
        self.hough_lines_removal_var.set(config.hough_lines_removal)
        self.hough_threshold_var.set(config.hough_threshold)
        self.hough_min_line_length_var.set(config.hough_min_line_length)
        self.hough_max_line_gap_var.set(config.hough_max_line_gap)

        self.stroke_width_normalization_var.set(config.stroke_width_normalization)
        self.stroke_iterations_var.set(config.stroke_iterations)
        self.morph_open_var.set(config.morph_open)
        self.morph_close_var.set(config.morph_close)
        self.tophat_var.set(config.tophat)
        self.tophat_kernel_size_var.set(config.tophat_kernel_size)
        self.blackhat_var.set(config.blackhat)
        self.blackhat_kernel_size_var.set(config.blackhat_kernel_size)
        self.gradient_var.set(config.gradient)
        self.gradient_kernel_size_var.set(config.gradient_kernel_size)
        self.morphological_gradient_var.set(config.morphological_gradient)
        self.morphological_gradient_kernel_var.set(config.morphological_gradient_kernel)

        # Advanced character operations
        self.char_sep_kernel_size_var.set(config.char_sep_kernel_size)
        self.character_dilation_var.set(config.character_dilation)
        self.dilation_kernel_size_var.set(config.dilation_kernel_size)
        self.dilation_iterations_var.set(config.dilation_iterations)
        self.character_erosion_var.set(config.character_erosion)
        self.erosion_kernel_size_var.set(config.erosion_kernel_size)
        self.erosion_iterations_var.set(config.erosion_iterations)
        self.min_contour_area_var.set(config.min_contour_area)

        self.text_enhancement_var.set(config.text_enhancement)
        self.text_kernel_size_var.set(config.text_kernel_size)
        self.detail_enhancement_var.set(config.detail_enhancement)
        self.detail_sigma_s_var.set(config.detail_sigma_s)
        self.detail_sigma_r_var.set(config.detail_sigma_r)
        self.edge_strength_var.set(config.edge_strength)
        self.unsharp_mask_var.set(config.unsharp_mask)
        self.unsharp_strength_var.set(config.unsharp_strength)

        # Advanced threshold
        self.adaptive_block_size_var.set(config.adaptive_block_size)
        self.adaptive_c_var.set(config.adaptive_c)

        self.contour_filtering_var.set(config.contour_filtering)
        self.contour_area_min_var.set(config.contour_area_min)
        self.contour_area_max_var.set(config.contour_area_max)
        self.connected_components_filtering_var.set(config.connected_components_filtering)
        self.cc_min_area_var.set(config.cc_min_area)
        self.cc_max_area_var.set(config.cc_max_area)
        self.aspect_ratio_filtering_var.set(config.aspect_ratio_filtering)
        self.min_aspect_ratio_var.set(config.min_aspect_ratio)
        self.max_aspect_ratio_var.set(config.max_aspect_ratio)

        self.distance_transform_var.set(config.distance_transform)
        self.distance_transform_type_var.set(config.distance_transform_type)
        self.skeletonize_var.set(config.skeletonize)
        self.watershed_markers_var.set(config.watershed_markers)
        self.lbp_radius_var.set(config.lbp_radius)
        self.lbp_n_points_var.set(config.lbp_n_points)

        self._update_resize_dependent_controls()
        self._update_grayscale_dependent_controls()
        self._update_binary_dependent_controls()
        self._update_crop_status()

    def _setup_variables(self) -> None:
        """Setup tkinter variables for all processing options"""
        config = self.app.processing_config

        self.color_space_var = tk.StringVar(value=config.color_space)
        self.crop_enabled_var = tk.BooleanVar(value=config.crop_enabled)
        self.bbox_var = tk.StringVar(value=self._format_bbox(config.bbox))
        self.resize_enabled_var = tk.BooleanVar(value=config.resize_enabled)
        self.resize_width_var = tk.IntVar(value=config.resize_width)
        self.resize_height_var = tk.IntVar(value=config.resize_height)
        self.resize_maintain_aspect_var = tk.BooleanVar(value=config.resize_maintain_aspect_ratio)
        self.gamma_correction_var = tk.BooleanVar(value=config.gamma_correction)
        self.gamma_value_var = tk.DoubleVar(value=config.gamma_value)

        self.denoise_nl_means_var = tk.BooleanVar(value=config.denoise_nl_means)
        self.denoise_h_var = tk.DoubleVar(value=config.denoise_h)
        self.denoise_template_window_var = tk.IntVar(value=config.denoise_template_window)
        self.denoise_search_window_var = tk.IntVar(value=config.denoise_search_window)
        self.edge_preserving_filter_var = tk.BooleanVar(value=config.edge_preserving_filter)
        self.edge_filter_flags_var = tk.IntVar(value=config.edge_filter_flags)
        self.edge_sigma_s_var = tk.DoubleVar(value=config.edge_sigma_s)
        self.edge_sigma_r_var = tk.DoubleVar(value=config.edge_sigma_r)
        self.noise_reduction_bilateral_var = tk.BooleanVar(value=config.noise_reduction_bilateral)
        self.bilateral_iterations_var = tk.IntVar(value=config.bilateral_iterations)

        self.bilateral_filter_var = tk.BooleanVar(value=config.bilateral_filter)
        self.bilateral_d_var = tk.IntVar(value=config.bilateral_d)
        self.bilateral_sigma_color_var = tk.IntVar(value=config.bilateral_sigma_color)
        self.bilateral_sigma_space_var = tk.IntVar(value=config.bilateral_sigma_space)
        self.gaussian_blur_var = tk.BooleanVar(value=config.gaussian_blur)
        self.gaussian_kernel_var = tk.IntVar(value=config.gaussian_kernel)
        self.gaussian_sigma_var = tk.DoubleVar(value=config.gaussian_sigma)
        self.median_filter_var = tk.BooleanVar(value=config.median_filter)
        self.median_kernel_var = tk.IntVar(value=config.median_kernel)
        self.background_subtraction_var = tk.BooleanVar(value=config.background_subtraction)
        self.bg_threshold_var = tk.IntVar(value=config.bg_threshold)

        self.histogram_equalization_var = tk.BooleanVar(value=config.histogram_equalization)
        self.clahe_var = tk.BooleanVar(value=config.clahe)
        self.clahe_clip_limit_var = tk.DoubleVar(value=config.clahe_clip_limit)
        self.clahe_tile_size_var = tk.IntVar(value=config.clahe_tile_size)
        self.adaptive_hist_eq_var = tk.BooleanVar(value=config.adaptive_hist_eq)
        self.adaptive_hist_kernel_var = tk.IntVar(value=config.adaptive_hist_kernel)
        self.multi_otsu_var = tk.BooleanVar(value=config.multi_otsu)
        self.multi_otsu_classes_var = tk.IntVar(value=config.multi_otsu_classes)
        self.intensity_normalization_var = tk.BooleanVar(value=config.intensity_normalization)
        self.norm_min_var = tk.IntVar(value=config.norm_min)
        self.norm_max_var = tk.IntVar(value=config.norm_max)
        self.contrast_stretching_var = tk.BooleanVar(value=config.contrast_stretching)
        self.stretch_min_percentile_var = tk.DoubleVar(value=config.stretch_min_percentile)
        self.stretch_max_percentile_var = tk.DoubleVar(value=config.stretch_max_percentile)

        self.vertical_line_removal_var = tk.BooleanVar(value=config.vertical_line_removal)
        self.vertical_kernel_size_var = tk.IntVar(value=config.vertical_kernel_size)
        self.horizontal_line_removal_var = tk.BooleanVar(value=config.horizontal_line_removal)
        self.horizontal_kernel_size_var = tk.IntVar(value=config.horizontal_kernel_size)
        self.hough_lines_removal_var = tk.BooleanVar(value=config.hough_lines_removal)
        self.hough_threshold_var = tk.IntVar(value=config.hough_threshold)
        self.hough_min_line_length_var = tk.IntVar(value=config.hough_min_line_length)
        self.hough_max_line_gap_var = tk.IntVar(value=config.hough_max_line_gap)

        self.stroke_width_normalization_var = tk.BooleanVar(value=config.stroke_width_normalization)
        self.stroke_iterations_var = tk.IntVar(value=config.stroke_iterations)
        self.morphology_var = tk.BooleanVar(value=config.morphology)
        self.morph_kernel_size_var = tk.IntVar(value=config.morph_kernel_size)
        self.morph_open_var = tk.BooleanVar(value=config.morph_open)
        self.morph_close_var = tk.BooleanVar(value=config.morph_close)
        self.tophat_var = tk.BooleanVar(value=config.tophat)
        self.tophat_kernel_size_var = tk.IntVar(value=config.tophat_kernel_size)
        self.blackhat_var = tk.BooleanVar(value=config.blackhat)
        self.blackhat_kernel_size_var = tk.IntVar(value=config.blackhat_kernel_size)
        self.gradient_var = tk.BooleanVar(value=config.gradient)
        self.gradient_kernel_size_var = tk.IntVar(value=config.gradient_kernel_size)
        self.morphological_gradient_var = tk.BooleanVar(value=config.morphological_gradient)
        self.morphological_gradient_kernel_var = tk.IntVar(value=config.morphological_gradient_kernel)

        # Character operations variables
        self.character_separation_var = tk.BooleanVar(value=config.character_separation)
        self.char_sep_kernel_size_var = tk.IntVar(value=config.char_sep_kernel_size)
        self.character_dilation_var = tk.BooleanVar(value=config.character_dilation)
        self.dilation_kernel_size_var = tk.IntVar(value=config.dilation_kernel_size)
        self.dilation_iterations_var = tk.IntVar(value=config.dilation_iterations)
        self.character_erosion_var = tk.BooleanVar(value=config.character_erosion)
        self.erosion_kernel_size_var = tk.IntVar(value=config.erosion_kernel_size)
        self.erosion_iterations_var = tk.IntVar(value=config.erosion_iterations)
        self.noise_dots_removal_var = tk.BooleanVar(value=config.noise_dots_removal)
        self.min_contour_area_var = tk.IntVar(value=config.min_contour_area)

        self.text_enhancement_var = tk.BooleanVar(value=config.text_enhancement)
        self.text_kernel_size_var = tk.IntVar(value=config.text_kernel_size)
        self.detail_enhancement_var = tk.BooleanVar(value=config.detail_enhancement)
        self.detail_sigma_s_var = tk.DoubleVar(value=config.detail_sigma_s)
        self.detail_sigma_r_var = tk.DoubleVar(value=config.detail_sigma_r)
        self.edge_enhancement_var = tk.BooleanVar(value=config.edge_enhancement)
        self.edge_strength_var = tk.DoubleVar(value=config.edge_strength)
        self.unsharp_mask_var = tk.BooleanVar(value=config.unsharp_mask)
        self.unsharp_strength_var = tk.DoubleVar(value=config.unsharp_strength)
        self.sharpen_var = tk.BooleanVar(value=config.sharpen)
        self.sharpen_strength_var = tk.DoubleVar(value=config.sharpen_strength)

        # Threshold variables
        self.threshold_enabled_var = tk.BooleanVar(value=config.threshold_enabled)
        self.threshold_type_var = tk.StringVar(value=config.threshold_type)
        self.threshold_value_var = tk.IntVar(value=config.threshold_value)
        self.adaptive_block_size_var = tk.IntVar(value=config.adaptive_block_size)
        self.adaptive_c_var = tk.IntVar(value=config.adaptive_c)

        self.contour_filtering_var = tk.BooleanVar(value=config.contour_filtering)
        self.contour_area_min_var = tk.IntVar(value=config.contour_area_min)
        self.contour_area_max_var = tk.IntVar(value=config.contour_area_max)
        self.connected_components_filtering_var = tk.BooleanVar(value=config.connected_components_filtering)
        self.cc_min_area_var = tk.IntVar(value=config.cc_min_area)
        self.cc_max_area_var = tk.IntVar(value=config.cc_max_area)
        self.aspect_ratio_filtering_var = tk.BooleanVar(value=config.aspect_ratio_filtering)
        self.min_aspect_ratio_var = tk.DoubleVar(value=config.min_aspect_ratio)
        self.max_aspect_ratio_var = tk.DoubleVar(value=config.max_aspect_ratio)

        self.distance_transform_var = tk.BooleanVar(value=config.distance_transform)
        self.distance_transform_type_var = tk.IntVar(value=config.distance_transform_type)
        self.skeletonize_var = tk.BooleanVar(value=config.skeletonize)
        self.watershed_markers_var = tk.BooleanVar(value=config.watershed_markers)
        self.local_binary_pattern_var = tk.BooleanVar(value=config.local_binary_pattern)
        self.lbp_radius_var = tk.IntVar(value=config.lbp_radius)
        self.lbp_n_points_var = tk.IntVar(value=config.lbp_n_points)

    def _create_frame(self) -> None:
        """Create processing frame"""
        self.frame = ttk.Frame(self.parent)
        _, self.scrollable_frame, _ = create_scrollable_frame(self.frame)

        self._create_format_section()
        self._create_preprocessing_section()
        self._create_noise_reduction_section()
        self._create_enhancement_section()
        self._create_morphology_section()
        self._create_threshold_section()
        self._create_advanced_operations_section()

        self._update_resize_dependent_controls()
        self._update_grayscale_dependent_controls()
        self._update_binary_dependent_controls()

    def _create_format_section(self) -> None:
        """Create input and format section"""
        input_frame = create_labeled_frame(self.scrollable_frame, "📥 Format")
        input_frame.pack(fill=tk.X, pady=5, padx=5)

        color_frame, self.color_space_combobox = create_combobox(
            input_frame,
            "Color Space",
            self.color_space_var,
            ["Grayscale", "RGB", "BGR", "HSV", "LAB", "YUV", "YCrCb"],
            self._on_color_space_changed,
        )
        color_frame.pack(fill=tk.X, pady=2)

    def _on_color_space_changed(self, value) -> None:
        """Handle color space change"""
        self.app.processing_config.color_space = value
        self._update_grayscale_dependent_controls()
        self._update_binary_dependent_controls()
        self.app.update_image_display()

    def _create_preprocessing_section(self) -> None:
        """Create preprocessing section with resize, crop, and gamma correction"""
        preprocess_frame = create_labeled_frame(self.scrollable_frame, "🔧 Preprocessing")
        preprocess_frame.pack(fill=tk.X, pady=5, padx=5)

        create_checkbox(
            preprocess_frame,
            "Resize",
            self.resize_enabled_var,
            self._on_resize_enabled_changed,
        ).pack(anchor=tk.W, pady=2)

        resize_controls = ttk.Frame(preprocess_frame)
        resize_controls.pack(fill=tk.X, pady=2)

        self.resize_maintain_aspect_checkbox = create_checkbox(
            resize_controls,
            "Maintain Ratio",
            self.resize_maintain_aspect_var,
            self._on_resize_maintain_aspect_ratio_changed,
        )
        self.resize_maintain_aspect_checkbox.pack(anchor=tk.W, pady=2)

        width_frame, _, _ = create_slider(
            resize_controls,
            "Width",
            self.resize_width_var,
            320,
            3840,
            self._on_resize_width_changed,
        )
        width_frame.pack(fill=tk.X, pady=1)

        height_frame, _, _ = create_slider(
            resize_controls,
            "Height",
            self.resize_height_var,
            240,
            2160,
            self._on_resize_height_changed,
        )
        height_frame.pack(fill=tk.X, pady=1)

        ttk.Separator(preprocess_frame, orient="horizontal").pack(fill=tk.X, pady=5)

        create_checkbox(
            preprocess_frame,
            "Crop",
            self.crop_enabled_var,
            self._on_crop_enabled_changed,
        ).pack(anchor=tk.W, pady=2)

        bbox_frame = ttk.Frame(preprocess_frame)
        bbox_frame.pack(fill=tk.X, pady=2)

        ttk.Label(bbox_frame, text="(x1,y1,x2,y2):").pack(side=tk.LEFT, padx=(0, 5))
        bbox_entry = ttk.Entry(
            bbox_frame,
            textvariable=self.bbox_var,
            width=20,
        )
        bbox_entry.pack(side=tk.LEFT, padx=(0, 5))
        bbox_entry.bind("<KeyRelease>", self._on_bbox_key_release)
        bbox_entry.bind("<FocusOut>", self._on_bbox_focus_out)
        bbox_entry.bind("<Return>", self._on_bbox_enter)
        bbox_entry.bind("<Key>", self._on_bbox_key)

        self.bbox_var.trace_add("write", self._on_bbox_variable_changed)

        self.crop_status_label = ttk.Label(preprocess_frame, text="Crop: Disabled", foreground="gray")
        self.crop_status_label.pack(anchor=tk.W, pady=2)

        ttk.Separator(preprocess_frame, orient="horizontal").pack(fill=tk.X, pady=5)

        create_checkbox(
            preprocess_frame,
            "Gamma Correction",
            self.gamma_correction_var,
            self._on_gamma_correction_changed,
        ).pack(anchor=tk.W, pady=2)

        gamma_frame, _, _ = create_slider(
            preprocess_frame,
            "Gamma Value",
            self.gamma_value_var,
            0.1,
            3.0,
            self._on_gamma_value_changed,
        )
        gamma_frame.pack(fill=tk.X, pady=2)

    def _on_resize_enabled_changed(self) -> None:
        """Handle resize enabled change"""
        self.app.processing_config.resize_enabled = self.resize_enabled_var.get()
        self._update_crop_status()
        self._update_resize_dependent_controls()
        self.app.update_image_display()

    def _on_resize_width_changed(self, value) -> None:
        """Handle resize width change"""
        self.app.processing_config.resize_width = int(float(value))
        self.app.update_image_display()
        self._update_crop_status()

    def _on_resize_height_changed(self, value) -> None:
        """Handle resize height change"""
        self.app.processing_config.resize_height = int(float(value))
        self.app.update_image_display()
        self._update_crop_status()

    def _on_resize_maintain_aspect_ratio_changed(self) -> None:
        """Handle maintain aspect ratio change"""
        self.app.processing_config.resize_maintain_aspect_ratio = self.resize_maintain_aspect_var.get()
        self.app.update_image_display()

    def _on_gamma_correction_changed(self) -> None:
        """Handle gamma correction change"""
        self.app.processing_config.gamma_correction = self.gamma_correction_var.get()
        self.app.update_image_display()

    def _on_gamma_value_changed(self, value) -> None:
        """Handle gamma value change"""
        self.app.processing_config.gamma_value = float(value)
        self.app.update_image_display()

    def _on_crop_enabled_changed(self) -> None:
        """Handle crop enabled change"""
        self.app.processing_config.crop_enabled = self.crop_enabled_var.get()
        self._update_crop_status()
        self.app.update_image_display()

    def _update_crop_status(self) -> None:
        """Update crop status label with real-time feedback"""
        if not self.crop_enabled_var.get():
            self.crop_status_label.config(text="Crop: Disabled", foreground="gray")
            return

        if self.app.processed_image is None:
            self.crop_status_label.config(text="Crop: No image", foreground="gray")
            return

        bbox_str = self.bbox_var.get()
        bbox = self._parse_bbox(bbox_str)

        if bbox is None:
            if bbox_str.strip():
                self.crop_status_label.config(text="Crop: Invalid format - use x1,y1,x2,y2", foreground="red")
            else:
                self.crop_status_label.config(text="Crop: Enter coordinates", foreground="orange")
            return

        x1, y1, x2, y2 = bbox

        img_height, img_width = self.app.processed_image.shape[:2]

        if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
            self.crop_status_label.config(
                text=f"Crop: Out of bounds - Image: {img_width}×{img_height}",
                foreground="red",
            )
            return

        if x2 <= x1 or y2 <= y1:
            self.crop_status_label.config(
                text=f"Crop: Invalid coordinates ({x1},{y1}) to ({x2},{y2})",
                foreground="red",
            )
            return

        width = x2 - x1
        height = y2 - y1

        self.crop_status_label.config(
            text=f"Crop: {width}×{height} | Image: {img_width}×{img_height}",
            foreground="green",
        )

    def _format_bbox(self, bbox) -> str:
        """Format bbox tuple to string for display"""
        if bbox is None:
            return ""

        return f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

    def _parse_bbox(self, bbox_str: str) -> tuple[int, int, int, int] | None:
        """Parse bbox string to tuple with flexible formatting"""
        if not bbox_str or bbox_str.strip() == "":
            return None

        try:
            cleaned_str = bbox_str.replace(" ", "").replace("(", "").replace(")", "")
            coords = [int(x.strip()) for x in cleaned_str.split(",") if x.strip()]

            if len(coords) == 4:
                return (coords[0], coords[1], coords[2], coords[3])
            if len(coords) > 4:
                return (coords[0], coords[1], coords[2], coords[3])
        except (ValueError, AttributeError):
            pass
        return None

    def _on_bbox_key_release(self, event) -> None:
        """Handle bbox key release"""
        self._on_bbox_changed()

    def _on_bbox_focus_out(self, event) -> None:
        """Handle bbox focus out"""
        self._on_bbox_changed()

    def _on_bbox_enter(self, event) -> None:
        """Handle bbox enter"""
        self._on_bbox_changed()

    def _on_bbox_key(self, event) -> None:
        """Handle bbox key"""
        self._on_bbox_changed()

    def _on_bbox_variable_changed(self, *args) -> None:
        """Handle bbox variable change"""
        self._on_bbox_changed()

    def _on_bbox_changed(self) -> None:
        """Handle bbox change with real-time updates"""
        bbox_str = self.bbox_var.get()
        bbox = self._parse_bbox(bbox_str)

        if bbox is not None:
            x1, y1, x2, y2 = bbox

            self.app.processing_config.bbox = bbox
            self._update_crop_status()

            if x2 > x1 and y2 > y1:
                self.app.update_image_display()
        else:
            self._update_crop_status()

    def _create_threshold_section(self) -> None:
        """Create consolidated threshold section"""
        threshold_frame = create_labeled_frame(self.scrollable_frame, "🎯 Threshold")
        threshold_frame.pack(fill=tk.X, pady=5, padx=5)

        self.threshold_enabled_checkbox = create_checkbox(
            threshold_frame,
            "Threshold",
            self.threshold_enabled_var,
            self._on_threshold_enabled_changed,
        )
        self.threshold_enabled_checkbox.pack(anchor=tk.W, pady=1)

        threshold_type_frame, self.threshold_combobox = create_combobox(
            threshold_frame,
            "Threshold Type",
            self.threshold_type_var,
            ["BINARY", "BINARY_INV", "OTSU_BINARY", "ADAPTIVE_MEAN", "ADAPTIVE_GAUSSIAN"],
            self._on_threshold_type_changed,
        )
        threshold_type_frame.pack(fill=tk.X, pady=1)

        threshold_value_frame, _, _ = create_slider(
            threshold_frame,
            "Threshold Value",
            self.threshold_value_var,
            0,
            255,
            self._on_threshold_value_changed,
        )
        threshold_value_frame.pack(fill=tk.X, pady=1)

        adaptive_block_frame, _, _ = create_slider(
            threshold_frame,
            "Adaptive Block Size",
            self.adaptive_block_size_var,
            3,
            31,
            self._on_adaptive_block_size_changed,
        )
        adaptive_block_frame.pack(fill=tk.X, pady=1)

        adaptive_c_frame, _, _ = create_slider(
            threshold_frame,
            "Adaptive C",
            self.adaptive_c_var,
            0,
            20,
            self._on_adaptive_c_changed,
        )
        adaptive_c_frame.pack(fill=tk.X, pady=1)

        ttk.Separator(threshold_frame, orient="horizontal").pack(fill=tk.X, pady=3)

        self.multi_otsu_checkbox = create_checkbox(
            threshold_frame,
            "Multi-OTSU Thresholding",
            self.multi_otsu_var,
            self._on_multi_otsu_changed,
        )
        self.multi_otsu_checkbox.pack(anchor=tk.W, pady=1)

        multi_otsu_classes_frame, _, _ = create_slider(
            threshold_frame,
            "OTSU Classes",
            self.multi_otsu_classes_var,
            2,
            5,
            self._on_multi_otsu_classes_changed,
        )
        multi_otsu_classes_frame.pack(fill=tk.X, pady=1)

    def _on_threshold_enabled_changed(self) -> None:
        """Handle threshold enabled change"""
        self.app.processing_config.threshold_enabled = self.threshold_enabled_var.get()
        self._update_binary_dependent_controls()
        self.app.update_image_display()

    def _on_threshold_type_changed(self, value: str | None = None) -> None:
        """Handle threshold type change"""
        self.app.processing_config.threshold_type = self.threshold_type_var.get()
        self.app.update_image_display()

    def _on_threshold_value_changed(self, value) -> None:
        """Handle threshold value change"""
        self.app.processing_config.threshold_value = int(float(value))
        self.app.update_image_display()

    def _on_adaptive_block_size_changed(self, value) -> None:
        """Handle adaptive block size change"""
        self.app.processing_config.adaptive_block_size = int(float(value))
        self.app.update_image_display()

    def _on_adaptive_c_changed(self, value) -> None:
        """Handle adaptive C change"""
        self.app.processing_config.adaptive_c = int(float(value))
        self.app.update_image_display()

    def _on_multi_otsu_classes_changed(self, value) -> None:
        """Handle multi-OTSU classes change"""
        self.app.processing_config.multi_otsu_classes = int(float(value))
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

    def _create_advanced_noise_section(self) -> None:
        """Create advanced noise reduction section"""
        noise_advanced_frame = create_labeled_frame(self.scrollable_frame, "🔧 Advanced Noise Reduction")
        noise_advanced_frame.pack(fill=tk.X, pady=5, padx=5)

        create_checkbox(
            noise_advanced_frame,
            "NL-Means Denoising",
            self.denoise_nl_means_var,
            self._on_denoise_nl_means_changed,
        ).pack(anchor=tk.W, pady=2)

        denoise_h_frame, _, _ = create_slider(
            noise_advanced_frame,
            "Denoise H",
            self.denoise_h_var,
            1.0,
            30.0,
            self._on_denoise_h_changed,
        )
        denoise_h_frame.pack(fill=tk.X, pady=2)

        denoise_template_frame, _, _ = create_slider(
            noise_advanced_frame,
            "Template Window",
            self.denoise_template_window_var,
            3,
            15,
            self._on_denoise_template_window_changed,
        )
        denoise_template_frame.pack(fill=tk.X, pady=2)

        denoise_search_frame, _, _ = create_slider(
            noise_advanced_frame,
            "Search Window",
            self.denoise_search_window_var,
            7,
            35,
            self._on_denoise_search_window_changed,
        )
        denoise_search_frame.pack(fill=tk.X, pady=2)

        create_checkbox(
            noise_advanced_frame,
            "Edge Preserving Filter",
            self.edge_preserving_filter_var,
            self._on_edge_preserving_filter_changed,
        ).pack(anchor=tk.W, pady=2)

        edge_sigma_s_frame, _, _ = create_slider(
            noise_advanced_frame,
            "Edge Sigma S",
            self.edge_sigma_s_var,
            10.0,
            200.0,
            self._on_edge_sigma_s_changed,
        )
        edge_sigma_s_frame.pack(fill=tk.X, pady=2)

        edge_sigma_r_frame, _, _ = create_slider(
            noise_advanced_frame,
            "Edge Sigma R",
            self.edge_sigma_r_var,
            0.1,
            1.0,
            self._on_edge_sigma_r_changed,
        )
        edge_sigma_r_frame.pack(fill=tk.X, pady=2)

        create_checkbox(
            noise_advanced_frame,
            "Noise Reduction Bilateral",
            self.noise_reduction_bilateral_var,
            self._on_noise_reduction_bilateral_changed,
        ).pack(anchor=tk.W, pady=2)

        bilateral_iterations_frame, _, _ = create_slider(
            noise_advanced_frame,
            "Bilateral Iterations",
            self.bilateral_iterations_var,
            1,
            5,
            self._on_bilateral_iterations_changed,
        )
        bilateral_iterations_frame.pack(fill=tk.X, pady=2)

    def _on_denoise_nl_means_changed(self) -> None:
        """Handle NL-means denoising change"""
        self.app.processing_config.denoise_nl_means = self.denoise_nl_means_var.get()
        self.app.update_image_display()

    def _on_denoise_h_changed(self, value) -> None:
        """Handle denoise H change"""
        self.app.processing_config.denoise_h = float(value)
        self.app.update_image_display()

    def _on_denoise_template_window_changed(self, value) -> None:
        """Handle denoise template window change"""
        self.app.processing_config.denoise_template_window = int(float(value))
        self.app.update_image_display()

    def _on_denoise_search_window_changed(self, value) -> None:
        """Handle denoise search window change"""
        self.app.processing_config.denoise_search_window = int(float(value))
        self.app.update_image_display()

    def _on_edge_preserving_filter_changed(self) -> None:
        """Handle edge preserving filter change"""
        self.app.processing_config.edge_preserving_filter = self.edge_preserving_filter_var.get()
        self.app.update_image_display()

    def _on_edge_sigma_s_changed(self, value) -> None:
        """Handle edge sigma S change"""
        self.app.processing_config.edge_sigma_s = float(value)
        self.app.update_image_display()

    def _on_edge_sigma_r_changed(self, value) -> None:
        """Handle edge sigma R change"""
        self.app.processing_config.edge_sigma_r = float(value)
        self.app.update_image_display()

    def _on_noise_reduction_bilateral_changed(self) -> None:
        """Handle noise reduction bilateral change"""
        self.app.processing_config.noise_reduction_bilateral = self.noise_reduction_bilateral_var.get()
        self.app.update_image_display()

    def _on_bilateral_iterations_changed(self, value) -> None:
        """Handle bilateral iterations change"""
        self.app.processing_config.bilateral_iterations = int(float(value))
        self.app.update_image_display()

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

    def _create_advanced_filters_section(self) -> None:
        """Create advanced filters section"""
        filters_advanced_frame = create_labeled_frame(self.scrollable_frame, "🔧 Advanced Filters")
        filters_advanced_frame.pack(fill=tk.X, pady=5, padx=5)

        bilateral_sigma_color_frame, _, _ = create_slider(
            filters_advanced_frame,
            "Bilateral Sigma Color",
            self.bilateral_sigma_color_var,
            10,
            150,
            self._on_bilateral_sigma_color_changed,
        )
        bilateral_sigma_color_frame.pack(fill=tk.X, pady=2)

        bilateral_sigma_space_frame, _, _ = create_slider(
            filters_advanced_frame,
            "Bilateral Sigma Space",
            self.bilateral_sigma_space_var,
            10,
            150,
            self._on_bilateral_sigma_space_changed,
        )
        bilateral_sigma_space_frame.pack(fill=tk.X, pady=2)

        gaussian_sigma_frame, _, _ = create_slider(
            filters_advanced_frame,
            "Gaussian Sigma",
            self.gaussian_sigma_var,
            0.0,
            5.0,
            self._on_gaussian_sigma_changed,
        )
        gaussian_sigma_frame.pack(fill=tk.X, pady=2)

        create_checkbox(
            filters_advanced_frame,
            "Background Subtraction",
            self.background_subtraction_var,
            self._on_background_subtraction_changed,
        ).pack(anchor=tk.W, pady=2)

        bg_threshold_frame, _, _ = create_slider(
            filters_advanced_frame,
            "BG Threshold",
            self.bg_threshold_var,
            0,
            100,
            self._on_bg_threshold_changed,
        )
        bg_threshold_frame.pack(fill=tk.X, pady=2)

    def _on_bilateral_sigma_color_changed(self, value) -> None:
        """Handle bilateral sigma color change"""
        self.app.processing_config.bilateral_sigma_color = int(float(value))
        self.app.update_image_display()

    def _on_bilateral_sigma_space_changed(self, value) -> None:
        """Handle bilateral sigma space change"""
        self.app.processing_config.bilateral_sigma_space = int(float(value))
        self.app.update_image_display()

    def _on_gaussian_sigma_changed(self, value) -> None:
        """Handle gaussian sigma change"""
        self.app.processing_config.gaussian_sigma = float(value)
        self.app.update_image_display()

    def _on_background_subtraction_changed(self) -> None:
        """Handle background subtraction change"""
        self.app.processing_config.background_subtraction = self.background_subtraction_var.get()
        self.app.update_image_display()

    def _on_bg_threshold_changed(self, value) -> None:
        """Handle background threshold change"""
        self.app.processing_config.bg_threshold = int(float(value))
        self.app.update_image_display()

    def _create_noise_reduction_section(self) -> None:
        """Create consolidated noise reduction section"""
        noise_frame = create_labeled_frame(self.scrollable_frame, "🔇 Noise Reduction")
        noise_frame.pack(fill=tk.X, pady=5, padx=5)

        create_checkbox(
            noise_frame,
            "Bilateral Filter",
            self.bilateral_filter_var,
            self._on_bilateral_filter_changed,
        ).pack(anchor=tk.W, pady=1)

        bilateral_controls = ttk.Frame(noise_frame)
        bilateral_controls.pack(fill=tk.X, pady=1)

        bilateral_d_frame, _, _ = create_slider(
            bilateral_controls,
            "Bilateral D",
            self.bilateral_d_var,
            1,
            15,
            self._on_bilateral_d_changed,
        )
        bilateral_d_frame.pack(fill=tk.X, pady=1)

        bilateral_sigma_color_frame, _, _ = create_slider(
            bilateral_controls,
            "Sigma Color",
            self.bilateral_sigma_color_var,
            10,
            150,
            self._on_bilateral_sigma_color_changed,
        )
        bilateral_sigma_color_frame.pack(fill=tk.X, pady=1)

        bilateral_sigma_space_frame, _, _ = create_slider(
            bilateral_controls,
            "Sigma Space",
            self.bilateral_sigma_space_var,
            10,
            150,
            self._on_bilateral_sigma_space_changed,
        )
        bilateral_sigma_space_frame.pack(fill=tk.X, pady=1)

        create_checkbox(
            noise_frame,
            "Gaussian Blur",
            self.gaussian_blur_var,
            self._on_gaussian_blur_changed,
        ).pack(anchor=tk.W, pady=1)

        gaussian_controls = ttk.Frame(noise_frame)
        gaussian_controls.pack(fill=tk.X, pady=1)

        gaussian_kernel_frame, _, _ = create_slider(
            gaussian_controls,
            "Kernel Size",
            self.gaussian_kernel_var,
            1,
            10,
            self._on_gaussian_kernel_changed,
        )
        gaussian_kernel_frame.pack(fill=tk.X, pady=1)

        gaussian_sigma_frame, _, _ = create_slider(
            gaussian_controls,
            "Sigma",
            self.gaussian_sigma_var,
            0.0,
            5.0,
            self._on_gaussian_sigma_changed,
        )
        gaussian_sigma_frame.pack(fill=tk.X, pady=1)

        create_checkbox(
            noise_frame,
            "Median Filter",
            self.median_filter_var,
            self._on_median_filter_changed,
        ).pack(anchor=tk.W, pady=1)

        median_kernel_frame, _, _ = create_slider(
            noise_frame,
            "Median Kernel",
            self.median_kernel_var,
            1,
            10,
            self._on_median_kernel_changed,
        )
        median_kernel_frame.pack(fill=tk.X, pady=1)

        ttk.Separator(noise_frame, orient="horizontal").pack(fill=tk.X, pady=3)

        create_checkbox(
            noise_frame,
            "NL-Means Denoising",
            self.denoise_nl_means_var,
            self._on_denoise_nl_means_changed,
        ).pack(anchor=tk.W, pady=1)

        nlmeans_controls = ttk.Frame(noise_frame)
        nlmeans_controls.pack(fill=tk.X, pady=1)

        denoise_h_frame, _, _ = create_slider(
            nlmeans_controls,
            "Denoise H",
            self.denoise_h_var,
            1.0,
            30.0,
            self._on_denoise_h_changed,
        )
        denoise_h_frame.pack(fill=tk.X, pady=1)

        denoise_template_frame, _, _ = create_slider(
            nlmeans_controls,
            "Template Window",
            self.denoise_template_window_var,
            3,
            15,
            self._on_denoise_template_window_changed,
        )
        denoise_template_frame.pack(fill=tk.X, pady=1)

        denoise_search_frame, _, _ = create_slider(
            nlmeans_controls,
            "Search Window",
            self.denoise_search_window_var,
            7,
            35,
            self._on_denoise_search_window_changed,
        )
        denoise_search_frame.pack(fill=tk.X, pady=1)

        create_checkbox(
            noise_frame,
            "Edge Preserving Filter",
            self.edge_preserving_filter_var,
            self._on_edge_preserving_filter_changed,
        ).pack(anchor=tk.W, pady=1)

        edge_controls = ttk.Frame(noise_frame)
        edge_controls.pack(fill=tk.X, pady=1)

        edge_sigma_s_frame, _, _ = create_slider(
            edge_controls,
            "Edge Sigma S",
            self.edge_sigma_s_var,
            10.0,
            200.0,
            self._on_edge_sigma_s_changed,
        )
        edge_sigma_s_frame.pack(fill=tk.X, pady=1)

        edge_sigma_r_frame, _, _ = create_slider(
            edge_controls,
            "Edge Sigma R",
            self.edge_sigma_r_var,
            0.1,
            1.0,
            self._on_edge_sigma_r_changed,
        )
        edge_sigma_r_frame.pack(fill=tk.X, pady=1)

        create_checkbox(
            noise_frame,
            "Additional Bilateral Iterations",
            self.noise_reduction_bilateral_var,
            self._on_noise_reduction_bilateral_changed,
        ).pack(anchor=tk.W, pady=1)

        bilateral_iterations_frame, _, _ = create_slider(
            noise_frame,
            "Iterations",
            self.bilateral_iterations_var,
            1,
            5,
            self._on_bilateral_iterations_changed,
        )
        bilateral_iterations_frame.pack(fill=tk.X, pady=1)

    def _create_enhancement_section(self) -> None:
        """Create consolidated enhancement section"""
        enhance_frame = create_labeled_frame(self.scrollable_frame, "✨ Enhancement")
        enhance_frame.pack(fill=tk.X, pady=5, padx=5)

        self.histogram_equalization_checkbox = create_checkbox(
            enhance_frame,
            "Histogram Equalization",
            self.histogram_equalization_var,
            self._on_histogram_eq_changed,
        )
        self.histogram_equalization_checkbox.pack(anchor=tk.W, pady=1)

        self.clahe_checkbox = create_checkbox(enhance_frame, "CLAHE", self.clahe_var, self._on_clahe_changed)
        self.clahe_checkbox.pack(anchor=tk.W, pady=1)

        clahe_controls = ttk.Frame(enhance_frame)
        clahe_controls.pack(fill=tk.X, pady=1)

        clahe_clip_frame, _, _ = create_slider(
            clahe_controls,
            "Clip Limit",
            self.clahe_clip_limit_var,
            0.5,
            10.0,
            self._on_clahe_clip_changed,
        )
        clahe_clip_frame.pack(fill=tk.X, pady=1)

        clahe_tile_frame, _, _ = create_slider(
            clahe_controls,
            "Tile Size",
            self.clahe_tile_size_var,
            4,
            16,
            self._on_clahe_tile_size_changed,
        )
        clahe_tile_frame.pack(fill=tk.X, pady=1)

        self.adaptive_hist_eq_checkbox = create_checkbox(
            enhance_frame,
            "Adaptive Histogram Equalization",
            self.adaptive_hist_eq_var,
            self._on_adaptive_hist_eq_changed,
        )
        self.adaptive_hist_eq_checkbox.pack(anchor=tk.W, pady=1)

        ttk.Separator(enhance_frame, orient="horizontal").pack(fill=tk.X, pady=3)

        create_checkbox(
            enhance_frame,
            "Intensity Normalization",
            self.intensity_normalization_var,
            self._on_intensity_normalization_changed,
        ).pack(anchor=tk.W, pady=1)

        create_checkbox(
            enhance_frame,
            "Contrast Stretching",
            self.contrast_stretching_var,
            self._on_contrast_stretching_changed,
        ).pack(anchor=tk.W, pady=1)

        ttk.Separator(enhance_frame, orient="horizontal").pack(fill=tk.X, pady=3)

        create_checkbox(enhance_frame, "Sharpen", self.sharpen_var, self._on_sharpen_changed).pack(anchor=tk.W, pady=1)

        sharpen_strength_frame, _, _ = create_slider(
            enhance_frame,
            "Sharpen Strength",
            self.sharpen_strength_var,
            0.0,
            1.0,
            self._on_sharpen_strength_changed,
        )
        sharpen_strength_frame.pack(fill=tk.X, pady=1)

        create_checkbox(
            enhance_frame,
            "Unsharp Mask",
            self.unsharp_mask_var,
            self._on_unsharp_mask_changed,
        ).pack(anchor=tk.W, pady=1)

        unsharp_strength_frame, _, _ = create_slider(
            enhance_frame,
            "Unsharp Strength",
            self.unsharp_strength_var,
            0.5,
            3.0,
            self._on_unsharp_strength_changed,
        )
        unsharp_strength_frame.pack(fill=tk.X, pady=1)

        self.edge_enhancement_checkbox = create_checkbox(
            enhance_frame,
            "Edge Enhancement",
            self.edge_enhancement_var,
            self._on_edge_enhancement_changed,
        )
        self.edge_enhancement_checkbox.pack(anchor=tk.W, pady=1)

        edge_strength_frame, _, _ = create_slider(
            enhance_frame,
            "Edge Strength",
            self.edge_strength_var,
            0.0,
            3.0,
            self._on_edge_strength_changed,
        )
        edge_strength_frame.pack(fill=tk.X, pady=1)

        ttk.Separator(enhance_frame, orient="horizontal").pack(fill=tk.X, pady=3)

        create_checkbox(
            enhance_frame,
            "Text Enhancement",
            self.text_enhancement_var,
            self._on_text_enhancement_changed,
        ).pack(anchor=tk.W, pady=1)

        text_kernel_frame, _, _ = create_slider(
            enhance_frame,
            "Text Kernel Size",
            self.text_kernel_size_var,
            1,
            5,
            self._on_text_kernel_size_changed,
        )
        text_kernel_frame.pack(fill=tk.X, pady=1)

        create_checkbox(
            enhance_frame,
            "Detail Enhancement",
            self.detail_enhancement_var,
            self._on_detail_enhancement_changed,
        ).pack(anchor=tk.W, pady=1)

        detail_controls = ttk.Frame(enhance_frame)
        detail_controls.pack(fill=tk.X, pady=1)

        detail_sigma_s_frame, _, _ = create_slider(
            detail_controls,
            "Detail Sigma S",
            self.detail_sigma_s_var,
            1.0,
            50.0,
            self._on_detail_sigma_s_changed,
        )
        detail_sigma_s_frame.pack(fill=tk.X, pady=1)

        detail_sigma_r_frame, _, _ = create_slider(
            detail_controls,
            "Detail Sigma R",
            self.detail_sigma_r_var,
            0.05,
            1.0,
            self._on_detail_sigma_r_changed,
        )
        detail_sigma_r_frame.pack(fill=tk.X, pady=1)

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

    def _create_advanced_enhancement_section(self) -> None:
        """Create advanced enhancement section"""
        enhance_advanced_frame = create_labeled_frame(self.scrollable_frame, "✨ Advanced Enhancement")
        enhance_advanced_frame.pack(fill=tk.X, pady=5, padx=5)

        create_checkbox(
            enhance_advanced_frame,
            "Text Enhancement",
            self.text_enhancement_var,
            self._on_text_enhancement_changed,
        ).pack(anchor=tk.W, pady=2)

        text_kernel_frame, _, _ = create_slider(
            enhance_advanced_frame,
            "Text Kernel Size",
            self.text_kernel_size_var,
            1,
            5,
            self._on_text_kernel_size_changed,
        )
        text_kernel_frame.pack(fill=tk.X, pady=2)

        create_checkbox(
            enhance_advanced_frame,
            "Detail Enhancement",
            self.detail_enhancement_var,
            self._on_detail_enhancement_changed,
        ).pack(anchor=tk.W, pady=2)

        detail_sigma_s_frame, _, _ = create_slider(
            enhance_advanced_frame,
            "Detail Sigma S",
            self.detail_sigma_s_var,
            1.0,
            50.0,
            self._on_detail_sigma_s_changed,
        )
        detail_sigma_s_frame.pack(fill=tk.X, pady=2)

        detail_sigma_r_frame, _, _ = create_slider(
            enhance_advanced_frame,
            "Detail Sigma R",
            self.detail_sigma_r_var,
            0.05,
            1.0,
            self._on_detail_sigma_r_changed,
        )
        detail_sigma_r_frame.pack(fill=tk.X, pady=2)

        edge_strength_frame, _, _ = create_slider(
            enhance_advanced_frame,
            "Edge Strength",
            self.edge_strength_var,
            0.0,
            3.0,
            self._on_edge_strength_changed,
        )
        edge_strength_frame.pack(fill=tk.X, pady=2)

        create_checkbox(
            enhance_advanced_frame,
            "Unsharp Mask",
            self.unsharp_mask_var,
            self._on_unsharp_mask_changed,
        ).pack(anchor=tk.W, pady=2)

        unsharp_strength_frame, _, _ = create_slider(
            enhance_advanced_frame,
            "Unsharp Strength",
            self.unsharp_strength_var,
            0.5,
            3.0,
            self._on_unsharp_strength_changed,
        )
        unsharp_strength_frame.pack(fill=tk.X, pady=2)

    def _on_text_enhancement_changed(self) -> None:
        """Handle text enhancement change"""
        self.app.processing_config.text_enhancement = self.text_enhancement_var.get()
        self.app.update_image_display()

    def _on_text_kernel_size_changed(self, value) -> None:
        """Handle text kernel size change"""
        self.app.processing_config.text_kernel_size = int(float(value))
        self.app.update_image_display()

    def _on_detail_enhancement_changed(self) -> None:
        """Handle detail enhancement change"""
        self.app.processing_config.detail_enhancement = self.detail_enhancement_var.get()
        self.app.update_image_display()

    def _on_detail_sigma_s_changed(self, value) -> None:
        """Handle detail sigma S change"""
        self.app.processing_config.detail_sigma_s = float(value)
        self.app.update_image_display()

    def _on_detail_sigma_r_changed(self, value) -> None:
        """Handle detail sigma R change"""
        self.app.processing_config.detail_sigma_r = float(value)
        self.app.update_image_display()

    def _on_edge_strength_changed(self, value) -> None:
        """Handle edge strength change"""
        self.app.processing_config.edge_strength = float(value)
        self.app.update_image_display()

    def _on_unsharp_mask_changed(self) -> None:
        """Handle unsharp mask change"""
        self.app.processing_config.unsharp_mask = self.unsharp_mask_var.get()
        self.app.update_image_display()

    def _on_unsharp_strength_changed(self, value) -> None:
        """Handle unsharp strength change"""
        self.app.processing_config.unsharp_strength = float(value)
        self.app.update_image_display()

    def _create_morphology_section(self) -> None:
        """Create consolidated morphology section"""
        morph_frame = create_labeled_frame(self.scrollable_frame, "🔄 Morphology")
        morph_frame.pack(fill=tk.X, pady=5, padx=5)

        create_checkbox(
            morph_frame,
            "Basic Morphology",
            self.morphology_var,
            self._on_morphology_changed,
        ).pack(anchor=tk.W, pady=1)

        basic_morph_controls = ttk.Frame(morph_frame)
        basic_morph_controls.pack(fill=tk.X, pady=1)

        morph_kernel_frame, _, _ = create_slider(
            basic_morph_controls,
            "Kernel Size",
            self.morph_kernel_size_var,
            1,
            10,
            self._on_morph_kernel_changed,
        )
        morph_kernel_frame.pack(fill=tk.X, pady=1)

        create_checkbox(
            basic_morph_controls,
            "Opening",
            self.morph_open_var,
            self._on_morph_open_changed,
        ).pack(anchor=tk.W, pady=1)

        create_checkbox(
            basic_morph_controls,
            "Closing",
            self.morph_close_var,
            self._on_morph_close_changed,
        ).pack(anchor=tk.W, pady=1)

        ttk.Separator(morph_frame, orient="horizontal").pack(fill=tk.X, pady=3)

        create_checkbox(
            morph_frame,
            "Stroke Width Normalization",
            self.stroke_width_normalization_var,
            self._on_stroke_width_normalization_changed,
        ).pack(anchor=tk.W, pady=1)

        stroke_iterations_frame, _, _ = create_slider(
            morph_frame,
            "Stroke Iterations",
            self.stroke_iterations_var,
            1,
            5,
            self._on_stroke_iterations_changed,
        )
        stroke_iterations_frame.pack(fill=tk.X, pady=1)

        create_checkbox(
            morph_frame,
            "Top Hat",
            self.tophat_var,
            self._on_tophat_changed,
        ).pack(anchor=tk.W, pady=1)

        tophat_kernel_frame, _, _ = create_slider(
            morph_frame,
            "Top Hat Kernel Size",
            self.tophat_kernel_size_var,
            1,
            10,
            self._on_tophat_kernel_size_changed,
        )
        tophat_kernel_frame.pack(fill=tk.X, pady=1)

        create_checkbox(
            morph_frame,
            "Black Hat",
            self.blackhat_var,
            self._on_blackhat_changed,
        ).pack(anchor=tk.W, pady=1)

        blackhat_kernel_frame, _, _ = create_slider(
            morph_frame,
            "Black Hat Kernel Size",
            self.blackhat_kernel_size_var,
            1,
            10,
            self._on_blackhat_kernel_size_changed,
        )
        blackhat_kernel_frame.pack(fill=tk.X, pady=1)

        create_checkbox(
            morph_frame,
            "Gradient",
            self.gradient_var,
            self._on_gradient_changed,
        ).pack(anchor=tk.W, pady=1)

        gradient_kernel_frame, _, _ = create_slider(
            morph_frame,
            "Gradient Kernel Size",
            self.gradient_kernel_size_var,
            1,
            10,
            self._on_gradient_kernel_size_changed,
        )
        gradient_kernel_frame.pack(fill=tk.X, pady=1)

        create_checkbox(
            morph_frame,
            "Morphological Gradient",
            self.morphological_gradient_var,
            self._on_morphological_gradient_changed,
        ).pack(anchor=tk.W, pady=1)

        morph_grad_kernel_frame, _, _ = create_slider(
            morph_frame,
            "Morph Gradient Kernel",
            self.morphological_gradient_kernel_var,
            1,
            10,
            self._on_morphological_gradient_kernel_changed,
        )
        morph_grad_kernel_frame.pack(fill=tk.X, pady=1)

        # Character operations
        ttk.Separator(morph_frame, orient="horizontal").pack(fill=tk.X, pady=3)

        create_checkbox(
            morph_frame,
            "Character Separation",
            self.character_separation_var,
            self._on_character_separation_changed,
        ).pack(anchor=tk.W, pady=1)

        char_sep_kernel_frame, _, _ = create_slider(
            morph_frame,
            "Char Sep Kernel Size",
            self.char_sep_kernel_size_var,
            1,
            5,
            self._on_char_sep_kernel_size_changed,
        )
        char_sep_kernel_frame.pack(fill=tk.X, pady=1)

        create_checkbox(
            morph_frame,
            "Character Dilation",
            self.character_dilation_var,
            self._on_character_dilation_changed,
        ).pack(anchor=tk.W, pady=1)

        dilation_controls = ttk.Frame(morph_frame)
        dilation_controls.pack(fill=tk.X, pady=1)

        dilation_kernel_frame, _, _ = create_slider(
            dilation_controls,
            "Dilation Kernel Size",
            self.dilation_kernel_size_var,
            1,
            5,
            self._on_dilation_kernel_size_changed,
        )
        dilation_kernel_frame.pack(fill=tk.X, pady=1)

        dilation_iterations_frame, _, _ = create_slider(
            dilation_controls,
            "Dilation Iterations",
            self.dilation_iterations_var,
            1,
            5,
            self._on_dilation_iterations_changed,
        )
        dilation_iterations_frame.pack(fill=tk.X, pady=1)

        create_checkbox(
            morph_frame,
            "Character Erosion",
            self.character_erosion_var,
            self._on_character_erosion_changed,
        ).pack(anchor=tk.W, pady=1)

        erosion_controls = ttk.Frame(morph_frame)
        erosion_controls.pack(fill=tk.X, pady=1)

        erosion_kernel_frame, _, _ = create_slider(
            erosion_controls,
            "Erosion Kernel Size",
            self.erosion_kernel_size_var,
            1,
            5,
            self._on_erosion_kernel_size_changed,
        )
        erosion_kernel_frame.pack(fill=tk.X, pady=1)

        erosion_iterations_frame, _, _ = create_slider(
            erosion_controls,
            "Erosion Iterations",
            self.erosion_iterations_var,
            1,
            5,
            self._on_erosion_iterations_changed,
        )
        erosion_iterations_frame.pack(fill=tk.X, pady=1)

        ttk.Separator(morph_frame, orient="horizontal").pack(fill=tk.X, pady=3)

        create_checkbox(
            morph_frame,
            "Remove Vertical Lines",
            self.vertical_line_removal_var,
            self._on_vertical_line_removal_changed,
        ).pack(anchor=tk.W, pady=1)

        vertical_kernel_frame, _, _ = create_slider(
            morph_frame,
            "Vertical Kernel Size",
            self.vertical_kernel_size_var,
            1,
            15,
            self._on_vertical_kernel_size_changed,
        )
        vertical_kernel_frame.pack(fill=tk.X, pady=1)

        create_checkbox(
            morph_frame,
            "Remove Horizontal Lines",
            self.horizontal_line_removal_var,
            self._on_horizontal_line_removal_changed,
        ).pack(anchor=tk.W, pady=1)

        horizontal_kernel_frame, _, _ = create_slider(
            morph_frame,
            "Horizontal Kernel Size",
            self.horizontal_kernel_size_var,
            1,
            15,
            self._on_horizontal_kernel_size_changed,
        )
        horizontal_kernel_frame.pack(fill=tk.X, pady=1)

        create_checkbox(
            morph_frame,
            "Hough Lines Removal",
            self.hough_lines_removal_var,
            self._on_hough_lines_removal_changed,
        ).pack(anchor=tk.W, pady=1)

        self.noise_dots_removal_checkbox = create_checkbox(
            morph_frame,
            "Remove Noise Dots",
            self.noise_dots_removal_var,
            self._on_noise_dots_removal_changed,
        )
        self.noise_dots_removal_checkbox.pack(anchor=tk.W, pady=1)

        min_contour_frame, _, _ = create_slider(
            morph_frame,
            "Min Contour Area",
            self.min_contour_area_var,
            1,
            100,
            self._on_min_contour_area_changed,
        )
        min_contour_frame.pack(fill=tk.X, pady=1)

    def _on_morph_open_changed(self) -> None:
        """Handle morphology open change"""
        self.app.processing_config.morph_open = self.morph_open_var.get()
        self.app.update_image_display()

    def _on_morph_close_changed(self) -> None:
        """Handle morphology close change"""
        self.app.processing_config.morph_close = self.morph_close_var.get()
        self.app.update_image_display()

    def _on_stroke_iterations_changed(self, value) -> None:
        """Handle stroke iterations change"""
        self.app.processing_config.stroke_iterations = int(float(value))
        self.app.update_image_display()

    def _on_tophat_kernel_size_changed(self, value) -> None:
        """Handle tophat kernel size change"""
        self.app.processing_config.tophat_kernel_size = int(float(value))
        self.app.update_image_display()

    def _on_blackhat_kernel_size_changed(self, value) -> None:
        """Handle blackhat kernel size change"""
        self.app.processing_config.blackhat_kernel_size = int(float(value))
        self.app.update_image_display()

    def _on_gradient_kernel_size_changed(self, value) -> None:
        """Handle gradient kernel size change"""
        self.app.processing_config.gradient_kernel_size = int(float(value))
        self.app.update_image_display()

    def _on_morphological_gradient_changed(self) -> None:
        """Handle morphological gradient change"""
        self.app.processing_config.morphological_gradient = self.morphological_gradient_var.get()
        self.app.update_image_display()

    def _on_morphological_gradient_kernel_changed(self, value) -> None:
        """Handle morphological gradient kernel change"""
        self.app.processing_config.morphological_gradient_kernel = int(float(value))
        self.app.update_image_display()

    def _on_char_sep_kernel_size_changed(self, value) -> None:
        """Handle character separation kernel size change"""
        self.app.processing_config.char_sep_kernel_size = int(float(value))
        self.app.update_image_display()

    def _on_dilation_kernel_size_changed(self, value) -> None:
        """Handle dilation kernel size change"""
        self.app.processing_config.dilation_kernel_size = int(float(value))
        self.app.update_image_display()

    def _on_dilation_iterations_changed(self, value) -> None:
        """Handle dilation iterations change"""
        self.app.processing_config.dilation_iterations = int(float(value))
        self.app.update_image_display()

    def _on_erosion_kernel_size_changed(self, value) -> None:
        """Handle erosion kernel size change"""
        self.app.processing_config.erosion_kernel_size = int(float(value))
        self.app.update_image_display()

    def _on_erosion_iterations_changed(self, value) -> None:
        """Handle erosion iterations change"""
        self.app.processing_config.erosion_iterations = int(float(value))
        self.app.update_image_display()

    def _on_morphology_changed(self) -> None:
        """Handle morphology change"""
        self.app.processing_config.morphology = self.morphology_var.get()
        self.app.update_image_display()

    def _on_morph_kernel_changed(self, value) -> None:
        """Handle morphology kernel change"""
        self.app.processing_config.morph_kernel_size = int(float(value))
        self.app.update_image_display()

    def _create_histogram_advanced_section(self) -> None:
        """Create advanced histogram section"""
        hist_frame = create_labeled_frame(self.scrollable_frame, "📊 Advanced Histogram")
        hist_frame.pack(fill=tk.X, pady=5, padx=5)

        clahe_tile_frame, _, _ = create_slider(
            hist_frame,
            "CLAHE Tile Size",
            self.clahe_tile_size_var,
            4,
            16,
            self._on_clahe_tile_size_changed,
        )
        clahe_tile_frame.pack(fill=tk.X, pady=2)

        create_checkbox(
            hist_frame,
            "Intensity Normalization",
            self.intensity_normalization_var,
            self._on_intensity_normalization_changed,
        ).pack(anchor=tk.W, pady=2)

        create_checkbox(
            hist_frame,
            "Contrast Stretching",
            self.contrast_stretching_var,
            self._on_contrast_stretching_changed,
        ).pack(anchor=tk.W, pady=2)

    def _on_clahe_tile_size_changed(self, value) -> None:
        """Handle CLAHE tile size change"""
        self.app.processing_config.clahe_tile_size = int(float(value))
        self.app.update_image_display()

    def _on_intensity_normalization_changed(self) -> None:
        """Handle intensity normalization change"""
        self.app.processing_config.intensity_normalization = self.intensity_normalization_var.get()
        self.app.update_image_display()

    def _on_contrast_stretching_changed(self) -> None:
        """Handle contrast stretching change"""
        self.app.processing_config.contrast_stretching = self.contrast_stretching_var.get()
        self.app.update_image_display()

    def _create_line_removal_advanced_section(self) -> None:
        """Create advanced line removal section"""
        line_frame = create_labeled_frame(self.scrollable_frame, "📏 Advanced Line Removal")
        line_frame.pack(fill=tk.X, pady=5, padx=5)

        vertical_kernel_frame, _, _ = create_slider(
            line_frame,
            "Vertical Kernel Size",
            self.vertical_kernel_size_var,
            1,
            15,
            self._on_vertical_kernel_size_changed,
        )
        vertical_kernel_frame.pack(fill=tk.X, pady=2)

        horizontal_kernel_frame, _, _ = create_slider(
            line_frame,
            "Horizontal Kernel Size",
            self.horizontal_kernel_size_var,
            1,
            15,
            self._on_horizontal_kernel_size_changed,
        )
        horizontal_kernel_frame.pack(fill=tk.X, pady=2)

        create_checkbox(
            line_frame,
            "Hough Lines Removal",
            self.hough_lines_removal_var,
            self._on_hough_lines_removal_changed,
        ).pack(anchor=tk.W, pady=2)

    def _on_vertical_kernel_size_changed(self, value) -> None:
        """Handle vertical kernel size change"""
        self.app.processing_config.vertical_kernel_size = int(float(value))
        self.app.update_image_display()

    def _on_horizontal_kernel_size_changed(self, value) -> None:
        """Handle horizontal kernel size change"""
        self.app.processing_config.horizontal_kernel_size = int(float(value))
        self.app.update_image_display()

    def _on_hough_lines_removal_changed(self) -> None:
        """Handle Hough lines removal change"""
        self.app.processing_config.hough_lines_removal = self.hough_lines_removal_var.get()
        self.app.update_image_display()

    def _create_advanced_morphology_section(self) -> None:
        """Create advanced morphology section"""
        morph_adv_frame = create_labeled_frame(self.scrollable_frame, "🔄 Advanced Morphology")
        morph_adv_frame.pack(fill=tk.X, pady=5, padx=5)

        create_checkbox(
            morph_adv_frame,
            "Stroke Width Normalization",
            self.stroke_width_normalization_var,
            self._on_stroke_width_normalization_changed,
        ).pack(anchor=tk.W, pady=2)

        create_checkbox(
            morph_adv_frame,
            "Tophat",
            self.tophat_var,
            self._on_tophat_changed,
        ).pack(anchor=tk.W, pady=2)

        create_checkbox(
            morph_adv_frame,
            "Blackhat",
            self.blackhat_var,
            self._on_blackhat_changed,
        ).pack(anchor=tk.W, pady=2)

        create_checkbox(
            morph_adv_frame,
            "Gradient",
            self.gradient_var,
            self._on_gradient_changed,
        ).pack(anchor=tk.W, pady=2)

    def _on_stroke_width_normalization_changed(self) -> None:
        """Handle stroke width normalization change"""
        self.app.processing_config.stroke_width_normalization = self.stroke_width_normalization_var.get()
        self.app.update_image_display()

    def _on_tophat_changed(self) -> None:
        """Handle tophat change"""
        self.app.processing_config.tophat = self.tophat_var.get()
        self.app.update_image_display()

    def _on_blackhat_changed(self) -> None:
        """Handle blackhat change"""
        self.app.processing_config.blackhat = self.blackhat_var.get()
        self.app.update_image_display()

    def _on_gradient_changed(self) -> None:
        """Handle gradient change"""
        self.app.processing_config.gradient = self.gradient_var.get()
        self.app.update_image_display()

    def _create_character_operations_section(self) -> None:
        """Create character operations section"""
        char_frame = create_labeled_frame(self.scrollable_frame, "🔤 Character Operations")
        char_frame.pack(fill=tk.X, pady=5, padx=5)

        create_checkbox(
            char_frame,
            "Character Dilation",
            self.character_dilation_var,
            self._on_character_dilation_changed,
        ).pack(anchor=tk.W, pady=2)

        create_checkbox(
            char_frame,
            "Character Erosion",
            self.character_erosion_var,
            self._on_character_erosion_changed,
        ).pack(anchor=tk.W, pady=2)

        min_contour_frame, _, _ = create_slider(
            char_frame,
            "Min Contour Area",
            self.min_contour_area_var,
            1,
            100,
            self._on_min_contour_area_changed,
        )
        min_contour_frame.pack(fill=tk.X, pady=2)

    def _on_character_dilation_changed(self) -> None:
        """Handle character dilation change"""
        self.app.processing_config.character_dilation = self.character_dilation_var.get()
        self.app.update_image_display()

    def _on_character_erosion_changed(self) -> None:
        """Handle character erosion change"""
        self.app.processing_config.character_erosion = self.character_erosion_var.get()
        self.app.update_image_display()

    def _on_min_contour_area_changed(self, value) -> None:
        """Handle min contour area change"""
        self.app.processing_config.min_contour_area = int(float(value))
        self.app.update_image_display()

    def _create_contour_filtering_section(self) -> None:
        """Create contour filtering section"""
        contour_frame = create_labeled_frame(self.scrollable_frame, "🎯 Contour Filtering")
        contour_frame.pack(fill=tk.X, pady=5, padx=5)

        create_checkbox(
            contour_frame,
            "Contour Filtering",
            self.contour_filtering_var,
            self._on_contour_filtering_changed,
        ).pack(anchor=tk.W, pady=2)

        create_checkbox(
            contour_frame,
            "Connected Components Filtering",
            self.connected_components_filtering_var,
            self._on_connected_components_filtering_changed,
        ).pack(anchor=tk.W, pady=2)

        create_checkbox(
            contour_frame,
            "Aspect Ratio Filtering",
            self.aspect_ratio_filtering_var,
            self._on_aspect_ratio_filtering_changed,
        ).pack(anchor=tk.W, pady=2)

    def _on_contour_filtering_changed(self) -> None:
        """Handle contour filtering change"""
        self.app.processing_config.contour_filtering = self.contour_filtering_var.get()
        self.app.update_image_display()

    def _on_connected_components_filtering_changed(self) -> None:
        """Handle connected components filtering change"""
        self.app.processing_config.connected_components_filtering = self.connected_components_filtering_var.get()
        self.app.update_image_display()

    def _on_aspect_ratio_filtering_changed(self) -> None:
        """Handle aspect ratio filtering change"""
        self.app.processing_config.aspect_ratio_filtering = self.aspect_ratio_filtering_var.get()
        self.app.update_image_display()

    def _on_lbp_radius_changed(self, value) -> None:
        """Handle LBP radius change"""
        self.app.processing_config.lbp_radius = int(float(value))
        self.app.update_image_display()

    def _create_advanced_operations_section(self) -> None:
        """Create consolidated advanced operations section"""
        advanced_frame = create_labeled_frame(self.scrollable_frame, "🧬 Advanced Operations")
        advanced_frame.pack(fill=tk.X, pady=5, padx=5)

        create_checkbox(
            advanced_frame,
            "Background Subtraction",
            self.background_subtraction_var,
            self._on_background_subtraction_changed,
        ).pack(anchor=tk.W, pady=1)

        bg_threshold_frame, _, _ = create_slider(
            advanced_frame,
            "BG Threshold",
            self.bg_threshold_var,
            0,
            100,
            self._on_bg_threshold_changed,
        )
        bg_threshold_frame.pack(fill=tk.X, pady=1)

        ttk.Separator(advanced_frame, orient="horizontal").pack(fill=tk.X, pady=3)

        self.contour_filtering_checkbox = create_checkbox(
            advanced_frame,
            "Contour Filtering",
            self.contour_filtering_var,
            self._on_contour_filtering_changed,
        )
        self.contour_filtering_checkbox.pack(anchor=tk.W, pady=1)

        contour_controls = ttk.Frame(advanced_frame)
        contour_controls.pack(fill=tk.X, pady=1)

        contour_area_min_frame, _, _ = create_slider(
            contour_controls,
            "Min Contour Area",
            self.contour_area_min_var,
            10,
            1000,
            self._on_contour_area_min_changed,
        )
        contour_area_min_frame.pack(fill=tk.X, pady=1)

        contour_area_max_frame, _, _ = create_slider(
            contour_controls,
            "Max Contour Area",
            self.contour_area_max_var,
            100,
            50000,
            self._on_contour_area_max_changed,
        )
        contour_area_max_frame.pack(fill=tk.X, pady=1)

        self.connected_components_filtering_checkbox = create_checkbox(
            advanced_frame,
            "Connected Components Filtering",
            self.connected_components_filtering_var,
            self._on_connected_components_filtering_changed,
        )
        self.connected_components_filtering_checkbox.pack(anchor=tk.W, pady=1)

        self.aspect_ratio_filtering_checkbox = create_checkbox(
            advanced_frame,
            "Aspect Ratio Filtering",
            self.aspect_ratio_filtering_var,
            self._on_aspect_ratio_filtering_changed,
        )
        self.aspect_ratio_filtering_checkbox.pack(anchor=tk.W, pady=1)

        ttk.Separator(advanced_frame, orient="horizontal").pack(fill=tk.X, pady=3)

        self.distance_transform_checkbox = create_checkbox(
            advanced_frame,
            "Distance Transform",
            self.distance_transform_var,
            self._on_distance_transform_changed,
        )
        self.distance_transform_checkbox.pack(anchor=tk.W, pady=1)

        distance_type_frame, _, _ = create_slider(
            advanced_frame,
            "Distance Transform Type",
            self.distance_transform_type_var,
            1,
            5,
            self._on_distance_transform_type_changed,
        )
        distance_type_frame.pack(fill=tk.X, pady=1)

        self.skeletonize_checkbox = create_checkbox(
            advanced_frame,
            "Skeletonize",
            self.skeletonize_var,
            self._on_skeletonize_changed,
        )
        self.skeletonize_checkbox.pack(anchor=tk.W, pady=1)

        self.watershed_markers_checkbox = create_checkbox(
            advanced_frame,
            "Watershed Markers",
            self.watershed_markers_var,
            self._on_watershed_markers_changed,
        )
        self.watershed_markers_checkbox.pack(anchor=tk.W, pady=1)

        ttk.Separator(advanced_frame, orient="horizontal").pack(fill=tk.X, pady=3)

        self.local_binary_pattern_checkbox = create_checkbox(
            advanced_frame,
            "Local Binary Pattern",
            self.local_binary_pattern_var,
            self._on_local_binary_pattern_changed,
        )
        self.local_binary_pattern_checkbox.pack(anchor=tk.W, pady=1)

        lbp_controls = ttk.Frame(advanced_frame)
        lbp_controls.pack(fill=tk.X, pady=1)

        lbp_radius_frame, _, _ = create_slider(
            lbp_controls,
            "LBP Radius",
            self.lbp_radius_var,
            1,
            8,
            self._on_lbp_radius_changed,
        )
        lbp_radius_frame.pack(fill=tk.X, pady=1)

        lbp_n_points_frame, _, _ = create_slider(
            lbp_controls,
            "LBP N Points",
            self.lbp_n_points_var,
            8,
            32,
            self._on_lbp_n_points_changed,
        )
        lbp_n_points_frame.pack(fill=tk.X, pady=1)

    def _on_contour_area_min_changed(self, value) -> None:
        """Handle contour area min change"""
        self.app.processing_config.contour_area_min = int(float(value))
        self.app.update_image_display()

    def _on_contour_area_max_changed(self, value) -> None:
        """Handle contour area max change"""
        self.app.processing_config.contour_area_max = int(float(value))
        self.app.update_image_display()

    def _on_distance_transform_type_changed(self, value) -> None:
        """Handle distance transform type change"""
        self.app.processing_config.distance_transform_type = int(float(value))
        self.app.update_image_display()

    def _on_lbp_n_points_changed(self, value) -> None:
        """Handle LBP N points change"""
        self.app.processing_config.lbp_n_points = int(float(value))
        self.app.update_image_display()

    def _on_distance_transform_changed(self) -> None:
        """Handle distance transform change"""
        self.app.processing_config.distance_transform = self.distance_transform_var.get()
        self.app.update_image_display()

    def _on_skeletonize_changed(self) -> None:
        """Handle skeletonize change"""
        self.app.processing_config.skeletonize = self.skeletonize_var.get()
        self.app.update_image_display()

    def _on_watershed_markers_changed(self) -> None:
        """Handle watershed markers change"""
        self.app.processing_config.watershed_markers = self.watershed_markers_var.get()
        self.app.update_image_display()

    def _on_edge_enhancement_changed(self) -> None:
        """Handle edge enhancement change"""
        self.app.processing_config.edge_enhancement = self.edge_enhancement_var.get()
        self.app.update_image_display()

    def _on_histogram_eq_changed(self) -> None:
        """Handle histogram equalization change"""
        self.app.processing_config.histogram_equalization = self.histogram_equalization_var.get()
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

    def _update_resize_dependent_controls(self) -> None:
        """Update state of controls that depend on resize"""
        is_resize = self.resize_enabled_var.get()
        state = tk.NORMAL if is_resize else tk.DISABLED

        self.resize_maintain_aspect_checkbox.config(state=state)

        if state == tk.DISABLED:
            self.app.processing_config.resize_maintain_aspect_ratio = False
        else:
            self.app.processing_config.resize_maintain_aspect_ratio = self.resize_maintain_aspect_var.get()

    def _is_image_grayscale(self) -> bool:
        """Check if processed image is grayscale"""
        if self.app.processed_image is None:
            return False

        return len(self.app.processed_image.shape) == 2

    def _is_image_binary(self) -> bool:
        """Check if processed image is binary (only 0 and 255 values)"""
        if self.app.processed_image is None:
            return False

        unique_values = np.unique(self.app.processed_image)
        return len(unique_values) == 2 and set(unique_values) == {0, 255}

    def _update_grayscale_dependent_controls(self) -> None:
        """Update state of controls that depend on grayscale"""
        is_grayscale_config = self.color_space_var.get() == "Grayscale"
        is_grayscale_image = self._is_image_grayscale()
        is_grayscale = is_grayscale_config or is_grayscale_image
        state = tk.NORMAL if is_grayscale else tk.DISABLED

        self.threshold_enabled_checkbox.config(state=state)
        self.clahe_checkbox.config(state=state)
        self.edge_enhancement_checkbox.config(state=state)
        self.histogram_equalization_checkbox.config(state=state)
        self.adaptive_hist_eq_checkbox.config(state=state)
        self.multi_otsu_checkbox.config(state=state)
        self.local_binary_pattern_checkbox.config(state=state)

        if state == tk.DISABLED:
            self.app.processing_config.threshold_enabled = False
            self.app.processing_config.edge_enhancement = False
            self.app.processing_config.histogram_equalization = False
            self.app.processing_config.adaptive_hist_eq = False
            self.app.processing_config.multi_otsu = False
            self.app.processing_config.local_binary_pattern = False
        else:
            self.app.processing_config.threshold_enabled = self.threshold_enabled_var.get()
            self.app.processing_config.edge_enhancement = self.edge_enhancement_var.get()
            self.app.processing_config.histogram_equalization = self.histogram_equalization_var.get()
            self.app.processing_config.adaptive_hist_eq = self.adaptive_hist_eq_var.get()
            self.app.processing_config.multi_otsu = self.multi_otsu_var.get()
            self.app.processing_config.local_binary_pattern = self.local_binary_pattern_var.get()

    def _update_binary_dependent_controls(self) -> None:
        """Update state of controls that depend on binary"""
        is_binary_config = self.threshold_enabled_var.get()
        is_binary_image = self._is_image_binary()
        is_binary = is_binary_config or is_binary_image
        state = tk.NORMAL if is_binary else tk.DISABLED

        self.noise_dots_removal_checkbox.config(state=state)
        self.contour_filtering_checkbox.config(state=state)
        self.connected_components_filtering_checkbox.config(state=state)
        self.aspect_ratio_filtering_checkbox.config(state=state)
        self.distance_transform_checkbox.config(state=state)
        self.skeletonize_checkbox.config(state=state)
        self.watershed_markers_checkbox.config(state=state)
        self.local_binary_pattern_checkbox.config(state=state)

        if state == tk.DISABLED:
            self.app.processing_config.noise_dots_removal = False
            self.app.processing_config.contour_filtering = False
            self.app.processing_config.connected_components_filtering = False
            self.app.processing_config.aspect_ratio_filtering = False
            self.app.processing_config.distance_transform = False
            self.app.processing_config.skeletonize = False
            self.app.processing_config.watershed_markers = False
            self.app.processing_config.local_binary_pattern = False
        else:
            self.app.processing_config.noise_dots_removal = self.noise_dots_removal_var.get()
            self.app.processing_config.contour_filtering = self.contour_filtering_var.get()
            self.app.processing_config.connected_components_filtering = self.connected_components_filtering_var.get()
            self.app.processing_config.aspect_ratio_filtering = self.aspect_ratio_filtering_var.get()
            self.app.processing_config.distance_transform = self.distance_transform_var.get()
            self.app.processing_config.skeletonize = self.skeletonize_var.get()
            self.app.processing_config.watershed_markers = self.watershed_markers_var.get()
            self.app.processing_config.local_binary_pattern = self.local_binary_pattern_var.get()
