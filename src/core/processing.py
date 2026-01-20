import cv2
import numpy as np
from skimage.filters import threshold_multiotsu

from src.config import ProcessingConfig
from src.infra.cache import image_cache


@image_cache(max_size=128)
def process_image(image: np.ndarray, config: ProcessingConfig | None = None) -> np.ndarray:
    if config is None:
        config = ProcessingConfig()

    image_processed = image.copy()

    processors = [
        _apply_resize,
        _apply_crop,
        _apply_color_space,
        _apply_gamma_correction,
        _apply_denoising,
        _apply_filters,
        _apply_histogram_operations,
        _apply_line_removal,
        _apply_morphological_operations,
        _apply_character_operations,
        _apply_enhancement_operations,
        _apply_threshold,
        _apply_advanced_morphology,
        _apply_contour_filtering,
        _apply_advanced_operations,
    ]

    for processor in processors:
        image_processed = processor(image_processed, config)

    return image_processed


def _ensure_binary(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if len(np.unique(image)) > 2:
        raise ValueError("Image is not binary")

    return image


def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


def _apply_crop(image: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    if not config.crop_enabled or config.bbox is None:
        return image

    h, w = image.shape[:2]
    x1, y1, x2, y2 = config.bbox

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 > x1 and y2 > y1:
        return image[y1:y2, x1:x2]

    return image


def _apply_resize(image: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    if not config.resize_enabled:
        return image

    h, w = image.shape[:2]

    if config.resize_maintain_aspect_ratio:
        aspect = w / h

        if aspect > 1:
            new_w = config.resize_width
            new_h = int(config.resize_width / aspect)
        else:
            new_h = config.resize_height
            new_w = int(config.resize_height * aspect)
    else:
        new_w, new_h = config.resize_width, config.resize_height

    if (w, h) != (new_w, new_h):
        is_upscaling = w < new_w or h < new_h
        interpolation = cv2.INTER_LANCZOS4 if is_upscaling else cv2.INTER_AREA

        image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    return image


def _apply_color_space(image: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    match config.color_space:
        case "Grayscale":
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image
        case "RGB":
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        case "BGR":
            return image  # Already in BGR format
        case "HSV":
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            return image
        case "LAB":
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            return image
        case "YUV":
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            return image
        case "YCrCb":
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            return image
        case _:
            return image


def _apply_gamma_correction(image: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    if not config.gamma_correction or config.gamma_value == 1.0:
        return image

    inv_gamma = 1.0 / config.gamma_value
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def _apply_denoising(image: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    if config.denoise_nl_means:
        image = cv2.fastNlMeansDenoising(
            image,
            None,
            config.denoise_h,
            config.denoise_template_window,
            config.denoise_search_window,
        )

    if config.edge_preserving_filter:
        image = cv2.edgePreservingFilter(
            image,
            flags=config.edge_filter_flags,
            sigma_s=config.edge_sigma_s,
            sigma_r=config.edge_sigma_r,
        )

    if config.noise_reduction_bilateral:
        for _ in range(config.bilateral_iterations):
            image = cv2.bilateralFilter(image, 5, 80, 80)

    return image


def _apply_filters(image: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    if config.bilateral_filter:
        image = cv2.bilateralFilter(
            image,
            config.bilateral_d,
            config.bilateral_sigma_color,
            config.bilateral_sigma_space,
        )

    if config.gaussian_blur:
        kernel_size = config.gaussian_kernel * 2 + 1
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), config.gaussian_sigma)

    if config.median_filter:
        image = cv2.medianBlur(image, config.median_kernel * 2 + 1)

    if config.background_subtraction:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        image = cv2.subtract(image, background)
        image = cv2.add(image, np.full_like(image, config.bg_threshold))

    return image


def _apply_histogram_operations(image: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    if config.histogram_equalization and len(image.shape) == 2:
        image = cv2.equalizeHist(image)

    if config.clahe and len(image.shape) == 2:
        clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=(config.clahe_tile_size, config.clahe_tile_size),
        )
        image = clahe.apply(image)
    elif config.adaptive_hist_eq and len(image.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(config.adaptive_hist_kernel, config.adaptive_hist_kernel))
        image = clahe.apply(image)

    if config.multi_otsu and len(image.shape) == 2:
        thresholds = threshold_multiotsu(image, classes=config.multi_otsu_classes)
        regions = np.digitize(image, bins=thresholds)
        image = (regions * (255 // (config.multi_otsu_classes - 1))).astype(np.uint8)

    return image


def _apply_line_removal(image: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    if config.vertical_line_removal:
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, config.vertical_kernel_size))
        vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)
        image = cv2.subtract(image, vertical_lines)

    if config.horizontal_line_removal:
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config.horizontal_kernel_size, 1))
        horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel)
        image = cv2.subtract(image, horizontal_lines)

    return image


def _apply_morphological_operations(image: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    if config.stroke_width_normalization:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

        for _ in range(config.stroke_iterations):
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    if config.morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.morph_kernel_size, config.morph_kernel_size))

        if config.morph_open:
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        if config.morph_close:
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return image


def _apply_character_operations(image: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    if config.character_separation:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.char_sep_kernel_size, config.char_sep_kernel_size),
        )

        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    if config.character_dilation:
        dil_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.dilation_kernel_size, config.dilation_kernel_size),
        )

        for _ in range(config.dilation_iterations):
            image = cv2.dilate(image, dil_kernel, iterations=1)

    if config.character_erosion:
        ero_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.erosion_kernel_size, config.erosion_kernel_size),
        )

        for _ in range(config.erosion_iterations):
            image = cv2.erode(image, ero_kernel, iterations=1)

    if config.noise_dots_removal:
        binary_image = _ensure_binary(image)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < config.min_contour_area:
                cv2.drawContours(image, [contour], -1, (0,), -1)

    return image


def _apply_enhancement_operations(image: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    if config.text_enhancement:
        kernel_size = config.text_kernel_size * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    if config.detail_enhancement:
        if len(image.shape) == 2:
            color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            color_image = image.copy()

        enhanced = cv2.detailEnhance(
            color_image,
            sigma_s=config.detail_sigma_s,
            sigma_r=config.detail_sigma_r,
        )

        if len(image.shape) == 2:
            image = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        else:
            image = enhanced

    if config.edge_enhancement:
        gray_image = _ensure_grayscale(image)
        edges = cv2.Canny(gray_image, 50, 150)

        if len(image.shape) == 3:
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        image = cv2.addWeighted(image, 1.0, edges, config.edge_strength, 0)

    if config.unsharp_mask:
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        image = cv2.addWeighted(image, config.unsharp_strength, gaussian, 1 - config.unsharp_strength, 0)

    if config.sharpen:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        image = cv2.addWeighted(image, 1 - config.sharpen_strength, sharpened, config.sharpen_strength, 0)

    return image


def _apply_threshold(image: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    if not config.threshold_enabled:
        return image

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    match config.threshold_type:
        case "BINARY":
            _, image = cv2.threshold(image, config.threshold_value, 255, cv2.THRESH_BINARY)

        case "BINARY_INV":
            _, image = cv2.threshold(image, config.threshold_value, 255, cv2.THRESH_BINARY_INV)

        case "OTSU_BINARY":
            if image.size > 0 and len(np.unique(image)) > 1:
                _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, image = cv2.threshold(image, config.threshold_value, 255, cv2.THRESH_BINARY)

        case "ADAPTIVE_MEAN":
            if config.adaptive_block_size % 2 == 1:
                block_size = config.adaptive_block_size
            else:
                block_size = config.adaptive_block_size + 1

            image = cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                block_size,
                config.adaptive_c,
            )

        case "ADAPTIVE_GAUSSIAN":
            if config.adaptive_block_size % 2 == 1:
                block_size = config.adaptive_block_size
            else:
                block_size = config.adaptive_block_size + 1

            image = cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                config.adaptive_c,
            )

    return image


def _apply_advanced_morphology(image: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    if config.tophat:
        tophat_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.tophat_kernel_size, config.tophat_kernel_size),
        )
        image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, tophat_kernel)

    if config.blackhat:
        blackhat_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.blackhat_kernel_size, config.blackhat_kernel_size),
        )
        image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, blackhat_kernel)

    if config.gradient:
        gradient_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.gradient_kernel_size, config.gradient_kernel_size),
        )
        image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, gradient_kernel)
    elif config.morphological_gradient:
        morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.morphological_gradient_kernel, config.morphological_gradient_kernel),
        )
        image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, morph_kernel)

    return image


def _apply_contour_filtering(image: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    if config.contour_filtering:
        binary_image = _ensure_binary(image)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(image)

        for contour in contours:
            area = cv2.contourArea(contour)

            if config.contour_area_min <= area <= config.contour_area_max:
                cv2.drawContours(mask, [contour], -1, (255,), -1)

        image = cv2.bitwise_and(image, mask)

    if config.connected_components_filtering:
        binary_image = _ensure_binary(image)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        mask = np.zeros_like(image)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            if config.cc_min_area <= area <= config.cc_max_area:
                mask[labels == i] = 255

        image = mask

    if config.aspect_ratio_filtering:
        binary_image = _ensure_binary(image)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(image)

        for contour in contours:
            _, _, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0

            if config.min_aspect_ratio <= aspect_ratio <= config.max_aspect_ratio:
                cv2.drawContours(mask, [contour], -1, (255,), -1)

        image = cv2.bitwise_and(image, mask)

    return image


def _apply_advanced_operations(image: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    if config.hough_lines_removal:
        lines = cv2.HoughLinesP(
            image,
            1,
            np.pi / 180,
            config.hough_threshold,
            minLineLength=config.hough_min_line_length,
            maxLineGap=config.hough_max_line_gap,
        )

        if lines is not None:
            for line in lines:
                coords = line.flatten()
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                cv2.line(image, (x1, y1), (x2, y2), (0,), 2)

    if config.intensity_normalization:
        image = cv2.normalize(image, image, config.norm_min, config.norm_max, cv2.NORM_MINMAX)

    if config.contrast_stretching:
        p_min = float(np.percentile(image.astype(np.float64), config.stretch_min_percentile))
        p_max = float(np.percentile(image.astype(np.float64), config.stretch_max_percentile))
        image = np.clip((image - p_min) * 255 / (p_max - p_min), 0, 255).astype(np.uint8)

    if config.distance_transform:
        binary_image = _ensure_binary(image)
        image = cv2.distanceTransform(binary_image, config.distance_transform_type, 3)
        image = cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if config.skeletonize:
        binary_image = _ensure_binary(image)
        image = _apply_skeletonization(binary_image)

    if config.watershed_markers:
        binary_image = _ensure_binary(image)
        image = _apply_watershed_markers(binary_image)

    if config.local_binary_pattern:
        from skimage.feature import local_binary_pattern

        gray_image = _ensure_grayscale(image)
        lbp = local_binary_pattern(gray_image, config.lbp_n_points, config.lbp_radius, method="uniform")

        if lbp.max() > 0:
            image = (lbp * (255.0 / lbp.max())).astype(np.uint8)
        else:
            image = lbp.astype(np.uint8)

    return image


def _apply_skeletonization(image: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skeleton = np.zeros_like(image)
    working_image = image.copy()

    while True:
        eroded = cv2.erode(working_image, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(working_image, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        working_image = eroded.copy()

        if cv2.countNonZero(working_image) == 0:
            break

    return skeleton


def _apply_watershed_markers(image: np.ndarray) -> np.ndarray:
    binary_image = _ensure_binary(image)
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary_image, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    return cv2.subtract(sure_bg, sure_fg)
