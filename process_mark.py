"""process_final.py - Unified whole-slide preprocessing

Automatically detects and removes pen/stroke annotations of different colors
(red, blue, green, black/gray). Supports .ndpi and .svs formats.

Author: consolidated from multiple process_*.py scripts
"""

import argparse
import openslide
import cv2
from PIL import Image
import numpy as np
import os


def detect_pen_marks(hsv_image, verbose=True):
    """Auto-detect pen/stroke colors present in the image.

    Returns a list of detected colors.
    """
    detected_colors = []
    
    # ===== Detect red strokes =====
    # Red wraps around 0 degrees in HSV, so we detect it in two ranges.
    lower_red1 = np.array([0, 150, 30])
    upper_red1 = np.array([10, 255, 220])
    lower_red2 = np.array([170, 150, 30])
    upper_red2 = np.array([180, 255, 220])
    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    red_pixels = np.sum(red_mask > 0)
    if red_pixels > 1000:  # Threshold: >1000 red pixels
        detected_colors.append('red')
    
    # ===== Detect blue strokes =====
    lower_blue = np.array([100, 150, 80])
    upper_blue = np.array([130, 255, 220])
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    blue_pixels = np.sum(blue_mask > 0)
    if blue_pixels > 1000:
        detected_colors.append('blue')
    
    # ===== Detect green strokes =====
    lower_green = np.array([35, 60, 40])
    upper_green = np.array([95, 255, 220])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    green_pixels = np.sum(green_mask > 0)
    if green_pixels > 1000:
        detected_colors.append('green')
    
    # ===== Detect black/gray strokes =====
    # Blue-gray ink
    lower_blue_grey = np.array([100, 10, 30])
    upper_blue_grey = np.array([140, 80, 130])
    mask_blue_grey = cv2.inRange(hsv_image, lower_blue_grey, upper_blue_grey)
    
    # Neutral gray
    lower_neutral_grey = np.array([40, 0, 100])
    upper_neutral_grey = np.array([100, 30, 180])
    mask_neutral_grey = cv2.inRange(hsv_image, lower_neutral_grey, upper_neutral_grey)
    
    # Pure black
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 100, 60])
    mask_black = cv2.inRange(hsv_image, lower_black, upper_black)
    
    # Exclude purple/red tissue tones from H&E staining
    lower_purple_red1 = np.array([0, 20, 60])
    upper_purple_red1 = np.array([20, 255, 255])
    mask_tissue1 = cv2.inRange(hsv_image, lower_purple_red1, upper_purple_red1)
    
    lower_purple_red2 = np.array([150, 20, 60])
    upper_purple_red2 = np.array([180, 255, 255])
    mask_tissue2 = cv2.inRange(hsv_image, lower_purple_red2, upper_purple_red2)
    mask_tissue = cv2.bitwise_or(mask_tissue1, mask_tissue2)
    
    # Merge black/gray masks and exclude tissue mask
    black_combined = cv2.bitwise_or(mask_blue_grey, mask_neutral_grey)
    black_combined = cv2.bitwise_or(black_combined, mask_black)
    black_combined = cv2.bitwise_and(black_combined, cv2.bitwise_not(mask_tissue))
    
    # Connected components to detect elongated stroke-like regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(black_combined, connectivity=8)
    black_stroke_area = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
        # Elongated black regions are more likely to be pen strokes
        if (aspect_ratio > 2 and area > 80) or area > 500:
            black_stroke_area += area
    
    if black_stroke_area > 5000:  # Threshold: >5000 pixels of stroke-like black area
        detected_colors.append('black')
    
    if verbose:
        print(f"    Detected stroke colors: {detected_colors if detected_colors else 'none'}")
        print(
            f"    [red pixels: {red_pixels}, blue pixels: {blue_pixels}, green pixels: {green_pixels}, "
            f"black stroke area: {black_stroke_area}]"
        )
    
    return detected_colors


def remove_red_marks(detection_array, hsv, dilate_iterations=1, verbose=True):
    """Remove red pen/stroke annotations."""
    # Red strokes: detect only highly-saturated deep reds
    lower_red1 = np.array([0, 150, 30])
    upper_red1 = np.array([10, 255, 220])
    lower_red2 = np.array([170, 150, 30])
    upper_red2 = np.array([180, 255, 220])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask_combined = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Connected components: remove only large red regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(red_mask_combined, connectivity=8)
    
    red_mask = np.zeros_like(red_mask_combined)
    min_red_area = 500
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_red_area:
            red_mask[labels == i] = 255
    
    # Morphology
    kernel_pen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_pen, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel_pen, iterations=dilate_iterations)
    
    # Replace with white
    detection_array[red_mask > 0] = [255, 255, 255]
    
    if verbose:
        print(f"    ✓ Removed red strokes (dilate iterations: {dilate_iterations})")
    
    return detection_array


def remove_blue_marks(detection_array, hsv, dilate_iterations=3, verbose=True):
    """Remove blue pen/stroke annotations."""
    # Blue strokes: high saturation
    lower_blue = np.array([100, 150, 80])
    upper_blue = np.array([130, 255, 220])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Morphology
    kernel_pen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_pen, iterations=2)
    
    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(blue_mask, connectivity=8)
    
    blue_mask_final = np.zeros_like(blue_mask)
    min_blue_area = 200
    
    large_blue_count = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_blue_area:
            temp_mask = np.zeros_like(blue_mask)
            temp_mask[labels == i] = 255
            temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_DILATE, kernel_pen, iterations=dilate_iterations)
            blue_mask_final = cv2.bitwise_or(blue_mask_final, temp_mask)
            large_blue_count += 1
    
    # Replace with white
    detection_array[blue_mask_final > 0] = [255, 255, 255]
    
    if verbose:
        print(f"    ✓ Removed blue strokes ({large_blue_count} regions, dilate iterations: {dilate_iterations})")
    
    return detection_array


def remove_green_marks(detection_array, hsv, dilate_iterations=8, verbose=True):
    """Remove green pen/stroke annotations."""
    # Green strokes: includes deep green and near-black green
    lower_green = np.array([35, 60, 40])
    upper_green = np.array([95, 255, 220])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Morphology
    kernel_pen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel_pen, iterations=4)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_DILATE, kernel_pen, iterations=dilate_iterations)
    
    # Replace with white
    detection_array[green_mask > 0] = [255, 255, 255]
    
    if verbose:
        print(f"    ✓ Removed green strokes (dilate iterations: {dilate_iterations})")
    
    return detection_array


def remove_black_marks(detection_array, hsv, kernel_size=7, close_iters=8, dilate_iters=12, verbose=True):
    """Remove black/gray pen/stroke annotations."""
    # Blue-gray ink
    lower_blue_grey = np.array([100, 10, 30])
    upper_blue_grey = np.array([140, 80, 130])
    mask_blue_grey = cv2.inRange(hsv, lower_blue_grey, upper_blue_grey)
    
    # Neutral gray
    lower_neutral_grey = np.array([40, 0, 100])
    upper_neutral_grey = np.array([100, 30, 180])
    mask_neutral_grey = cv2.inRange(hsv, lower_neutral_grey, upper_neutral_grey)
    
    # Pure black
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 100, 60])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    
    # Exclude purple/red tissue tones from H&E staining
    lower_purple_red1 = np.array([0, 20, 60])
    upper_purple_red1 = np.array([20, 255, 255])
    mask_tissue1 = cv2.inRange(hsv, lower_purple_red1, upper_purple_red1)
    
    lower_purple_red2 = np.array([150, 20, 60])
    upper_purple_red2 = np.array([180, 255, 255])
    mask_tissue2 = cv2.inRange(hsv, lower_purple_red2, upper_purple_red2)
    mask_tissue = cv2.bitwise_or(mask_tissue1, mask_tissue2)
    
    # Merge and exclude tissue
    mask_combined = cv2.bitwise_or(mask_blue_grey, mask_neutral_grey)
    mask_combined = cv2.bitwise_or(mask_combined, mask_black)
    mask_combined = cv2.bitwise_and(mask_combined, cv2.bitwise_not(mask_tissue))
    
    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_combined, connectivity=8)
    
    black_mask = np.zeros_like(mask_combined)
    stroke_count = 0
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
        
        if (aspect_ratio > 2 and area > 80) or area > 500:
            black_mask[labels == i] = 255
            stroke_count += 1
    
    # Morphology
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=close_iters)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_DILATE, kernel_medium, iterations=dilate_iters)
    
    # Replace with white
    detection_array[black_mask > 0] = [255, 255, 255]
    
    if verbose:
        print(f"    ✓ Removed black/gray strokes ({stroke_count} regions)")
    
    return detection_array


def process_single_slide(
    slide_path,
    file_prefix,
    min_area,
    min_hole_area,
    *,
    out_dir=".",
    overwrite=False,
    auto_detect=True,
    force_colors=None,
    red_dilate_iterations=1,
    blue_dilate_iterations=3,
    green_dilate_iterations=8,
    black_kernel_size=7,
    black_close_iterations=8,
    black_dilate_iterations=12,
    verbose=True,
):
    """
    Process a single whole-slide image.

    Args:
        slide_path: Path to the slide file
        file_prefix: Output file prefix
        min_area: Minimum tissue region area
        min_hole_area: Minimum hole area
        auto_detect: Whether to auto-detect pen/stroke colors
        force_colors: Colors to force-process (e.g. ['red', 'blue'])
        verbose: Whether to print detailed logs
    """
    slide = openslide.OpenSlide(slide_path)
    
    # Level-0 dimensions
    level_0_dimensions = slide.level_dimensions[0]
    
    # Physical resolution (MPP)
    mpp_x = slide.properties.get('openslide.mpp-x')
    mpp_y = slide.properties.get('openslide.mpp-y')
    if mpp_x and mpp_y:
        avg_mpp = (float(mpp_x) + float(mpp_y)) / 2
        magnification = "40x" if avg_mpp < 0.3 else "20x" if avg_mpp < 0.6 else "Unknown"
    else:
        magnification = "Unknown"
    
    # Create output folders under out_dir
    origin_dir = os.path.join(out_dir, "origin_pic")
    circle_dir = os.path.join(out_dir, "circle_pic")
    mask_dir = os.path.join(out_dir, "mask_pic")
    os.makedirs(origin_dir, exist_ok=True)
    os.makedirs(circle_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # Default: do not overwrite; skip if main outputs exist
    circle_out_path = os.path.join(circle_dir, f"{file_prefix}_tissue_contour_800x800.png")
    origin_out_path = os.path.join(origin_dir, f"{file_prefix}_thumbnail_800x800.png")
    mask_out_path = os.path.join(mask_dir, f"{file_prefix}_binary_mask_800x800.png")

    if (not overwrite) and os.path.exists(circle_out_path) and os.path.exists(mask_out_path) and os.path.exists(origin_out_path):
        if verbose:
            print(f"  - Outputs already exist; skipping (use -overwrite to replace): {file_prefix}")
        slide.close()
        return {
            'outer_contours': None,
            'hole_contours': None,
            'detected_colors': [],
            'skipped': True,
        }
    
    # Choose detection level
    detection_level = min(3, slide.level_count - 1)
    level_dimensions = slide.level_dimensions[detection_level]
    
    # Read image
    detection_img = slide.read_region((0, 0), detection_level, level_dimensions)
    detection_img = detection_img.convert('RGB')
    detection_array = np.array(detection_img)
    
    # Keep original thumbnail for origin_pic
    detection_array_original = detection_array.copy()
    
    # ===== Auto-detect and remove pen/stroke annotations =====
    hsv = cv2.cvtColor(detection_array, cv2.COLOR_RGB2HSV)
    
    if auto_detect:
        detected_colors = detect_pen_marks(hsv, verbose=verbose)
        if force_colors:
            detected_colors = sorted(set(detected_colors).union(set(force_colors)))
    else:
        detected_colors = force_colors if force_colors else []
    
    # Process colors in order
    if 'red' in detected_colors:
        detection_array = remove_red_marks(
            detection_array, hsv, dilate_iterations=red_dilate_iterations, verbose=verbose
        )
        hsv = cv2.cvtColor(detection_array, cv2.COLOR_RGB2HSV)  # recompute HSV
    
    if 'green' in detected_colors:
        detection_array = remove_green_marks(
            detection_array, hsv, dilate_iterations=green_dilate_iterations, verbose=verbose
        )
        hsv = cv2.cvtColor(detection_array, cv2.COLOR_RGB2HSV)
    
    if 'blue' in detected_colors:
        detection_array = remove_blue_marks(
            detection_array, hsv, dilate_iterations=blue_dilate_iterations, verbose=verbose
        )
        hsv = cv2.cvtColor(detection_array, cv2.COLOR_RGB2HSV)
    
    if 'black' in detected_colors:
        detection_array = remove_black_marks(
            detection_array,
            hsv,
            kernel_size=black_kernel_size,
            close_iters=black_close_iterations,
            dilate_iters=black_dilate_iterations,
            verbose=verbose,
        )
    
    # ===== Tissue detection =====
    # Convert to grayscale
    gray = cv2.cvtColor(detection_array, cv2.COLOR_RGB2GRAY)
    
    # Otsu thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary_opened = cv2.morphologyEx(binary_closed, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours with hierarchy
    contours, hierarchy = cv2.findContours(binary_opened, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Adjust area thresholds by downsample factor
    downsample_factor = slide.level_downsamples[detection_level]
    adjusted_min_area = min_area / (downsample_factor ** 2)
    adjusted_min_hole_area = min_hole_area / (downsample_factor ** 2)
    
    outer_contours = []
    hole_contours = []
    
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if hierarchy[i][3] == -1:  # outer contour
                if area > adjusted_min_area:
                    outer_contours.append(cnt)
            else:  # inner holes
                if area > adjusted_min_hole_area:
                    hole_contours.append(cnt)
        
        # Keep at most 10 largest holes
        if len(hole_contours) > 10:
            hole_contours = sorted(hole_contours, key=cv2.contourArea, reverse=True)[:10]
            if verbose:
                print("    Limiting to the 10 largest holes")
        
        # Filter tissue fragments smaller than 1/5 of the largest tissue region
        if len(outer_contours) > 0:
            areas = [cv2.contourArea(cnt) for cnt in outer_contours]
            max_area_val = max(areas)
            min_fragment_area = max_area_val / 5
            filtered_contours = [cnt for cnt in outer_contours if cv2.contourArea(cnt) >= min_fragment_area]
            if len(filtered_contours) < len(outer_contours) and verbose:
                print(
                    f"    Filtered out {len(outer_contours) - len(filtered_contours)} small fragments "
                    "(< 1/5 of the largest tissue area)"
                )
            outer_contours = filtered_contours
    
    # ===== Draw and save results =====
    # Draw contours on the detection-resolution image
    result_detection = detection_array.copy()
    result_detection_bgr = cv2.cvtColor(result_detection, cv2.COLOR_RGB2BGR)
    
    cv2.drawContours(result_detection_bgr, outer_contours, -1, (0, 255, 0), 3)
    cv2.drawContours(result_detection_bgr, hole_contours, -1, (255, 0, 0), 3)
    
    result_detection_rgb = cv2.cvtColor(result_detection_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize to 800x800
    h, w = result_detection_rgb.shape[:2]
    scale = min(800 / w, 800 / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    result_resized = cv2.resize(result_detection_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create 800x800 light background
    result_final = np.ones((800, 800, 3), dtype=np.uint8) * 240
    
    # Center placement
    y_offset = (800 - new_h) // 2
    x_offset = (800 - new_w) // 2
    result_final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = result_resized
    
    # Save contour overlay
    result_pil = Image.fromarray(result_final)
    result_pil.save(circle_out_path)

    # Save original thumbnail
    thumbnail_resized = cv2.resize(detection_array_original, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    thumbnail_final = np.ones((800, 800, 3), dtype=np.uint8) * 240
    thumbnail_final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = thumbnail_resized
    thumbnail_pil = Image.fromarray(thumbnail_final)
    thumbnail_pil.save(origin_out_path)
    
    # Save tissue mask
    mask_with_holes = np.zeros_like(binary_opened)
    cv2.drawContours(mask_with_holes, outer_contours, -1, 255, -1)
    cv2.drawContours(mask_with_holes, hole_contours, -1, 0, -1)
    binary_pil = Image.fromarray(mask_with_holes)
    binary_pil.save(mask_out_path)
    
    # Print summary
    if verbose:
        print(f"\n  Level-0 dimensions: {level_0_dimensions}")
        print(f"  Detection level {detection_level} dimensions: {level_dimensions}")
        print(f"  Downsample factor: {downsample_factor:.2f}")
        print(f"  Estimated magnification: {magnification}")
        print(f"  Detected {len(outer_contours)} tissue regions (green)")
        print(f"  Detected {len(hole_contours)} holes (blue)")
    
    slide.close()
    
    return {
        'outer_contours': len(outer_contours),
        'hole_contours': len(hole_contours),
        'detected_colors': detected_colors,
        'skipped': False,
    }


def process_files(
    input_folder,
    min_area,
    min_hole_area,
    file_type="ndpi",
    *,
    out_dir=".",
    overwrite=False,
    auto_detect=True,
    force_colors=None,
    red_dilate_iterations=1,
    blue_dilate_iterations=3,
    green_dilate_iterations=8,
    black_kernel_size=7,
    black_close_iterations=8,
    black_dilate_iterations=12,
    verbose=True,
):
    """
    Batch-process whole-slide images in a folder.

    Args:
        input_folder: Input folder path
        min_area: Minimum tissue region area
        min_hole_area: Minimum hole area
        file_type: File type ('ndpi' or 'svs')
        auto_detect: Whether to auto-detect pen/stroke colors
        force_colors: Colors to force-process
        verbose: Whether to print detailed logs
    """
    file_ext = f".{file_type}"
    files = [f for f in os.listdir(input_folder) if f.endswith(file_ext)]
    
    if len(files) == 0:
        print(f"Warning: no {file_ext} files found in {input_folder}")
        return

    if verbose:
        print(f"\n{'='*60}")
        print("process_final.py - Unified WSI preprocessing")
        print(f"{'='*60}")
        print(f"Input folder: {input_folder}")
        print(f"File type: {file_type}")
        print(f"Found {len(files)} files")
        print(f"Auto-detect strokes: {'yes' if auto_detect else 'no'}")
        if force_colors:
            print(f"Force-process colors: {force_colors}")
        print(
            "Params: "
            f"red_dilate={red_dilate_iterations}, "
            f"blue_dilate={blue_dilate_iterations}, "
            f"green_dilate={green_dilate_iterations}, "
            f"black_kernel={black_kernel_size}, "
            f"black_close={black_close_iterations}, "
            f"black_dilate={black_dilate_iterations}"
        )
        print(f"Output dir: {os.path.abspath(out_dir)}")
        print(f"Overwrite existing results: {'yes' if overwrite else 'no (skip existing outputs by default)'}")
        print(f"{'='*60}\n")
    
    for idx, file in enumerate(files, start=1):
        slide_path = os.path.join(input_folder, file)
        file_prefix = os.path.splitext(file)[0]
        
        if verbose:
            print(f"\n[{idx}/{len(files)}] Processing: {file}")
            print("-" * 40)
        
        try:
            process_single_slide(
                slide_path=slide_path,
                file_prefix=file_prefix,
                min_area=min_area,
                min_hole_area=min_hole_area,
                out_dir=out_dir,
                overwrite=overwrite,
                auto_detect=auto_detect,
                force_colors=force_colors,
                red_dilate_iterations=red_dilate_iterations,
                blue_dilate_iterations=blue_dilate_iterations,
                green_dilate_iterations=green_dilate_iterations,
                black_kernel_size=black_kernel_size,
                black_close_iterations=black_close_iterations,
                black_dilate_iterations=black_dilate_iterations,
                verbose=verbose
            )
            if verbose:
                print("  ✓ Done")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

        if verbose:
            print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified whole-slide preprocessing - auto-detect and remove pen/stroke annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-pic_path", required=True, 
                        help="Folder path containing slide files")
    parser.add_argument("-out_dir", default=".",
                        help="Output root directory (creates origin_pic/circle_pic/mask_pic under it; default: current dir)")
    parser.add_argument("-overwrite", action='store_true',
                        help="Overwrite existing output files (default: skip the slide if outputs exist)")
    parser.add_argument("-min_area", type=float, default=1000, 
                        help="Minimum tissue region area (default: 1000)")
    parser.add_argument("-min_hole", type=float, default=60000, 
                        help="Minimum hole area (default: 60000)")
    parser.add_argument("-file_type", choices=["ndpi", "svs"], default="ndpi", 
                        help="File type: ndpi or svs (default: ndpi)")
    parser.add_argument("-colors", nargs='+', choices=['red', 'blue', 'green', 'black'],
                        help="Force-process specified stroke colors (can provide multiple)")
    parser.add_argument("-no_auto_detect", action='store_true',
                        help="Disable auto-detection; only process colors provided via -colors")
    parser.add_argument("-quiet", action='store_true',
                        help="Reduce console output")

    # Common tunable parameters (kept for compatibility)
    parser.add_argument("-red_dilate_iterations", type=int, default=1,
                        help="Red stroke dilate iterations (default: 1)")
    parser.add_argument("-blue_dilate_iterations", type=int, default=3,
                        help="Blue stroke dilate iterations (default: 3)")
    parser.add_argument("-green_dilate_iterations", type=int, default=8,
                        help="Green stroke dilate iterations (default: 8)")

    # Black/gray stroke parameters (recommended: black_*; kernel/close/dilate are legacy names)
    parser.add_argument("-kernel_size", type=int, default=7,
                        help="[legacy] Black/gray kernel size for close/dilate (default: 7); prefer -black_kernel_size")
    parser.add_argument("-close_iterations", type=int, default=8,
                        help="[legacy] Black/gray close iterations (default: 8); prefer -black_close_iterations")
    parser.add_argument("-dilate_iterations", type=int, default=12,
                        help="[legacy] Black/gray dilate iterations (default: 12); prefer -black_dilate_iterations")

    # Intuitive black_* aliases (equivalent to the legacy three; black_* takes precedence)
    parser.add_argument("-black_kernel_size", type=int, default=None,
                        help="Black/gray strokes: kernel size for close/dilate (higher priority than -kernel_size)")
    parser.add_argument("-black_close_iterations", type=int, default=None,
                        help="Black/gray strokes: close iterations (higher priority than -close_iterations)")
    parser.add_argument("-black_dilate_iterations", type=int, default=None,
                        help="Black/gray strokes: dilate iterations (higher priority than -dilate_iterations)")
    
    args = parser.parse_args()

    # Unify black parameters: allow -black_* to override legacy flag names
    black_kernel_size = args.black_kernel_size if args.black_kernel_size is not None else args.kernel_size
    black_close_iterations = (
        args.black_close_iterations if args.black_close_iterations is not None else args.close_iterations
    )
    black_dilate_iterations = (
        args.black_dilate_iterations if args.black_dilate_iterations is not None else args.dilate_iterations
    )
    
    # Determine auto-detection mode
    auto_detect = not args.no_auto_detect
    force_colors = args.colors

    # If user explicitly provides any black_* param, treat it as "must process black".
    if (args.black_kernel_size is not None) or (args.black_close_iterations is not None) or (args.black_dilate_iterations is not None):
        if force_colors is None:
            force_colors = ['black']
        elif 'black' not in force_colors:
            force_colors = list(force_colors) + ['black']
    
    # If -colors is provided while auto-detect is on, behavior is union(auto_detect, forced_colors)
    if force_colors and auto_detect and (not args.quiet):
        print("Note: -colors is provided; it will be merged with auto-detection results")
    
    process_files(
        input_folder=args.pic_path,
        min_area=args.min_area,
        min_hole_area=args.min_hole,
        file_type=args.file_type,
        out_dir=args.out_dir,
        overwrite=args.overwrite,
        auto_detect=auto_detect,
        force_colors=force_colors,
        red_dilate_iterations=args.red_dilate_iterations,
        blue_dilate_iterations=args.blue_dilate_iterations,
        green_dilate_iterations=args.green_dilate_iterations,
        black_kernel_size=black_kernel_size,
        black_close_iterations=black_close_iterations,
        black_dilate_iterations=black_dilate_iterations,
        verbose=not args.quiet
    )
