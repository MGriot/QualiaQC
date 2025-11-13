# vector_tools.py
import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Set

import numpy as np
import cv2
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

# -------------------------
# Data Classes
# -------------------------

@dataclass
class PageSegmentation:
    label_map: np.ndarray  # int32 [H, W] region labels (0..N); 0 means no-region (edge)
    num_labels: int
    border_labels: Set[int]  # labels touching borders (usually outside "background" regions)

# -------------------------
# Utility functions
# -------------------------

def otsu_threshold(gray_u8: np.ndarray) -> int:
    """Compute Otsu threshold for a uint8 grayscale image. Returns integer threshold [0..255]."""
    hist, bin_edges = np.histogram(gray_u8.flatten(), bins=256, range=(0, 255))
    total = gray_u8.size
    sum_total = np.dot(np.arange(256), hist)

    sumB = 0.0
    wB = 0.0
    var_max = -1.0
    threshold = 127  # fallback

    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        # Between class variance
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > var_max:
            var_max = var_between
            threshold = t
    return threshold

def to_uint8_grayscale(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL image to uint8 grayscale numpy array [H, W]."""
    gray = pil_img.convert("L")
    return np.array(gray, dtype=np.uint8)

def numpy_mask_from_label(label_map: np.ndarray, label_id: int) -> np.ndarray:
    """Return boolean mask where label_map == label_id."""
    return label_map == label_id

def pil_mask_from_bool(mask: np.ndarray) -> Image.Image:
    """Convert boolean mask to single-channel 'L' PIL image (0/255)."""
    return Image.fromarray((mask.astype(np.uint8)) * 255, mode="L")

def safe_make_dir(path: str):
    os.makedirs(path, exist_ok=True)

# -------------------------
# New Vectorization Algorithm
# -------------------------

def _skeletonize(img_u8: np.ndarray) -> np.ndarray:
    """
    Performs morphological skeletonization on a binary image.
    Assumes input is a single-channel uint8 image (0 or 255).
    """
    img = img_u8.copy() // 255
    skeleton = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
            
    return skeleton * 255

def _get_line_intersection(p1, p2, p3, p4):
    """
    Calculates the intersection point of two line segments.
    Returns the intersection point (x, y) or None if they don't intersect within the segments.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return None  # Parallel or collinear

    t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    u_num = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))

    t = t_num / den
    u = u_num / den

    # Check if intersection is strictly within both line segments (not at endpoints)
    if 0.01 < t < 0.99 and 0.01 < u < 0.99:
        px = int(x1 + t * (x2 - x1))
        py = int(y1 + t * (y2 - y1))
        return (px, py)
        
    return None

def _post_process_vectors(segments: list) -> list:
    """Deduplicates and splits segments at intersections."""
    # 1. Deduplicate exact-match segments
    canonical_segments = set()
    for seg in segments:
        p1 = tuple(seg[0][0])
        p2 = tuple(seg[1][0])
        canonical_segments.add(tuple(sorted((p1, p2))))
    
    unique_segments = [[np.array(p, dtype=np.int32) for p in sorted(seg)] for seg in canonical_segments]

    # 2. Find all intersection points
    split_points_map = {i: [list(seg[0]), list(seg[1])] for i, seg in enumerate(unique_segments)}
    
    for i in range(len(unique_segments)):
        for j in range(i + 1, len(unique_segments)):
            p1, p2 = unique_segments[i]
            p3, p4 = unique_segments[j]
            
            intersect_pt = _get_line_intersection(p1, p2, p3, p4)
            
            if intersect_pt:
                split_points_map[i].append(list(intersect_pt))
                split_points_map[j].append(list(intersect_pt))

    # 3. Rebuild segments from the split points
    new_segments = []
    for i, points in split_points_map.items():
        if len(points) > 2: # If there were any intersections
            p1 = points[0]
            # Sort points by distance from the first endpoint to create segments in order
            points.sort(key=lambda p: (p[0] - p1[0])**2 + (p[1] - p1[1])**2)
            
            for k in range(len(points) - 1):
                pt1 = points[k]
                pt2 = points[k+1]
                # Avoid creating zero-length segments
                if pt1 != pt2:
                    new_segments.append(np.array([[pt1], [pt2]], dtype=np.int32))
        else: # No intersections for this segment
            p1, p2 = unique_segments[i]
            new_segments.append(np.array([[p1], [p2]], dtype=np.int32))

    # Final deduplication pass
    final_canonical_segments = set()
    for seg in new_segments:
        p1 = tuple(seg[0][0])
        p2 = tuple(seg[1][0])
        final_canonical_segments.add(tuple(sorted((p1, p2))))
        
    final_result = [np.array([[p1], [p2]], dtype=np.int32) for p1, p2 in final_canonical_segments]
    
    return final_result

def _vectorize_skeleton(skeleton: np.ndarray, poly_epsilon_frac: float) -> list:
    """
    Traces paths in a skeleton image (including loops) and simplifies them into line segments.
    """
    h, w = skeleton.shape
    skel_copy = skeleton.copy()
    paths = []

    def trace_path(start_node):
        if skel_copy[start_node[1], start_node[0]] == 0: return None
        path = [start_node]
        current_point = start_node
        skel_copy[current_point[1], current_point[0]] = 0
        while True:
            y, x = current_point[1], current_point[0]
            found_next = False
            for dy, dx in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and skel_copy[ny, nx] == 255:
                    next_point = (nx, ny)
                    path.append(next_point)
                    skel_copy[ny, nx] = 0
                    current_point = next_point
                    found_next = True
                    break
            if not found_next: return path

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skel_copy[y, x] == 255:
                neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2] == 255) - 1
                if neighbors == 1:
                    path = trace_path((x, y))
                    if path: paths.append(path)
    
    for y in range(h):
        for x in range(w):
            if skel_copy[y, x] == 255:
                path = trace_path((x, y))
                if path:
                    path.append(path[0])
                    paths.append(path)

    raw_segments = []
    for path in paths:
        if len(path) < 2: continue
        path_np = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
        is_closed = np.array_equal(path_np[0], path_np[-1])
        arc_length = cv2.arcLength(path_np, is_closed)
        if arc_length == 0: continue
        epsilon = max(1.0, poly_epsilon_frac * arc_length)
        approx_corners = cv2.approxPolyDP(path_np, epsilon, is_closed)
        if len(approx_corners) > 1:
            for i in range(len(approx_corners) - 1):
                p1 = approx_corners[i][0]
                p2 = approx_corners[i+1][0]
                raw_segments.append(np.array([[p1], [p2]], dtype=np.int32))
            if is_closed:
                p1 = approx_corners[-1][0]
                p2 = approx_corners[0][0]
                raw_segments.append(np.array([[p1], [p2]], dtype=np.int32))
    
    return raw_segments

def detect_lines(
    pil_img: Image.Image,
    poly_epsilon_frac: float = 0.01,
    **kwargs
) -> Tuple[list, int]:
    """
    Detects centerlines of strokes, deduplicates, and splits them at intersections.
    """
    gray = to_uint8_grayscale(pil_img)
    equalized = cv2.equalizeHist(gray)
    inverted = 255 - equalized
    t, line_mask = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    skeleton = _skeletonize(line_mask)
    raw_segments = _vectorize_skeleton(skeleton, poly_epsilon_frac)
    final_segments = _post_process_vectors(raw_segments)
    return final_segments, int(t)

# -------------------------
# Original Segmentation Algorithm (for fill)
# -------------------------

def segment_closed_regions(
    pil_img: Image.Image,
    dilation_radius: int = 2,
    invert_lines: bool = False,
    threshold: Optional[int] = None,
) -> Tuple[PageSegmentation, int]:
    """
    Segment connected non-line regions.
    Returns PageSegmentation object and the threshold value used.
    """
    gray = to_uint8_grayscale(pil_img)
    t = otsu_threshold(gray) if threshold is None else threshold
    if invert_lines:
        line_mask = gray >= t
    else:
        line_mask = gray <= t
    if dilation_radius > 0:
        structure = ndi.generate_binary_structure(2, 2)
        line_mask = ndi.binary_dilation(line_mask, structure=structure, iterations=dilation_radius)
    fillable = ~line_mask
    structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
    labeled, num = ndi.label(fillable, structure=structure)
    h, w = labeled.shape
    border = np.concatenate([labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]])
    border_labels = set(np.unique(border))
    if 0 in border_labels:
        border_labels.remove(0)
    segmentation = PageSegmentation(label_map=labeled.astype(np.int32), num_labels=num, border_labels=border_labels)
    return segmentation, t
