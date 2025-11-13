# vector_tools.py
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Set

import numpy as np
import cv2
from PIL import Image

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
# New, Faster Vectorization and Segmentation
# -------------------------

def detect_lines(
    pil_img: Image.Image,
    **kwargs
) -> Tuple[list, int]:
    """
    Detects line segments in an image using the fast Line Segment Detector (LSD).
    This is much faster and often more accurate than skeletonization-based methods.
    """
    gray = to_uint8_grayscale(pil_img)
    
    # Create a Line Segment Detector instance.
    lsd = cv2.createLineSegmentDetector(0)
    
    # Detect lines in the image
    lines, _, _, _ = lsd.detect(gray)
    
    # The threshold is still needed by the GUI for the label, so we compute it here.
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if lines is None:
        return [], int(t)

    # Convert lines to the format expected by the GUI
    # The format is a list of numpy arrays, each of shape (2, 1, 2)
    final_segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        final_segments.append(np.array([[p1], [p2]], dtype=np.int32))

    return final_segments, int(t)


def segment_closed_regions(
    pil_img: Image.Image,
    dilation_radius: int = 2,
    invert_lines: bool = False,
    threshold: Optional[int] = None,
) -> Tuple[PageSegmentation, int]:
    """
    Segment connected non-line regions using fast OpenCV operations.
    Returns PageSegmentation object and the threshold value used.
    """
    gray = to_uint8_grayscale(pil_img)
    
    # Use fast OpenCV Otsu thresholding
    if threshold is None:
        t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        t = threshold
        
    # Create binary mask of the lines
    if invert_lines:
        _, line_mask_binary = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    else:
        _, line_mask_binary = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)

    # Dilate the lines to close gaps using a fast OpenCV implementation
    if dilation_radius > 0:
        # Use a square kernel for dilation
        kernel_size = dilation_radius * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_mask = cv2.dilate(line_mask_binary, kernel, iterations=1)
    else:
        dilated_mask = line_mask_binary

    # Invert the mask to get fillable areas
    fillable = cv2.bitwise_not(dilated_mask)
    
    # Use cv2.connectedComponents for fast region labeling
    num, labeled = cv2.connectedComponents(fillable, connectivity=8)
    
    h, w = labeled.shape
    border = np.concatenate([labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]])
    border_labels = set(np.unique(border))
    
    # The background label from connectedComponents is 0. The GUI expects 0 to be a non-region.
    if 0 in border_labels:
        border_labels.remove(0)
        
    segmentation = PageSegmentation(label_map=labeled.astype(np.int32), num_labels=num, border_labels=border_labels)
    return segmentation, int(t)