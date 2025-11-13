# ui_helpers.py
import math
import tkinter as tk
from typing import Tuple
import fitz
from PIL import Image, ImageDraw, ImageTk, ImageFont
import numpy as np
from scipy import ndimage as ndi

from .vector_tools import PageSegmentation, pil_mask_from_bool, numpy_mask_from_label

def pil_from_pixmap(pix: fitz.Pixmap) -> Image.Image:
    """Convert PyMuPDF Pixmap to PIL Image with alpha if present."""
    if pix.alpha:  # retain transparency if any
        mode = "RGBA"
        data = pix.samples
    else:
        mode = "RGB"
        data = pix.samples
    img = Image.frombytes(mode, (pix.width, pix.height), bytes(data))
    return img

def alpha_composite(base_rgba: Image.Image, overlay_rgba: Image.Image) -> Image.Image:
    """Alpha-composite two RGBA PIL images."""
    # Ensure same size
    if base_rgba.size != overlay_rgba.size:
        overlay_rgba = overlay_rgba.resize(base_rgba.size, Image.NEAREST)
    out = base_rgba.copy()
    out.alpha_composite(overlay_rgba)
    return out

def color_tuple_to_rgba(
    color_tuple: Tuple[int, int, int],
    alpha: int = 150
) -> Tuple[int, int, int, int]:
    r, g, b = color_tuple
    return (int(r), int(g), int(b), int(alpha))

def draw_lines_on_image(img, line_contours, selected_lines=None, vector_groups=None, colors=None):
    """Draws all detected line segments and highlights selected and grouped ones."""
    if selected_lines is None: selected_lines = set()
    if vector_groups is None: vector_groups = []
    if colors is None:
        colors = {
            "default": (0, 191, 255, 100),   # Faint deep sky blue
            "selected": (50, 205, 50, 220),  # Bright lime green
            "grouped": (255, 255, 0, 150),    # Yellow
        }
    
    draw = ImageDraw.Draw(img)
    
    # Create a map of index to group color
    line_to_group_color = {}
    group_color = colors.get("grouped", (255, 255, 0, 150))
    for group in vector_groups:
        for line_idx in group:
            line_to_group_color[line_idx] = group_color

    # Draw all segments
    for i, contour in enumerate(line_contours):
        if contour is None: continue
        pts = [tuple(p[0]) for p in contour]
        if len(pts) < 2: continue
        
        width = 2
        # Default color
        color = colors.get("default", (0, 191, 255, 100))
        
        # Group color
        if i in line_to_group_color:
            color = line_to_group_color[i]
        
        # Selection color (overrides all others)
        if i in selected_lines:
            color = colors.get("selected", (50, 205, 50, 220))
            width = 4
            
        draw.line(pts, fill=color, width=width)
    del draw

def draw_region_label(overlay_image: Image.Image, region_id: int, mask_bool: np.ndarray, rgba: Tuple[int, int, int, int]):
    """Draws the region ID number in the center of the region."""
    center_y, center_x = ndi.center_of_mass(mask_bool)
    if math.isnan(center_y) or math.isnan(center_x): return

    brightness = (rgba[0] * 299 + rgba[1] * 587 + rgba[2] * 114) / 1000
    text_color = (0, 0, 0, 255) if brightness > 128 else (255, 255, 255, 255)

    draw = ImageDraw.Draw(overlay_image)
    try: font = ImageFont.truetype("arial.ttf", 24)
    except IOError: font = ImageFont.load_default()
    
    text = str(region_id)
    # Use textbbox for more accurate centering
    try:
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    except AttributeError: # Fallback for older Pillow versions
        text_w, text_h = draw.textsize(text, font=font)

    draw.text((center_x - text_w / 2, center_y - text_h / 2), text, font=font, fill=text_color)

def generate_segmentation_preview(base_image: Image.Image, segmentation: PageSegmentation) -> Image.Image:
    """Generates an image with region boundaries highlighted."""
    # Use a morphological gradient to find the boundaries between labeled regions
    label_map = segmentation.label_map
    structure = ndi.generate_binary_structure(2, 1)
    eroded = ndi.grey_erosion(label_map, footprint=structure)
    dilated = ndi.grey_dilation(label_map, footprint=structure)
    
    # Boundaries are where the dilated and eroded maps differ
    boundaries = (dilated != eroded) & (label_map > 0) # Exclude boundaries of the background
    
    # Create a PIL mask for the boundaries
    boundary_mask_pil = pil_mask_from_bool(boundaries)

    # Start with the base image
    preview_img = base_image.convert("RGBA")
    
    # Create a bright magenta overlay for the boundaries
    boundary_overlay = Image.new("RGBA", preview_img.size, (255, 0, 255, 200))
    
    # Paste the boundaries onto the preview image
    preview_img.paste(boundary_overlay, (0, 0), boundary_mask_pil)
    
    return preview_img


def refresh_vector_canvas(canvas: tk.Canvas, line_contours: list, zoom: float = 1.0) -> dict:
    """
    Draws line contours on a Tkinter canvas, sets the scrollregion,
    and returns a map from canvas item ID to contour index.
    """
    canvas.delete("all")
    line_map = {}
    for i, contour in enumerate(line_contours):
        if contour is None: continue
        pts = [tuple(p[0]) for p in contour]
        if len(pts) < 2:
            continue
        
        # Flatten the list of points and apply zoom
        scaled_pts = [coord * zoom for point in pts for coord in point]
        
        item_id = canvas.create_line(*scaled_pts, fill="black", width=2)
        line_map[item_id] = i

    # Correctly set the scrollregion based on the drawn content
    bbox = canvas.bbox("all")
    if bbox:
        canvas.config(scrollregion=bbox)

    return line_map