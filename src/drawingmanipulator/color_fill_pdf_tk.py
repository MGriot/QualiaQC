import io
import os
import sys
import time
import math
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox, simpledialog
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Set

import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


# -------------------------
# Utility functions
# -------------------------


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

def numpy_mask_from_label(label_map: np.ndarray, label_id: int) -> np.ndarray:
    """Return boolean mask where label_map == label_id."""
    return label_map == label_id

def pil_mask_from_bool(mask: np.ndarray) -> Image.Image:
    """Convert boolean mask to single-channel 'L' PIL image (0/255)."""
    return Image.fromarray((mask.astype(np.uint8)) * 255, mode="L")

def safe_make_dir(path: str):
    os.makedirs(path, exist_ok=True)


@dataclass
class PageSegmentation:
    label_map: np.ndarray  # int32 [H, W] region labels (0..N); 0 means no-region (edge)
    num_labels: int
    border_labels: Set[int]  # labels touching borders (usually outside "background" regions)


# -------------------------
# Segmentation
# -------------------------


def detect_lines(
    pil_img: Image.Image,
    dilation_radius: int = 2,
    invert_lines: bool = False,
    threshold: Optional[int] = None,
    poly_epsilon_frac: float = 0.0075, # Epsilon for corner detection
) -> Tuple[list, int]:
    """
    Detects line contours, splitting them at sharp angles.
    """
    gray = to_uint8_grayscale(pil_img)
    # Adaptive thresholding is much better at finding lines of varying intensity.
    # THRESH_BINARY_INV assumes dark lines on a light background.
    line_mask_u8 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )
    # The threshold value `t` is not relevant for adaptive threshold, but we return a placeholder.
    t = 127

    # The boolean mask is now derived from the adaptive threshold result.
    line_mask = line_mask_u8 > 0

    if dilation_radius > 0:
        structure = ndi.generate_binary_structure(2, 2)
        line_mask = ndi.binary_dilation(
            line_mask, structure=structure, iterations=dilation_radius
        )
    
    line_mask_u8 = line_mask.astype(np.uint8) * 255
    # Use RETR_EXTERNAL to get only outer contours of lines, simplifying things
    contours, _ = cv2.findContours(line_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_segments = []
    for contour in contours:
        # Treat contours as open curves for approximation
        arc_length = cv2.arcLength(contour, False)
        if arc_length < 10: # Ignore tiny noise contours
            continue

        # 1. Find corners using approxPolyDP
        approx_corners = cv2.approxPolyDP(contour, poly_epsilon_frac * arc_length, False)

        # 2. Create straight-line segments between the found corners
        if len(approx_corners) > 1:
            for i in range(len(approx_corners) - 1):
                p1 = approx_corners[i][0]
                p2 = approx_corners[i+1][0]
                
                # Create a new contour for this single straight segment
                line_segment = np.array([[p1], [p2]], dtype=np.int32)
                final_segments.append(line_segment)
        elif len(contour) > 1: # If it's a blob with no corners, add the original
            final_segments.append(contour)

    return final_segments, t


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

    # Use provided threshold or auto-threshold using Otsu
    t = otsu_threshold(gray) if threshold is None else threshold

    # If artwork is inverted (white lines on black background), allow inverting
    if invert_lines:
        line_mask = gray >= t
    else:
        line_mask = gray <= t

    # Seal gaps by dilating lines (morphological dilation)
    if dilation_radius > 0:
        structure = ndi.generate_binary_structure(2, 2)  # 8-connectivity
        line_mask = ndi.binary_dilation(
            line_mask, structure=structure, iterations=dilation_radius
        )

    # Fillable areas are where NOT line
    fillable = ~line_mask

    # Label connected components in fillable mask
    structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)  # 8-connected
    labeled, num = ndi.label(fillable, structure=structure)

    # Identify border-touching labels (usually outside background)
    h, w = labeled.shape
    border = np.concatenate(
        [labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]]
    )
    border_labels = set(np.unique(border))
    # Label 0 is non-fillable (lines)
    if 0 in border_labels:
        border_labels.remove(0)

    segmentation = PageSegmentation(
        label_map=labeled.astype(np.int32), num_labels=num, border_labels=border_labels
    )
    return segmentation, t


# -------------------------
# Tkinter Application
# -------------------------


class ColorFillPDFApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF Region Color Filler (Tkinter)")

        # State
        self.pdf_doc: Optional[fitz.Document] = None
        self.page_index: int = 0
        self.page_count: int = 0

        self.base_image: Optional[Image.Image] = None  # PIL RGB
        self.overlay_image: Optional[Image.Image] = None  # PIL RGBA
        self.composited_display: Optional[Image.Image] = None  # PIL Image for display
        self.tk_img: Optional[ImageTk.PhotoImage] = None

        self.segmentation: Optional[PageSegmentation] = None
        self.color_groups: list = []  # List of color group dicts
        self.active_color_group_id: Optional[str] = None

        self.dilation_radius: int = 2
        self.invert_lines: bool = False
        self.zoom_level: float = 1.0

        # UI-linked State Variables
        self.highlights_visible = tk.BooleanVar(value=True)
        self.manual_threshold = tk.IntVar(value=127)

        # Manual line definition mode
        self.mode = tk.StringVar(value="fill")  # 'fill' or 'define'
        self.line_contours: list = []
        self.selected_lines: set = set()

        # UI
        self.region_tree: Optional[ttk.Treeview] = None
        self.canvas: Optional[tk.Canvas] = None
        self.status: Optional[tk.StringVar] = None
        self.threshold_label: Optional[tk.Label] = None
        self._build_ui()

    # ---------- UI ----------

    def _build_ui(self):
        # Menu bar
        menubar = tk.Menu(self)

        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open PDF...", command=self.open_pdf)
        filemenu.add_separator()
        filemenu.add_command(
            label="Export Selected Region(s) as PNG...",
            command=self.export_selected_regions_png,
        )
        filemenu.add_command(
            label="Export ALL colored regions as PNGs...",
            command=self.export_all_regions_pngs,
        )
        filemenu.add_command(
            label="Export composited PNG...", command=self.export_composite_png
        )
        filemenu.add_command(
            label="Export current region as PDF...",
            command=self.export_current_region_pdf,
        )
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        editmenu = tk.Menu(menubar, tearoff=0)
        editmenu.add_command(label="Pick Color...", command=self.pick_color)
        editmenu.add_command(label="Clear all fills", command=self.clear_fills)
        menubar.add_cascade(label="Edit", menu=editmenu)

        pagemenu = tk.Menu(menubar, tearoff=0)
        pagemenu.add_command(label="Previous Page", command=self.prev_page)
        pagemenu.add_command(label="Next Page", command=self.next_page)
        menubar.add_cascade(label="Page", menu=pagemenu)

        options = tk.Menu(menubar, tearoff=0)
        options.add_command(label="Set line dilation...", command=self.set_dilation)
        options.add_command(
            label="Toggle invert lines (white-on-black)",
            command=self.toggle_invert_lines,
        )
        menubar.add_cascade(label="Options", menu=options)

        self.config(menu=menubar)

        # Toolbar
        toolbar = tk.Frame(self)
        tk.Button(toolbar, text="Open PDF...", command=self.open_pdf).pack(side=tk.LEFT, padx=2, pady=4)
        tk.Button(toolbar, text="Pick Color", command=self.pick_color).pack(side=tk.LEFT, padx=2, pady=4)
        tk.Button(toolbar, text="Prev Page", command=self.prev_page).pack(side=tk.LEFT, padx=2, pady=4)
        tk.Button(toolbar, text="Next Page", command=self.next_page).pack(side=tk.LEFT, padx=2, pady=4)
        tk.Button(toolbar, text="Clear Fills", command=self.clear_fills).pack(side=tk.LEFT, padx=2, pady=4)
        tk.Button(toolbar, text="Zoom In", command=self._zoom_in).pack(side=tk.LEFT, padx=2, pady=4)
        tk.Button(toolbar, text="Zoom Out", command=self._zoom_out).pack(side=tk.LEFT, padx=2, pady=4)
        tk.Checkbutton(toolbar, text="Show Color Fills", variable=self.highlights_visible, command=self._refresh_display).pack(side=tk.LEFT, padx=2, pady=4)
        toolbar.pack(fill=tk.X)

        # Main content area with PanedWindow
        main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # Left Pane: Controls
        left_pane = tk.Frame(main_pane, width=300)
        left_pane.pack_propagate(False)

        # -- Color Groups --
        group_frame = tk.LabelFrame(left_pane, text="Color Groups")
        group_frame.pack(fill=tk.X, padx=5, pady=5)

        btn_frame = tk.Frame(group_frame)
        btn_frame.pack(fill=tk.X)
        tk.Button(btn_frame, text="New", command=self._add_new_group).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(btn_frame, text="Rename", command=self._rename_selected_group).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(btn_frame, text="Delete", command=self._delete_selected_group).pack(side=tk.LEFT, expand=True, fill=tk.X)

        tree_container = tk.Frame(left_pane)
        tree_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        self.region_tree = ttk.Treeview(tree_container, columns=("color",), show="tree headings", selectmode="extended")
        self.region_tree.heading("#0", text="Region Name / ID")
        self.region_tree.heading("color", text="Color")
        self.region_tree.column("#0", width=150, anchor="w")
        self.region_tree.column("color", width=70, anchor="w")

        tree_scroll = ttk.Scrollbar(tree_container, orient="vertical", command=self.region_tree.yview)
        self.region_tree.configure(yscrollcommand=tree_scroll.set)

        self.region_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.region_tree.bind("<<TreeviewSelect>>", self._on_region_select)
        
        tk.Button(left_pane, text="Remove Selected Region(s)", command=self._remove_selected_region).pack(fill=tk.X, padx=5, pady=5)
        tk.Button(left_pane, text="Merge Selected Regions", command=self._merge_selected_regions).pack(fill=tk.X, padx=5, pady=5)

        # -- Mode Selection --
        mode_frame = tk.LabelFrame(left_pane, text="Interaction Mode")
        mode_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Radiobutton(mode_frame, text="Fill Area", variable=self.mode, value="fill", command=self._on_mode_change).pack(anchor=tk.W)
        tk.Radiobutton(mode_frame, text="Define by Line", variable=self.mode, value="define", command=self._on_mode_change).pack(anchor=tk.W)

        # -- Manual Definition Actions --
        self.define_frame = tk.LabelFrame(left_pane, text="Manual Definition")
        self.define_frame.pack(fill=tk.X, padx=5, pady=5)
        self.create_area_button = tk.Button(self.define_frame, text="Create Area from Selection", command=self._create_area_from_selection)
        self.create_area_button.pack(fill=tk.X, pady=2)
        self.clear_lines_button = tk.Button(self.define_frame, text="Clear Line Selection", command=self._clear_line_selection)
        self.clear_lines_button.pack(fill=tk.X, pady=2)


        # -- Segmentation Settings --
        seg_frame = tk.LabelFrame(left_pane, text="Segmentation")
        seg_frame.pack(fill=tk.X, padx=5, pady=5)

        self.threshold_label = tk.Label(seg_frame, text="Threshold: 127 (Auto)")
        self.threshold_label.pack()

        threshold_slider = tk.Scale(seg_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.manual_threshold, command=self._on_slider_move)
        threshold_slider.pack(fill=tk.X, padx=5)

        seg_btn_frame = tk.Frame(seg_frame)
        seg_btn_frame.pack(fill=tk.X)
        tk.Button(seg_btn_frame, text="Apply Manual", command=lambda: self._resegment_page(manual=True)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(seg_btn_frame, text="Reset to Auto", command=lambda: self._resegment_page(manual=False)).pack(side=tk.LEFT, expand=True, fill=tk.X)

        main_pane.add(left_pane, minsize=250)

        # Right Pane: Canvas
        self.canvas = tk.Canvas(main_pane, bg="#444444", highlightthickness=0, cursor="tcross")
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<Control-Button-4>", self._on_mouse_wheel) # Linux/Windows scroll up
        self.canvas.bind("<Control-Button-5>", self._on_mouse_wheel) # Linux/Windows scroll down
        self.canvas.bind("<Control-MouseWheel>", self._on_mouse_wheel) # Windows/macOS scroll
        self.canvas.bind("<ButtonPress-2>", self._on_drag_start) # Middle mouse button
        self.canvas.bind("<B2-Motion>", self._on_drag_motion)
        main_pane.add(self.canvas, minsize=400)
        
        main_pane.sash_place(0, 300, 0)

        # Status Bar
        self.status = tk.StringVar(value="Open a PDF to begin.")
        status_bar = tk.Label(self, textvariable=self.status, anchor="w")
        status_bar.pack(fill=tk.X)

        self.geometry("1200x900")

    # ---------- Mode Switching and Manual Definition ----------

    def _on_mode_change(self):
        """Called when the user switches between 'Fill' and 'Define' modes."""
        mode = self.mode.get()
        self._clear_line_selection() # Clear selection when changing modes
        if mode == 'fill':
            self.canvas.config(cursor="tcross")
            self.status.set("Mode: Fill Area. Click inside a region to color it.")
            self.create_area_button.config(state=tk.DISABLED)
            self.clear_lines_button.config(state=tk.DISABLED)
        elif mode == 'define':
            self.canvas.config(cursor="hand2")
            self.status.set("Mode: Define by Line. Click to select lines.")
            self.create_area_button.config(state=tk.NORMAL)
            self.clear_lines_button.config(state=tk.NORMAL)
        self._refresh_display()

    def _create_area_from_selection(self):
        """
        Creates a new fillable region by connecting the endpoints of all selected
        line segments, then filling the resulting shape while respecting holes.
        """
        if len(self.selected_lines) < 1:
            messagebox.showwarning("Selection Error", "Please select at least one line segment.")
            return

        if self.base_image is None or self.segmentation is None:
            return

        max_dist = simpledialog.askinteger(
            "Connect Gaps", "Enter max distance (pixels) to connect gaps:",
            initialvalue=20, minvalue=1, maxvalue=200, parent=self
        )
        if max_dist is None: return

        self.status.set("Connecting endpoints and creating area...")
        self.update_idletasks()

        # 1. Create a mask and draw the initially selected segments on it
        line_mask = np.zeros(self.base_image.size[::-1], dtype=np.uint8)
        selected_contours = [self.line_contours[i] for i in self.selected_lines]
        cv2.drawContours(line_mask, selected_contours, -1, 255, 1)

        # 2. Get all unique endpoints from all selected segments
        endpoints = set()
        for contour in selected_contours:
            if len(contour) > 0:
                endpoints.add(tuple(contour[0][0]))
                if len(contour) > 1:
                    endpoints.add(tuple(contour[-1][0]))
        
        endpoint_list = list(endpoints)
        if len(endpoint_list) < 2:
            messagebox.showwarning("Warning", "Not enough unique endpoints to connect gaps.")
        else:
            # 3. Connect all nearby endpoints in a single, robust step
            kdtree = cKDTree(endpoint_list)
            close_pairs = kdtree.query_pairs(r=max_dist)
            
            if close_pairs:
                for (i, j) in close_pairs:
                    p1 = endpoint_list[i]
                    p2 = endpoint_list[j]
                    cv2.line(line_mask, p1, p2, 255, 1)

        # 4. Thicken and close the final perimeter to ensure it's watertight
        kernel = np.ones((3,3), np.uint8)
        closed_line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 5. Find contours and fill the area, respecting holes
        new_contours, hierarchy = cv2.findContours(closed_line_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not new_contours:
            messagebox.showerror("Error", "Could not form a closed area from the selection.")
            self.status.set("Area creation failed.")
            return

        # 6. Create the final region mask
        region_mask = np.zeros(self.base_image.size[::-1], dtype=np.uint8)
        if hierarchy is not None:
            for i, contour in enumerate(new_contours):
                if hierarchy[0][i][3] == -1: # Outer contour
                    cv2.drawContours(region_mask, [contour], -1, 255, -1)
            for i, contour in enumerate(new_contours):
                if hierarchy[0][i][3] != -1: # Inner (hole) contour
                    cv2.drawContours(region_mask, [contour], -1, 0, -1)
        else:
            cv2.drawContours(region_mask, new_contours, -1, 255, -1)

        # 7. Add the new region to the segmentation data
        new_label_id = self.segmentation.num_labels + 1
        self.segmentation.label_map[region_mask == 255] = new_label_id
        self.segmentation.num_labels += 1
        
        # 8. Add new region to the active color group and draw it immediately
        if not self.active_color_group_id:
            messagebox.showwarning("No Active Group", "Area created, but no color group was active. Please select a group and click on the new area to color it.")
        else:
            active_group = next((g for g in self.color_groups if g['id'] == self.active_color_group_id), None)
            if active_group:
                active_group['regions'].add(new_label_id)
                # Directly draw the new region to ensure it appears
                mask_L = pil_mask_from_bool(region_mask)
                color_img = Image.new("RGBA", self.overlay_image.size, active_group['color'])
                self.overlay_image.paste(color_img, (0, 0), mask_L)

        # 9. Clean up and update UI
        self.selected_lines.clear()
        self.status.set(f"Successfully created new Area {new_label_id}.")
        self._update_region_list()
        self._refresh_display()

    def _clear_line_selection(self):
        """Clears the set of selected lines and redraws the canvas."""
        if not self.selected_lines:
            return
        self.selected_lines.clear()
        self.status.set("Line selection cleared.")
        self._refresh_display()


    def _add_new_group(self):
        name = tk.simpledialog.askstring("New Color Group", "Enter a name for the new group:", parent=self)
        if not name: return

        color_info = colorchooser.askcolor(title=f"Pick color for '{name}'")
        if not color_info or not color_info[0]: return
        
        r, g, b = color_info[0]
        rgba = (int(r), int(g), int(b), 150)

        group_id = f"group_{time.time()}" 
        new_group = {"id": group_id, "name": name, "color": rgba, "regions": set()}
        self.color_groups.append(new_group)
        self._update_region_list()
        self.region_tree.selection_set(group_id)

    def _rename_selected_group(self):
        if not self.region_tree: return
        selection = self.region_tree.selection()
        if len(selection) != 1 or not selection[0].startswith("group_"):
            messagebox.showwarning("Warning", "Please select exactly one group folder to rename.")
            return
        
        group_id = selection[0]
        group = next((g for g in self.color_groups if g['id'] == group_id), None)
        if not group: return

        new_name = tk.simpledialog.askstring("Rename Group", "Enter new name:", initialvalue=group['name'], parent=self)
        if new_name:
            group['name'] = new_name
            self._update_region_list()

    def _delete_selected_group(self):
        if not self.region_tree: return
        selection = self.region_tree.selection()
        if not selection: return

        group_ids_to_delete = {item for item in selection if item.startswith("group_")}
        if not group_ids_to_delete:
            messagebox.showwarning("Warning", "No group selected to delete. Select a group folder.")
            return

        if not messagebox.askyesno("Confirm Delete", f"Delete {len(group_ids_to_delete)} selected group(s) and all their regions?"):
            return

        self.color_groups = [g for g in self.color_groups if g['id'] not in group_ids_to_delete]
        
        self._redraw_all_overlays()
        self._refresh_display()
        self._update_region_list()
        self.status.set(f"Deleted {len(group_ids_to_delete)} group(s).")

    def _remove_selected_region(self):
        if not self.region_tree: return
        selection = self.region_tree.selection()
        if not selection: return

        regions_to_remove = {item for item in selection if item.startswith("region_")}
        if not regions_to_remove:
            messagebox.showwarning("Warning", "No regions (Areas) selected to remove.")
            return

        for region_id_str in regions_to_remove:
            region_id = int(region_id_str.split("_")[1])
            parent_id = self.region_tree.parent(region_id_str)
            group = next((g for g in self.color_groups if g['id'] == parent_id), None)

            if group and region_id in group['regions']:
                group['regions'].remove(region_id)
        
        self._redraw_all_overlays()
        self._refresh_display()
        self._update_region_list()
        self.status.set(f"Removed {len(regions_to_remove)} region(s).")

    def _merge_selected_regions(self):
        if not self.region_tree or self.segmentation is None: return

        selection = self.region_tree.selection()
        region_ids_str = {item for item in selection if item.startswith("region_")}

        if len(region_ids_str) < 2:
            messagebox.showwarning("Selection Error", "Please select at least two regions (Areas) to merge.")
            return

        # Verify all selected regions belong to the same group
        parent_ids = {self.region_tree.parent(item) for item in region_ids_str}
        if len(parent_ids) > 1:
            messagebox.showwarning("Selection Error", "All regions to be merged must belong to the same color group.")
            return
        
        parent_id = parent_ids.pop()
        if not parent_id:
            messagebox.showerror("Error", "Could not find the parent color group.")
            return

        group = next((g for g in self.color_groups if g['id'] == parent_id), None)
        if not group:
            messagebox.showerror("Error", "Could not find the parent color group data.")
            return

        # Prompt user for merge strength
        dilation_amount = simpledialog.askinteger(
            "Merge Strength", 
            "Enter connection strength (higher for distant regions):",
            initialvalue=self.dilation_radius + 3, minvalue=1, maxvalue=100, parent=self
        )
        if dilation_amount is None: # User cancelled
            return

        # --- Start of new logic ---
        label_map = self.segmentation.label_map
        region_ids = sorted([int(s.split("_")[1]) for s in region_ids_str])
        
        # 1. Identify protected regions that should not be overwritten
        labels_to_merge = set(region_ids)
        all_labels_on_page = set(np.unique(label_map)) - {0}
        protected_labels = all_labels_on_page - labels_to_merge
        
        protected_mask = np.zeros_like(label_map, dtype=bool)
        for label_id in protected_labels:
            protected_mask[label_map == label_id] = True

        # 2. Iteratively merge regions
        target_label, *source_labels = region_ids
        self.status.set(f"Merging {len(source_labels) + 1} regions...")
        self.update_idletasks()

        for source_label in source_labels:
            mask_target = (label_map == target_label)
            mask_source = (label_map == source_label)

            dilated_target = ndi.binary_dilation(mask_target, iterations=dilation_amount)
            dilated_source = ndi.binary_dilation(mask_source, iterations=dilation_amount)
            
            # 3. Create a fat bridge and ensure it doesn't touch protected regions
            bridge_mask = dilated_target & dilated_source
            safe_bridge_mask = bridge_mask & ~protected_mask

            # 4. Update the label map with the source region and the safe, fat bridge
            label_map[mask_source] = target_label
            label_map[safe_bridge_mask] = target_label

            if source_label in group["regions"]:
                group["regions"].remove(source_label)
        # --- End of new logic ---

        self.status.set(f"Merge complete. New region is Area {target_label}.")
        self._redraw_all_overlays()
        self._refresh_display()
        self._update_region_list()

    def _zoom_in(self):
        self.zoom_level *= 1.25
        self._refresh_display()

    def _zoom_out(self):
        self.zoom_level /= 1.25
        if self.zoom_level < 0.1: self.zoom_level = 0.1
        self._refresh_display()

    def _on_mouse_wheel(self, event):
        if event.delta > 0 or event.num == 4:
            self.zoom_level *= 1.1
        else:
            self.zoom_level /= 1.1
        if self.zoom_level < 0.1: self.zoom_level = 0.1
        self._refresh_display()
        return "break"

    def _on_drag_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def _on_drag_motion(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    # ---------- PDF & Segmentation ----------

    def open_pdf(self):
        path = filedialog.askopenfilename(title="Open PDF", filetypes=[("PDF files", "*.pdf")])
        if not path: return
        try:
            self.pdf_doc = fitz.open(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open PDF:\n{e}")
            return

        self.page_count = self.pdf_doc.page_count
        self.page_index = 0
        self.status.set(f"Loaded: {os.path.basename(path)} | Pages: {self.page_count}")
        self._load_page(self.page_index)

    def _load_page(self, index: int, threshold: Optional[int] = None):
        if not self.pdf_doc or not (0 <= index < self.pdf_doc.page_count): return

        page = self.pdf_doc.load_page(index)
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        self.base_image = pil_from_pixmap(pix).convert("RGB")
        
        self.color_groups.clear()
        self.active_color_group_id = None
        self._update_region_list()

        self.status.set("Segmenting regions and detecting lines...")
        self.update_idletasks()
        
        # Automatic segmentation for "Fill" mode
        self.segmentation, used_threshold = segment_closed_regions(
            self.base_image, self.dilation_radius, self.invert_lines, threshold
        )
        # Line detection for "Define by Line" mode
        self.line_contours, _ = detect_lines(
            self.base_image, self.dilation_radius, self.invert_lines, threshold
        )
        self.selected_lines.clear()
        self.manual_threshold.set(used_threshold)
        mode = "(Auto)" if threshold is None else "(Manual)"
        self.threshold_label.config(text=f"Threshold: {used_threshold} {mode}")

        num_regions = self.segmentation.num_labels
        self.status.set(f"Page {self.page_index + 1}/{self.page_count} | Regions: {num_regions} | Threshold: {used_threshold} {mode}")
        
        self._redraw_all_overlays()
        self._refresh_display()

    def _resegment_page(self, manual: bool):
        if self.base_image is None: return
        if not messagebox.askyesno("Confirm Resegment", "This will clear all colored regions on the current page. Proceed?"):
            return
        
        threshold = self.manual_threshold.get() if manual else None
        self._load_page(self.page_index, threshold=threshold)

    def prev_page(self):
        if self.pdf_doc and self.page_index > 0:
            self.page_index -= 1
            self._load_page(self.page_index)

    def next_page(self):
        if self.pdf_doc and self.page_index < self.page_count - 1:
            self.page_index += 1
            self._load_page(self.page_index)

    # ---------- Display handling ----------

    def _generate_segmentation_preview(self):
        if self.segmentation is None:
            return self.base_image.convert("RGBA")

        # Use a morphological gradient to find the boundaries between labeled regions
        label_map = self.segmentation.label_map
        structure = ndi.generate_binary_structure(2, 1)
        eroded = ndi.grey_erosion(label_map, footprint=structure)
        dilated = ndi.grey_dilation(label_map, footprint=structure)
        
        # Boundaries are where the dilated and eroded maps differ
        boundaries = (dilated != eroded) & (label_map > 0) # Exclude boundaries of the background
        
        # Create a PIL mask for the boundaries
        boundary_mask_pil = pil_mask_from_bool(boundaries)

        # Start with the base image
        preview_img = self.base_image.convert("RGBA")
        
        # Create a bright magenta overlay for the boundaries
        boundary_overlay = Image.new("RGBA", preview_img.size, (255, 0, 255, 200))
        
        # Paste the boundaries onto the preview image
        preview_img.paste(boundary_overlay, (0, 0), boundary_mask_pil)
        
        return preview_img

    def _refresh_display(self, *_):
        if self.base_image is None: return
        
        if self.highlights_visible.get():
            # Show base image + colored fills
            base_rgba = self.base_image.convert("RGBA")
            comp = alpha_composite(base_rgba, self.overlay_image)
        else:
            # Show segmentation preview (base image + all region borders)
            comp = self._generate_segmentation_preview()

        # If in define mode, draw all detected segments and highlight selected ones
        if self.mode.get() == 'define':
            draw = ImageDraw.Draw(comp)
            # 1. Draw all detected segments faintly to show what's clickable
            if self.line_contours:
                for contour in self.line_contours:
                    points = [tuple(p[0]) for p in contour]
                    if len(points) > 1:
                        draw.line(points, fill=(0, 191, 255, 100), width=2) # Faint deep sky blue for all segments

            # 2. Draw selected segments brightly on top for emphasis
            if self.selected_lines:
                for line_idx in self.selected_lines:
                    if line_idx < len(self.line_contours):
                        contour = self.line_contours[line_idx]
                        points = [tuple(p[0]) for p in contour]
                        if len(points) > 1:
                            draw.line(points, fill=(50, 205, 50, 220), width=4) # Bright lime green for selected
            del draw

        self.composited_display = comp

        iw, ih = comp.size
        scaled_iw = int(iw * self.zoom_level)
        scaled_ih = int(ih * self.zoom_level)

        disp_img = comp.resize((scaled_iw, scaled_ih), Image.Resampling.BILINEAR)
        self.tk_img = ImageTk.PhotoImage(disp_img)

        self.canvas.config(scrollregion=(0, 0, scaled_iw, scaled_ih))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_img, anchor="nw")

    def on_canvas_resize(self, event):
        self._refresh_display()

    # ---------- Interaction ----------

    def _update_region_list(self):
        if not self.region_tree: return
        
        selection = self.region_tree.selection()
        
        for item in self.region_tree.get_children():
            self.region_tree.delete(item)
            
        for group in sorted(self.color_groups, key=lambda g: g['name']):
            color_hex = f"#{group['color'][0]:02x}{group['color'][1]:02x}{group['color'][2]:02x}"
            group_node = self.region_tree.insert("", "end", iid=group['id'], text=group['name'], values=(color_hex,))
            for region_id in sorted(list(group['regions'])):
                self.region_tree.insert(group_node, "end", text=f"  Area {region_id}", iid=f"region_{region_id}")
        
        if selection:
            try: self.region_tree.selection_set(selection)
            except tk.TclError: pass # Item might have been deleted

    def _on_region_select(self, event):
        selection = self.region_tree.selection()
        if not selection: self.active_color_group_id = None; return
        
        last_selected_id = selection[-1]
        if last_selected_id.startswith("group_"):
            self.active_color_group_id = last_selected_id
            self.status.set(f"Active group: '{self.region_tree.item(last_selected_id, 'text')}'. Click to add regions.")
        elif last_selected_id.startswith("region_"):
            parent_id = self.region_tree.parent(last_selected_id)
            if parent_id: self.active_color_group_id = parent_id
            self.status.set(f"{len(selection)} region(s) selected.")

    def _on_slider_move(self, val_str):
        val = int(val_str)
        self.threshold_label.config(text=f"Threshold: {val} (Manual)")

    def pick_color(self):
        messagebox.showinfo("Info", "Create a 'New Group' to pick a color, or select an existing group to make it active.")

    def _handle_line_selection(self, event):
        """Handles clicking on the canvas to select a line in 'define' mode."""
        if not self.line_contours:
            self.status.set("No lines detected on this page.")
            return

        canvas_x, canvas_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        x, y = int(canvas_x / self.zoom_level), int(canvas_y / self.zoom_level)
        click_point = (x, y)

        # Find the closest contour to the click point
        min_dist = float('inf')
        best_idx = -1
        for i, contour in enumerate(self.line_contours):
            # pointPolygonTest returns positive (inside), negative (outside), or zero (on the line).
            # The third argument `True` makes it return the signed distance.
            dist = cv2.pointPolygonTest(contour, click_point, True)
            abs_dist = abs(dist)
            if abs_dist < min_dist:
                min_dist = abs_dist
                best_idx = i
        
        # Select the contour if the click is reasonably close to it
        # The threshold (e.g., 10 pixels) can be adjusted
        if best_idx != -1 and min_dist < 25: # Increased tolerance for easier selection
            if best_idx in self.selected_lines:
                self.selected_lines.remove(best_idx)
                self.status.set(f"Deselected line {best_idx}. Total selected: {len(self.selected_lines)}")
            else:
                self.selected_lines.add(best_idx)
                self.status.set(f"Selected line {best_idx}. Total selected: {len(self.selected_lines)}")
            self._refresh_display()
        else:
            self.status.set("No line found near click.")

    def on_canvas_click(self, event):
        if self.base_image is None: return

        # Route click based on mode
        if self.mode.get() == 'define':
            self._handle_line_selection(event)
            return
        
        # --- Original 'fill' mode logic ---
        if self.segmentation is None: return

        canvas_x, canvas_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        x, y = int(canvas_x / self.zoom_level), int(canvas_y / self.zoom_level)

        if not (0 <= x < self.base_image.width and 0 <= y < self.base_image.height): return

        label = int(self.segmentation.label_map[y, x])
        if label == 0 or label in self.segmentation.border_labels:
            self.status.set("Clicked on a line or non-fillable area.")
            return

        if not self.active_color_group_id:
            messagebox.showwarning("No Active Group", "Please select a color group from the list before adding a region.")
            return
            
        active_group = next((g for g in self.color_groups if g['id'] == self.active_color_group_id), None)
        if not active_group: messagebox.showerror("Error", "Active group not found. Please re-select."); return

        for group in self.color_groups: # Remove from any other group first
            if label in group['regions']: group['regions'].remove(label)
        
        active_group['regions'].add(label)
        
        self._redraw_all_overlays()
        self._refresh_display()
        self._update_region_list()
        self.status.set(f"Added Region {label} to group '{active_group['name']}'.")

    def _apply_overlay_for_region(self, region_id: int, rgba: Tuple[int, int, int, int]):
        if self.segmentation is None or self.overlay_image is None: return
        mask_bool = numpy_mask_from_label(self.segmentation.label_map, region_id)
        mask_L = pil_mask_from_bool(mask_bool)
        color_img = Image.new("RGBA", self.overlay_image.size, rgba)
        self.overlay_image.paste(color_img, (0, 0), mask_L)

        center_y, center_x = ndi.center_of_mass(mask_bool)
        if math.isnan(center_y) or math.isnan(center_x): return

        brightness = (rgba[0] * 299 + rgba[1] * 587 + rgba[2] * 114) / 1000
        text_color = (0, 0, 0, 255) if brightness > 128 else (255, 255, 255, 255)

        draw = ImageDraw.Draw(self.overlay_image)
        try: font = ImageFont.truetype("arial.ttf", 24)
        except IOError: font = ImageFont.load_default()
        
        text = str(region_id)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        
        draw.text((center_x - text_w / 2, center_y - text_h / 2), text, font=font, fill=text_color)

    def _redraw_all_overlays(self):
        if self.base_image is None: return
        self.overlay_image = Image.new("RGBA", self.base_image.size, (0, 0, 0, 0))
        for group in self.color_groups:
            for region_id in group['regions']:
                self._apply_overlay_for_region(region_id, group['color'])

    def clear_fills(self):
        if self.base_image is None: return
        self.color_groups.clear()
        self._redraw_all_overlays()
        self._refresh_display()
        self._update_region_list()
        self.status.set("Cleared all fills and groups.")

    def set_dilation(self):
        val_str = simpledialog.askstring("Set Line Dilation", "Dilation radius (0-10):", initialvalue=str(self.dilation_radius))
        if val_str is None: return
        try:
            val = int(val_str)
            if not (0 <= val <= 10): raise ValueError
            self.dilation_radius = val
            self._resegment_page(manual=False) # Resegment with new dilation
        except (ValueError, TypeError):
            messagebox.showerror("Error", "Invalid input. Please enter an integer between 0 and 10.")

    def toggle_invert_lines(self):
        self.invert_lines = not self.invert_lines
        status_msg = "ON" if self.invert_lines else "OFF"
        self.status.set(f"Invert lines is now {status_msg}.")
        if self.base_image is not None:
            self._resegment_page(manual=False) # Resegment with inverted lines

    # ---------- Export ----------

    def _get_selected_regions_for_export(self) -> set:
        if not self.region_tree: return set()
        selection = self.region_tree.selection()
        if not selection: return set()

        regions_to_export = set()
        all_groups = {g['id']: g for g in self.color_groups}

        for item_id in selection:
            if item_id.startswith("group_"):
                group = all_groups.get(item_id)
                if group:
                    for region_id in group['regions']:
                        regions_to_export.add((region_id, group['color']))
            elif item_id.startswith("region_"):
                region_id = int(item_id.split("_")[1])
                parent_id = self.region_tree.parent(item_id)
                group = all_groups.get(parent_id)
                if group:
                    regions_to_export.add((region_id, group['color']))
        
        return regions_to_export

    def export_selected_regions_png(self):
        if self.base_image is None or self.segmentation is None: return

        regions_to_export = self._get_selected_regions_for_export()
        if not regions_to_export:
            messagebox.showinfo("Info", "No regions are selected. Select items from the list on the left.")
            return

        export_img = Image.new("RGBA", self.base_image.size, (0, 0, 0, 0))

        for region_id, rgba in regions_to_export:
            mask_bool = numpy_mask_from_label(self.segmentation.label_map, region_id)
            mask_L = pil_mask_from_bool(mask_bool)
            color_img = Image.new("RGBA", self.base_image.size, rgba)
            export_img.paste(color_img, (0, 0), mask_L)

        path = filedialog.asksaveasfilename(
            title="Save Selected Regions as PNG",defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialfile=f"page{self.page_index+1:02d}_selected.png",
        )
        if not path: return
        try:
            export_img.save(path, format="PNG")
            self.status.set(f"Saved {len(regions_to_export)} regions to {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save PNG:\n{e}")

    def export_all_regions_pngs(self):
        if self.base_image is None or not self.color_groups: messagebox.showinfo("Info", "No regions to export."); return
        outdir = filedialog.askdirectory(title="Select output directory for PNGs")
        if not outdir: return
        
        count = 0
        try:
            for group in self.color_groups:
                rgba = group['color']
                for region_id in group['regions']:
                    mask_bool = numpy_mask_from_label(self.segmentation.label_map, region_id)
                    mask_L = pil_mask_from_bool(mask_bool)
                    region_img = Image.new("RGBA", self.base_image.size, (0, 0, 0, 0))
                    color_img = Image.new("RGBA", self.base_image.size, rgba)
                    region_img.paste(color_img, (0, 0), mask_L)

                    fname = f"page{self.page_index+1:02d}_group_{group['name']}_region_{region_id}.png"
                    region_img.save(os.path.join(outdir, fname), format="PNG")
                    count += 1
            self.status.set(f"Saved {count} region PNGs to {outdir}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export region PNGs:\n{e}")

    def export_composite_png(self):
        if self.composited_display is None: messagebox.showinfo("Info", "Nothing to export."); return
        
        full_comp = alpha_composite(self.base_image.convert("RGBA"), self.overlay_image)
        path = filedialog.asksaveasfilename(
            title="Save composited PNG",defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialfile=f"page{self.page_index+1:02d}_composite.png"
        )
        if not path: return
        try:
            full_comp.save(path, format="PNG")
            self.status.set(f"Saved composite PNG to {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save PNG:\n{e}")

    def export_current_region_pdf(self):
        if self.base_image is None or self.segmentation is None: return

        selection = self.region_tree.selection()
        if len(selection) != 1 or not selection[0].startswith("region_"):
            messagebox.showwarning("Invalid Selection", "Please select exactly one region (Area) to export as PDF.")
            return

        regions_to_export = self._get_selected_regions_for_export()
        if not regions_to_export: return

        region_id, rgba = list(regions_to_export)[0]

        mask_bool = numpy_mask_from_label(self.segmentation.label_map, region_id)
        mask_L = pil_mask_from_bool(mask_bool)
        region_img = Image.new("RGBA", self.base_image.size, (0, 0, 0, 0))
        color_img = Image.new("RGBA", self.base_image.size, rgba)
        region_img.paste(color_img, (0, 0), mask_L)

        png_bytes = io.BytesIO()
        region_img.save(png_bytes, format="PNG")
        png_bytes = png_bytes.getvalue()

        path = filedialog.asksaveasfilename(
            title="Save region as PDF",defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            initialfile=f"page{self.page_index+1:02d}_region_{region_id}.pdf"
        )
        if not path: return

        try:
            h, w = self.base_image.size[1], self.base_image.size[0]
            doc = fitz.open()
            page = doc.new_page(width=w, height=h)
            rect = fitz.Rect(0, 0, w, h)
            page.insert_image(rect, stream=png_bytes, keep_proportion=False)
            doc.save(path)
            doc.close()
            self.status.set(f"Saved region PDF to {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save PDF:\n{e}")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    app = ColorFillPDFApp()
    app.mainloop()