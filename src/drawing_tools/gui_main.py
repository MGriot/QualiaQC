# gui_main.py
import io
import os
import time
import math
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox, simpledialog
from typing import Optional, Tuple, Set

import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

# Import from our refactored modules
from .vector_tools import (
    detect_lines,
    segment_closed_regions,
    PageSegmentation,
    numpy_mask_from_label,
    pil_mask_from_bool,
    safe_make_dir,
)
from .ui_helpers import (
    alpha_composite,
    draw_lines_on_image,
    draw_region_label,
    generate_segmentation_preview,
    pil_from_pixmap,
)


class ColorFillPDFApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF Region Color & Vector Editor")

        # State
        self.pdf_doc: Optional[fitz.Document] = None
        self.page_index: int = 0
        self.page_count: int = 0

        self.base_image: Optional[Image.Image] = None
        self.overlay_image: Optional[Image.Image] = None
        self.composited_display: Optional[Image.Image] = None
        self.tk_img: Optional[ImageTk.PhotoImage] = None

        self.segmentation: Optional[PageSegmentation] = None
        self.color_groups: list = []
        self.active_color_group_id: Optional[str] = None

        self.dilation_radius: int = 2
        self.invert_lines: bool = False
        self.zoom_level: float = 1.0

        # UI-linked State Variables
        self.background_visible = tk.BooleanVar(value=True)
        self.background_visible.trace_add("write", self._refresh_display)
        self.highlights_visible = tk.BooleanVar(value=True)
        self.manual_threshold = tk.IntVar(value=127)
        self.vector_move_x = tk.StringVar(value="0")
        self.vector_move_y = tk.StringVar(value="0")
        self.vector_rotate_angle = tk.StringVar(value="0")
        self.create_with_holes = tk.BooleanVar(value=False)

        # Color and Opacity State
        self.vector_colors = {
            "default": (255, 0, 255, 255),  # Magenta
            "selected": (0, 255, 0, 255),   # Green
            "grouped": (255, 255, 0, 255),  # Yellow
        }
        self.overlay_alpha = tk.IntVar(value=150)

        # Unified mode and selection state
        self.mode = tk.StringVar(value="fill")
        self.line_contours: list = []
        self.selected_line_indices: set = set()
        self.vector_groups: list = [] # List of sets, where each set contains indices of grouped lines
        
        # Drag selection state
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.drag_rect_id = None

        # UI
        self.region_tree: Optional[ttk.Treeview] = None
        self.canvas: Optional[tk.Canvas] = None
        self.status: Optional[tk.StringVar] = None
        self.threshold_label: Optional[tk.Label] = None
        self._build_ui()

    # ---------- UI Construction ----------

    def _build_ui(self):
        self._build_menu()
        self._build_toolbar()

        main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_pane.pack(fill=tk.BOTH, expand=True)
        
        left_pane = self._build_left_pane(main_pane)
        main_pane.add(left_pane, minsize=300)

        self.canvas = tk.Canvas(main_pane, bg="#444444", highlightthickness=0, cursor="tcross")
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_drag_end)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<Control-Button-4>", self._on_mouse_wheel) # Linux/Windows scroll up
        self.canvas.bind("<Control-Button-5>", self._on_mouse_wheel) # Linux/Windows scroll down
        self.canvas.bind("<Control-MouseWheel>", self._on_mouse_wheel) # Windows/macOS scroll
        self.canvas.bind("<ButtonPress-2>", self._on_drag_start) # Middle mouse button
        self.canvas.bind("<B2-Motion>", self._on_drag_motion)
        self.canvas.bind("<Delete>", lambda e: self._delete_selected_vectors())
        main_pane.add(self.canvas, minsize=400)
        main_pane.sash_place(0, 300, 0)

        self.status = tk.StringVar(value="Open a PDF to begin.")
        status_bar = tk.Label(self, textvariable=self.status, anchor="w")
        status_bar.pack(fill=tk.X)

        self._on_mode_change() # Set initial UI state
        self.geometry("1200x900")

    def _build_left_pane(self, parent):
        left_pane = tk.Frame(parent)
        
        # Mode Selection
        mode_frame = tk.LabelFrame(left_pane, text="Interaction Mode")
        mode_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Radiobutton(mode_frame, text="Fill Area", variable=self.mode, value="fill", command=self._on_mode_change).pack(anchor=tk.W)
        tk.Radiobutton(mode_frame, text="Edit Vectors", variable=self.mode, value="define", command=self._on_mode_change).pack(anchor=tk.W)

        # Color Groups (for Fill mode)
        self.fill_tools_frame = tk.Frame(left_pane)
        self.fill_tools_frame.pack(fill=tk.BOTH, expand=True)
        group_frame = tk.LabelFrame(self.fill_tools_frame, text="Color Groups")
        group_frame.pack(fill=tk.X, padx=5, pady=5)
        btn_frame = tk.Frame(group_frame)
        btn_frame.pack(fill=tk.X)
        tk.Button(btn_frame, text="New", command=self._add_new_group).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(btn_frame, text="Set Color", command=self._change_group_color).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(btn_frame, text="Rename", command=self._rename_selected_group).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(btn_frame, text="Delete", command=self._delete_selected_group).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tree_container = tk.Frame(self.fill_tools_frame)
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
        tk.Button(self.fill_tools_frame, text="Remove Selected Region(s)", command=self._remove_selected_region).pack(fill=tk.X, padx=5, pady=5)
        tk.Button(self.fill_tools_frame, text="Merge Selected Regions", command=self._merge_selected_regions).pack(fill=tk.X, padx=5, pady=5)

        # Vector Tools (for Define mode)
        self.define_tools_frame = tk.Frame(left_pane)
        # Packed/unpacked in _on_mode_change
        vec_actions_frame = tk.LabelFrame(self.define_tools_frame, text="Vector Actions")
        vec_actions_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(vec_actions_frame, text="Create Area from Selection", command=self._create_area_from_selection).pack(fill=tk.X, pady=2)
        tk.Checkbutton(vec_actions_frame, text="Create with Holes", variable=self.create_with_holes).pack(anchor=tk.W, padx=5)
        tk.Button(vec_actions_frame, text="Connect Endpoints", command=self._connect_selected_vectors).pack(fill=tk.X, pady=2)
        tk.Button(vec_actions_frame, text="Delete Selection", command=self._delete_selected_vectors).pack(fill=tk.X, pady=2)
        tk.Button(vec_actions_frame, text="Clear Selection", command=self._clear_line_selection).pack(fill=tk.X, pady=2)
        
        grouping_frame = tk.LabelFrame(self.define_tools_frame, text="Grouping")
        grouping_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(grouping_frame, text="Group Selection", command=self._group_selected_vectors).pack(fill=tk.X, pady=2)
        tk.Button(grouping_frame, text="Ungroup Selection", command=self._ungroup_selected_vectors).pack(fill=tk.X, pady=2)

        transform_frame = tk.LabelFrame(self.define_tools_frame, text="Transform Selection")
        transform_frame.pack(fill=tk.X, padx=5, pady=5)
        move_frame = tk.Frame(transform_frame)
        move_frame.pack(fill=tk.X, pady=2)
        tk.Label(move_frame, text="Move dX:").pack(side=tk.LEFT)
        tk.Entry(move_frame, textvariable=self.vector_move_x, width=5).pack(side=tk.LEFT)
        tk.Label(move_frame, text="dY:").pack(side=tk.LEFT)
        tk.Entry(move_frame, textvariable=self.vector_move_y, width=5).pack(side=tk.LEFT)
        tk.Button(move_frame, text="Move", command=self._move_selected_vectors).pack(side=tk.LEFT, padx=5)
        rotate_frame = tk.Frame(transform_frame)
        rotate_frame.pack(fill=tk.X, pady=2)
        tk.Label(rotate_frame, text="Angle:").pack(side=tk.LEFT)
        tk.Entry(rotate_frame, textvariable=self.vector_rotate_angle, width=5).pack(side=tk.LEFT)
        tk.Button(rotate_frame, text="Rotate", command=self._rotate_selected_vectors).pack(side=tk.LEFT, padx=5)

        # Segmentation Settings (always visible)
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

        return left_pane

    def _build_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open PDF...", command=self.open_pdf)
        filemenu.add_command(label="Open Image...", command=self.open_image)
        filemenu.add_separator()
        filemenu.add_command(label="Export Selected Region(s) as PNG...", command=self.export_selected_regions_png)
        filemenu.add_command(label="Export ALL colored regions as PNGs...", command=self.export_all_regions_pngs)
        filemenu.add_command(label="Export composited PNG...", command=self.export_composite_png)
        filemenu.add_command(label="Export current region as PDF...", command=self.export_current_region_pdf)
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
        options.add_command(label="Toggle invert lines (white-on-black)", command=self.toggle_invert_lines)
        options.add_separator()
        options.add_command(label="Set Vector Color...", command=self._set_vector_color)
        options.add_command(label="Set Fill Opacity...", command=self._set_overlay_alpha)
        menubar.add_cascade(label="Options", menu=options)
        self.config(menu=menubar)

    def _build_toolbar(self):
        toolbar = tk.Frame(self)
        tk.Button(toolbar, text="Open PDF...", command=self.open_pdf).pack(side=tk.LEFT, padx=2, pady=4)
        tk.Button(toolbar, text="Open Image...", command=self.open_image).pack(side=tk.LEFT, padx=2, pady=4)
        self.prev_page_btn = tk.Button(toolbar, text="Prev Page", command=self.prev_page, state=tk.DISABLED)
        self.prev_page_btn.pack(side=tk.LEFT, padx=2, pady=4)
        self.next_page_btn = tk.Button(toolbar, text="Next Page", command=self.next_page, state=tk.DISABLED)
        self.next_page_btn.pack(side=tk.LEFT, padx=2, pady=4)
        tk.Button(toolbar, text="Zoom In", command=self._zoom_in).pack(side=tk.LEFT, padx=2, pady=4)
        tk.Button(toolbar, text="Zoom Out", command=self._zoom_out).pack(side=tk.LEFT, padx=2, pady=4)
        tk.Checkbutton(toolbar, text="Show Background", variable=self.background_visible).pack(side=tk.LEFT, padx=5, pady=4)
        tk.Checkbutton(toolbar, text="Show Overlays", variable=self.highlights_visible, command=self._refresh_display).pack(side=tk.LEFT, padx=2, pady=4)
        toolbar.pack(fill=tk.X)

    # ---------- Vector & Selection Logic ----------

    def _group_selected_vectors(self):
        if len(self.selected_line_indices) < 2:
            messagebox.showinfo("Info", "Select at least two lines to create a group.")
            return
        
        # Remove indices from any existing groups before re-grouping
        self._ungroup_selected_vectors(silent=True)
        
        new_group = set(self.selected_line_indices)
        self.vector_groups.append(new_group)
        self.status.set(f"Grouped {len(new_group)} lines.")
        self._refresh_display()

    def _ungroup_selected_vectors(self, silent=False):
        if not self.selected_line_indices:
            if not silent: messagebox.showinfo("Info", "No lines selected to ungroup.")
            return

        affected_groups_indices = []
        for i, group in enumerate(self.vector_groups):
            if self.selected_line_indices.intersection(group):
                affected_groups_indices.append(i)
        
        # Remove groups that were affected by the selection
        for i in sorted(affected_groups_indices, reverse=True):
            del self.vector_groups[i]

        if not silent:
            self.status.set("Ungrouped selected lines.")
            self._refresh_display()

    def _connect_selected_vectors(self):
        """Connects the nearest available endpoints of selected vectors and creates a fillable area."""
        if len(self.selected_line_indices) < 1:
            messagebox.showwarning("Selection Error", "Please select one or more lines to connect.")
            return

        selected_contours = [self.line_contours[i] for i in self.selected_line_indices]
        if not selected_contours: return

        endpoints = []
        for contour in selected_contours:
            endpoints.append(tuple(contour[0][0]))
            endpoints.append(tuple(contour[-1][0]))
        
        if len(endpoints) < 2: return

        kdtree = cKDTree(endpoints)
        new_segments = []
        used_indices = set()
        for i, p1 in enumerate(endpoints):
            if i in used_indices: continue
            distances, indices = kdtree.query(p1, k=len(endpoints))
            for j, dist in zip(indices, distances):
                if i == j or j in used_indices: continue
                p2 = endpoints[j]
                new_segments.append(np.array([[p1], [p2]], dtype=np.int32))
                used_indices.add(i)
                used_indices.add(j)
                break
        
        # --- Create fillable area from the new perimeter ---
        all_perimeter_segments = selected_contours + new_segments
        line_mask = np.zeros(self.base_image.size[::-1], dtype=np.uint8)
        cv2.drawContours(line_mask, all_perimeter_segments, -1, 255, 1)
        
        kernel = np.ones((3,3), np.uint8)
        closed_line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # --- Use the appropriate method based on the checkbox ---
        if self.create_with_holes.get():
            # Advanced method: Use hierarchy to create holes by inverting the mask
            inverted_closed_line_mask = cv2.bitwise_not(closed_line_mask)
            new_fill_contours, hierarchy = cv2.findContours(inverted_closed_line_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if not new_fill_contours:
                messagebox.showerror("Error", "Could not form a closed area from the selection.")
                return
            
            region_mask = np.zeros(self.base_image.size[::-1], dtype=np.uint8)
            # With an inverted mask, the first level (0) is the background.
            # Levels 1, 3, 5... are our primary shapes to fill.
            # Levels 2, 4, 6... are holes within those shapes.
            for i, contour in enumerate(new_fill_contours):
                level = 0
                parent_idx = hierarchy[0][i][3]
                while parent_idx != -1:
                    level += 1
                    parent_idx = hierarchy[0][parent_idx][3]
                
                if level % 2 == 1: # Fill odd levels
                    cv2.drawContours(region_mask, [contour], -1, 255, -1) # Fill
                elif level > 0 and level % 2 == 0: # Erase even, non-zero levels
                    cv2.drawContours(region_mask, [contour], -1, 0, -1) # Erase (hole)
        else:
            # Simple method: Fill the largest contour
            new_fill_contours, _ = cv2.findContours(closed_line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not new_fill_contours:
                messagebox.showerror("Error", "Could not form a closed area from the selection.")
                return
            largest_contour = max(new_fill_contours, key=cv2.contourArea)
            region_mask = np.zeros(self.base_image.size[::-1], dtype=np.uint8)
            cv2.drawContours(region_mask, [largest_contour], -1, 255, -1)

        new_label_id = self.segmentation.num_labels + 1
        self.segmentation.label_map[region_mask == 255] = new_label_id
        self.segmentation.num_labels += 1

        if self.active_color_group_id:
            active_group = next((g for g in self.color_groups if g['id'] == self.active_color_group_id), None)
            if active_group:
                active_group['regions'].add(new_label_id)
                self.status.set(f"Connected lines and created new Area {new_label_id}.")
                self._update_region_list()
            else:
                self.status.set(f"Connected lines, created Area {new_label_id}. No active group color applied.")
        else:
            self.status.set(f"Connected lines, created Area {new_label_id}. Select a color group to fill.")

        self.line_contours.extend(new_segments)
        self._redraw_all_overlays()
        self._refresh_display()

    def _on_canvas_drag_start(self, event):
        if self.mode.get() == 'define':
            self.drag_start_x = self.canvas.canvasx(event.x)
            self.drag_start_y = self.canvas.canvasy(event.y)
            self.drag_rect_id = self.canvas.create_rectangle(
                self.drag_start_x, self.drag_start_y, self.drag_start_x, self.drag_start_y,
                outline="blue", dash=(4, 4)
            )
        else: # Pass through to panning
            self._on_drag_start(event)

    def _on_canvas_drag_motion(self, event):
        if self.mode.get() == 'define' and self.drag_rect_id:
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            self.canvas.coords(self.drag_rect_id, self.drag_start_x, self.drag_start_y, x, y)
        else:
            self._on_drag_motion(event)

    def _on_canvas_drag_end(self, event):
        if self.mode.get() == 'define':
            if self.drag_rect_id:
                x1, y1, x2, y2 = self.canvas.coords(self.drag_rect_id)
                self.canvas.delete(self.drag_rect_id)
                self.drag_rect_id = None

                if abs(x1 - x2) < 5 and abs(y1 - y2) < 5:
                    self._handle_line_selection(event)
                    return
                
                rect_x1, rect_y1 = min(x1, x2) / self.zoom_level, min(y1, y2) / self.zoom_level
                rect_x2, rect_y2 = max(x1, x2) / self.zoom_level, max(y1, y2) / self.zoom_level
                
                newly_selected = set()
                for i, contour in enumerate(self.line_contours):
                    p1, p2 = contour[0][0], contour[-1][0]
                    if (rect_x1 < p1[0] < rect_x2 and rect_y1 < p1[1] < rect_y2) or \
                       (rect_x1 < p2[0] < rect_x2 and rect_y1 < p2[1] < rect_y2):
                        newly_selected.add(i)
                
                if not (event.state & 0x0004): # Not holding Ctrl
                    self.selected_line_indices.clear()
                self.selected_line_indices.update(newly_selected)
                self.status.set(f"{len(self.selected_line_indices)} lines selected.")
                self._refresh_display()
        else: # Fill mode
            self.on_fill_canvas_click(event)

    def _move_selected_vectors(self):
        try:
            dx = float(self.vector_move_x.get())
            dy = float(self.vector_move_y.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Move offsets must be numbers.")
            return
        if not self.selected_line_indices:
            messagebox.showinfo("Info", "No vector lines selected to move.")
            return
        
        translation_matrix = np.array([dx, dy], dtype=np.float32)
        for i in self.selected_line_indices:
            self.line_contours[i] = (self.line_contours[i].astype(np.float32) + translation_matrix).astype(np.int32)
        
        self._refresh_display()
        self.status.set(f"Moved {len(self.selected_line_indices)} lines. Apply changes to re-segment.")

    def _rotate_selected_vectors(self):
        try:
            angle_deg = float(self.vector_rotate_angle.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Rotation angle must be a number.")
            return
        if not self.selected_line_indices:
            messagebox.showinfo("Info", "No vector lines selected to rotate.")
            return
        
        selected_contours = [self.line_contours[i] for i in self.selected_line_indices]
        if not selected_contours: return

        all_points = np.vstack(selected_contours)
        x, y, w, h = cv2.boundingRect(all_points)
        center_x, center_y = x + w / 2, y + h / 2
        rot_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle_deg, 1.0)

        for i in self.selected_line_indices:
            contour = self.line_contours[i]
            points = contour.reshape(-1, 2)
            points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
            transformed_points = points_homogeneous.dot(rot_matrix.T)
            self.line_contours[i] = transformed_points.reshape(-1, 1, 2).astype(np.int32)
        
        self._refresh_display()
        self.status.set(f"Rotated {len(self.selected_line_indices)} lines. Apply changes to re-segment.")

    def _delete_selected_vectors(self):
        if not self.selected_line_indices:
            messagebox.showinfo("Info", "No vector lines selected to delete.")
            return
        
        self.line_contours = [contour for i, contour in enumerate(self.line_contours) if i not in self.selected_line_indices]
        self._remap_indices_after_deletion(self.selected_line_indices)
        self.selected_line_indices.clear()
        self.status.set("Deleted selected lines. Consider re-segmenting.")
        self._refresh_display()

    def _remap_indices_after_deletion(self, deleted_indices: set):
        """Adjusts all stored indices after some lines have been removed."""
        deleted_sorted = sorted(list(deleted_indices), reverse=True)
        
        new_groups = []
        for group in self.vector_groups:
            new_group = set()
            for idx in group:
                shift = sum(1 for del_idx in deleted_sorted if del_idx < idx)
                new_group.add(idx - shift)
            new_groups.append(new_group)
        self.vector_groups = new_groups

    def _sync_vector_changes(self):
        messagebox.showinfo("Info", "To apply vector changes, please use the segmentation controls (e.g., 'Reset to Auto') to re-process the page.")

    # ---------- PDF & Image Loading / Segmentation ----------
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

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        if not path: return
        try:
            image = Image.open(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{e}")
            return
        
        self.pdf_doc = None
        self.page_count = 1
        self.page_index = 0
        self.status.set(f"Loaded: {os.path.basename(path)}")
        self._process_image_data(image)

    def _process_image_data(self, image: Image.Image, threshold: Optional[int] = None):
        """Common logic to process a PIL image for segmentation and display."""
        self.base_image = image.convert("RGB")
        self.color_groups.clear()
        self.active_color_group_id = None
        self._update_region_list()
        self.status.set("Segmenting regions and detecting lines...")
        self.update_idletasks()

        self.segmentation, used_threshold = segment_closed_regions(
            self.base_image, self.dilation_radius, self.invert_lines, threshold
        )
        self.line_contours, _ = detect_lines(self.base_image)
        self.selected_line_indices.clear()
        self.vector_groups.clear()
        self.manual_threshold.set(used_threshold)

        mode = "(Auto)" if threshold is None else "(Manual)"
        self.threshold_label.config(text=f"Threshold: {used_threshold} {mode}")
        
        num_regions = self.segmentation.num_labels
        if self.pdf_doc:
            status_text = f"Page {self.page_index + 1}/{self.page_count} | Regions: {num_regions} | Threshold: {used_threshold} {mode}"
        else:
            status_text = f"Image Loaded | Regions: {num_regions} | Threshold: {used_threshold} {mode}"
        
        self.status.set(status_text)
        self._update_page_nav_buttons()
        self._redraw_all_overlays()
        self._refresh_display()

    def _load_page(self, index: int, threshold: Optional[int] = None):
        if not self.pdf_doc or not (0 <= index < self.pdf_doc.page_count): return
        self.page_index = index
        page = self.pdf_doc.load_page(index)
        mat = fitz.Matrix(2.0, 2.0)  # Render at high resolution
        pix = page.get_pixmap(matrix=mat, alpha=False)
        image = pil_from_pixmap(pix)
        self._process_image_data(image, threshold)

    def _resegment_page(self, manual: bool):
        if self.base_image is None: return
        if not messagebox.askyesno("Confirm Resegment", "This will clear all colored regions and vector edits on the current page. Proceed?"):
            return
        
        threshold = self.manual_threshold.get() if manual else None
        
        if self.pdf_doc:
            self._load_page(self.page_index, threshold=threshold)
        else:
            # For a plain image, we just re-process it
            self._process_image_data(self.base_image, threshold)

    def _update_page_nav_buttons(self):
        prev_state = tk.NORMAL if self.page_index > 0 else tk.DISABLED
        next_state = tk.NORMAL if self.page_index < self.page_count - 1 else tk.DISABLED
        
        if hasattr(self, 'prev_page_btn'):
            self.prev_page_btn.config(state=prev_state)
        if hasattr(self, 'next_page_btn'):
            self.next_page_btn.config(state=next_state)

    def prev_page(self):
        if self.pdf_doc and self.page_index > 0:
            self._load_page(self.page_index - 1)

    def next_page(self):
        if self.pdf_doc and self.page_index < self.page_count - 1:
            self._load_page(self.page_index + 1)

    # ---------- Display handling ----------
    def _refresh_display(self, *_):
        if self.base_image is None: return

        if self.background_visible.get():
            base_rgba = self.base_image.convert("RGBA")
        else:
            # Create a black background when the original image is hidden
            base_rgba = Image.new("RGBA", self.base_image.size, (0, 0, 0, 255))

        comp = alpha_composite(base_rgba, self.overlay_image)
        if self.highlights_visible.get():
            draw_lines_on_image(comp, self.line_contours, self.selected_line_indices, self.vector_groups, self.vector_colors)
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
        event.widget.scan_mark(event.x, event.y)

    def _on_drag_motion(self, event):
        event.widget.scan_dragto(event.x, event.y, gain=1)

    # ---------- Mode-Specific Logic ----------
    def _on_mode_change(self):
        mode = self.mode.get()
        self._clear_line_selection()
        if mode == 'fill':
            self.canvas.config(cursor="tcross")
            self.status.set("Mode: Fill Area. Click inside a region to color it.")
            self.define_tools_frame.pack_forget()
        elif mode == 'define':
            self.canvas.config(cursor="hand2")
            self.status.set("Mode: Edit Vectors. Click or drag to select lines.")
            self.define_tools_frame.pack(fill=tk.BOTH, expand=True)
        self._refresh_display()

    def _clear_line_selection(self):
        if not self.selected_line_indices: return
        self.selected_line_indices.clear()
        self.status.set("Line selection cleared.")
        self._refresh_display()

    def _handle_line_selection(self, event):
        canvas_x, canvas_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        x, y = int(canvas_x / self.zoom_level), int(canvas_y / self.zoom_level)
        click_point = (x, y)
        
        min_dist = float('inf')
        best_idx = -1
        for i, contour in enumerate(self.line_contours):
            p1 = np.array(contour[0][0])
            p2 = np.array(contour[-1][0])
            l2 = np.sum((p1 - p2)**2)
            if l2 == 0:
                dist = np.linalg.norm(np.array(click_point) - p1)
            else:
                t = max(0, min(1, np.dot(np.array(click_point) - p1, p2 - p1) / l2))
                projection = p1 + t * (p2 - p1)
                dist = np.linalg.norm(np.array(click_point) - projection)
            
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        
        if best_idx != -1 and min_dist < 10 / self.zoom_level:
            clicked_group = None
            for group in self.vector_groups:
                if best_idx in group:
                    clicked_group = group
                    break
            
            if event.state & 0x0004: # Ctrl key
                target_indices = clicked_group if clicked_group else {best_idx}
                if target_indices.issubset(self.selected_line_indices):
                    self.selected_line_indices.difference_update(target_indices)
                else:
                    self.selected_line_indices.update(target_indices)
            else: # New selection
                self.selected_line_indices.clear()
                self.selected_line_indices.update(clicked_group if clicked_group else {best_idx})
        else: 
            if not (event.state & 0x0004):
                self.selected_line_indices.clear()

        self.status.set(f"{len(self.selected_line_indices)} lines selected.")
        self._refresh_display()

    def on_fill_canvas_click(self, event):
        if self.base_image is None or self.segmentation is None: return
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
        for group in self.color_groups:
            if label in group['regions']: group['regions'].remove(label)
        active_group['regions'].add(label)
        self._redraw_all_overlays()
        self._refresh_display()
        self._update_region_list()
        self.status.set(f"Added Region {label} to group '{active_group['name']}'.")

    # ---------- Customization and UI Methods ----------
    def _set_vector_color(self):
        # This allows changing the default, selected, and grouped colors sequentially.
        original_colors = self.vector_colors.copy()
        for key in ["default", "selected", "grouped"]:
            prompt = f"Pick a color for '{key}' vector lines"
            initial_color = self.vector_colors[key]
            color_info = colorchooser.askcolor(title=prompt, initialcolor=initial_color)
            if not color_info or not color_info[0]:
                messagebox.showinfo("Info", f"Vector color setting for '{key}' was cancelled. No colors were changed.")
                self.vector_colors = original_colors # Revert all changes if any step is cancelled
                return
            r, g, b = color_info[0]
            self.vector_colors[key] = (int(r), int(g), int(b), 255)
        self._refresh_display()

    def _set_overlay_alpha(self):
        alpha = simpledialog.askinteger("Default Opacity", "Enter default opacity for new fills (0-255):", initialvalue=self.overlay_alpha.get(), minvalue=0, maxvalue=255, parent=self)
        if alpha is not None:
            self.overlay_alpha.set(alpha)
            messagebox.showinfo("Info", f"Default fill opacity set to {alpha}. This will apply to new groups.")

    def _change_group_color(self):
        if not self.region_tree: return
        selection = self.region_tree.selection()
        if not selection or not selection[0].startswith("group_"):
            messagebox.showwarning("Warning", "Please select exactly one group folder to change its color.")
            return
        
        group_id = selection[0]
        group = next((g for g in self.color_groups if g['id'] == group_id), None)
        if not group: return

        initial_color = group['color'][:3]
        initial_alpha = group['color'][3]

        color_info = colorchooser.askcolor(title=f"Pick new color for '{group['name']}'", initialcolor=initial_color)
        if not color_info or not color_info[0]: return
        r, g, b = color_info[0]

        alpha = simpledialog.askinteger("Overlay Opacity", "Enter opacity (0-255):", initialvalue=initial_alpha, minvalue=0, maxvalue=255, parent=self)
        if alpha is None: alpha = initial_alpha

        group['color'] = (int(r), int(g), int(b), alpha)
        
        self._update_region_list()
        self._redraw_all_overlays()
        self._refresh_display()

    # ---------- Core Application Logic ----------
    def _create_area_from_selection(self):
        if len(self.selected_line_indices) < 1:
            messagebox.showwarning("Selection Error", "Please select at least one line segment.")
            return
        if self.base_image is None or self.segmentation is None: return
        max_dist = simpledialog.askinteger("Connect Gaps", "Enter max distance (pixels) to connect gaps:", initialvalue=20, minvalue=1, maxvalue=200, parent=self)
        if max_dist is None: return
        self.status.set("Connecting endpoints and creating area...")
        self.update_idletasks()
        line_mask = np.zeros(self.base_image.size[::-1], dtype=np.uint8)
        selected_contours = [self.line_contours[i] for i in self.selected_line_indices]
        cv2.drawContours(line_mask, selected_contours, -1, 255, 1)
        endpoints = set()
        for contour in selected_contours:
            if len(contour) > 0:
                endpoints.add(tuple(contour[0][0]))
                if len(contour) > 1:
                    endpoints.add(tuple(contour[-1][0]))
        endpoint_list = list(endpoints)
        if len(endpoint_list) >= 2:
            kdtree = cKDTree(endpoint_list)
            close_pairs = kdtree.query_pairs(r=max_dist)
            if close_pairs:
                for (i, j) in close_pairs:
                    p1, p2 = endpoint_list[i], endpoint_list[j]
                    cv2.line(line_mask, p1, p2, 255, 1)
        
        kernel = np.ones((3,3), np.uint8)
        closed_line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        if self.create_with_holes.get():
            inverted_closed_line_mask = cv2.bitwise_not(closed_line_mask)
            new_contours, hierarchy = cv2.findContours(inverted_closed_line_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if not new_contours:
                messagebox.showerror("Error", "Could not form a closed area from the selection.")
                return
            region_mask = np.zeros(self.base_image.size[::-1], dtype=np.uint8)
            # With an inverted mask, the first level (0) is the background.
            # Levels 1, 3, 5... are our primary shapes to fill.
            # Levels 2, 4, 6... are holes within those shapes.
            for i, contour in enumerate(new_contours):
                level = 0
                parent_idx = hierarchy[0][i][3]
                while parent_idx != -1:
                    level += 1
                    parent_idx = hierarchy[0][parent_idx][3]
                
                if level % 2 == 1:  # Fill odd levels (shapes)
                    cv2.drawContours(region_mask, [contour], -1, 255, -1)
                elif level > 0 and level % 2 == 0:  # Erase even levels (holes)
                    cv2.drawContours(region_mask, [contour], -1, 0, -1)
        else:
            new_contours, _ = cv2.findContours(closed_line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not new_contours:
                messagebox.showerror("Error", "Could not form a closed area from the selection.")
                return
            largest_contour = max(new_contours, key=cv2.contourArea)
            region_mask = np.zeros(self.base_image.size[::-1], dtype=np.uint8)
            cv2.drawContours(region_mask, [largest_contour], -1, 255, -1)

        new_label_id = self.segmentation.num_labels + 1
        self.segmentation.label_map[region_mask == 255] = new_label_id
        self.segmentation.num_labels += 1
        if not self.active_color_group_id:
            messagebox.showwarning("No Active Group", "Area created, but no color group was active. Please select a group and click on the new area to color it.")
        else:
            active_group = next((g for g in self.color_groups if g['id'] == self.active_color_group_id), None)
            if active_group:
                active_group['regions'].add(new_label_id)
        
        self.selected_line_indices.clear()
        self.status.set(f"Successfully created new Area {new_label_id}.")
        self._update_region_list()
        self._redraw_all_overlays()
        self._refresh_display()
    def _add_new_group(self):
        name = tk.simpledialog.askstring("New Color Group", "Enter a name for the new group:", parent=self)
        if not name: return
        color_info = colorchooser.askcolor(title=f"Pick color for '{name}'")
        if not color_info or not color_info[0]: return
        r, g, b = color_info[0]
        rgba = (int(r), int(g), int(b), self.overlay_alpha.get())
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
        dilation_amount = simpledialog.askinteger("Merge Strength", "Enter connection strength (higher for distant regions):", initialvalue=self.dilation_radius + 3, minvalue=1, maxvalue=100, parent=self)
        if dilation_amount is None: return
        label_map = self.segmentation.label_map
        region_ids = sorted([int(s.split("_")[1]) for s in region_ids_str])
        labels_to_merge = set(region_ids)
        all_labels_on_page = set(np.unique(label_map)) - {0}
        protected_labels = all_labels_on_page - labels_to_merge
        protected_mask = np.zeros_like(label_map, dtype=bool)
        for label_id in protected_labels:
            protected_mask[label_map == label_id] = True
        target_label, *source_labels = region_ids
        self.status.set(f"Merging {len(source_labels) + 1} regions...")
        self.update_idletasks()
        for source_label in source_labels:
            mask_target = (label_map == target_label)
            mask_source = (label_map == source_label)
            dilated_target = ndi.binary_dilation(mask_target, iterations=dilation_amount)
            dilated_source = ndi.binary_dilation(mask_source, iterations=dilation_amount)
            bridge_mask = dilated_target & dilated_source
            safe_bridge_mask = bridge_mask & ~protected_mask
            label_map[mask_source] = target_label
            label_map[safe_bridge_mask] = target_label
            if source_label in group["regions"]:
                group["regions"].remove(source_label)
        self.status.set(f"Merge complete. New region is Area {target_label}.")
        self._redraw_all_overlays()
        self._refresh_display()
        self._update_region_list()
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
            except tk.TclError: pass
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
    def _apply_overlay_for_region(self, region_id: int, rgba: Tuple[int, int, int, int]):
        if self.segmentation is None or self.overlay_image is None: return
        mask_bool = numpy_mask_from_label(self.segmentation.label_map, region_id)
        mask_L = pil_mask_from_bool(mask_bool)
        color_img = Image.new("RGBA", self.overlay_image.size, rgba)
        self.overlay_image.paste(color_img, (0, 0), mask_L)
        draw_region_label(self.overlay_image, region_id, mask_bool, rgba)
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
            self._resegment_page(manual=False)
        except (ValueError, TypeError):
            messagebox.showerror("Error", "Invalid input. Please enter an integer between 0 and 10.")
    def toggle_invert_lines(self):
        self.invert_lines = not self.invert_lines
        status_msg = "ON" if self.invert_lines else "OFF"
        self.status.set(f"Invert lines is now {status_msg}.")
        if self.base_image is not None:
            self._resegment_page(manual=False)
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
        path = filedialog.asksaveasfilename(title="Save Selected Regions as PNG",defaultextension=".png",filetypes=[("PNG image", "*.png")],initialfile=f"page{self.page_index+1:02d}_selected.png")
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
        path = filedialog.asksaveasfilename(title="Save composited PNG",defaultextension=".png",filetypes=[("PNG image", "*.png")],initialfile=f"page{self.page_index+1:02d}_composite.png")
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
        path = filedialog.asksaveasfilename(title="Save region as PDF",defaultextension=".pdf",filetypes=[("PDF", "*.pdf")],initialfile=f"page{self.page_index+1:02d}_region_{region_id}.pdf")
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

if __name__ == "__main__":
    app = ColorFillPDFApp()
    app.mainloop()
