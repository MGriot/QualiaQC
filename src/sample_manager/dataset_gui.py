import tkinter as tk
from tkinter import messagebox
import os
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk

class DatasetManagerGUI:
    def __init__(self, master, image_paths: list, config_file_path: str):
        self.master = master
        self.image_paths = image_paths
        self.config_file_path = Path(config_file_path)
        self.current_image_index = 0
        self.image_points = {}
        self.current_cv_image = None
        self.current_photo = None

        self.master.title("Dataset Manager")
        self.master.grab_set()

        # --- Zoom/Pan State ---
        self.zoom_level = 1.0
        self.max_canvas_width = 1200
        self.max_canvas_height = 800

        # --- Mode and Area Selection State ---
        self.mode = tk.StringVar(value="points")
        self.rect = None
        self.start_x = None
        self.start_y = None

        self.num_clusters_var = tk.StringVar(value="10")

        self.load_config()
        self._build_ui()
        self.load_image()

    def _build_ui(self):
        # --- Top frame for canvas ---
        canvas_container = tk.Frame(self.master, relief=tk.SUNKEN, borderwidth=1)
        canvas_container.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_container, cursor="cross", bg="#333")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # --- Navigation and Zoom frame ---
        controls_frame = tk.Frame(self.master)
        controls_frame.pack(pady=5, padx=10, fill=tk.X)
        
        self.prev_button = tk.Button(controls_frame, text="<< Previous", command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        self.image_label = tk.Label(controls_frame, text="")
        self.image_label.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.next_button = tk.Button(controls_frame, text="Next >>", command=self.next_image)
        self.next_button.pack(side=tk.LEFT, padx=5)

        zoom_frame = tk.Frame(controls_frame)
        zoom_frame.pack(side=tk.RIGHT)
        tk.Button(zoom_frame, text="Zoom In", command=lambda: self._zoom(1.25)).pack(side=tk.LEFT)
        tk.Button(zoom_frame, text="Zoom Out", command=lambda: self._zoom(0.8)).pack(side=tk.LEFT, padx=5)

        # --- Mode selection frame ---
        mode_frame = tk.LabelFrame(self.master, text="Sampling Mode")
        mode_frame.pack(pady=5, padx=10, fill=tk.X)
        tk.Radiobutton(mode_frame, text="Add Points", variable=self.mode, value="points", command=self._update_bindings).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(mode_frame, text="Delete Point", variable=self.mode, value="delete", command=self._update_bindings).pack(side=tk.LEFT, padx=10)
        
        cluster_frame = tk.Frame(mode_frame)
        cluster_frame.pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(cluster_frame, text="Select Area (Cluster)", variable=self.mode, value="area", command=self._update_bindings).pack(side=tk.LEFT)
        tk.Label(cluster_frame, text="Num Clusters:").pack(side=tk.LEFT, padx=(5,0))
        self.num_clusters_entry = tk.Entry(cluster_frame, textvariable=self.num_clusters_var, width=5)
        self.num_clusters_entry.pack(side=tk.LEFT)

        # --- Action frame ---
        self.action_frame = tk.Frame(self.master)
        self.action_frame.pack(pady=10)
        self.clear_button = tk.Button(self.action_frame, text="Clear Points for this Image", command=self.clear_points)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        self.save_button = tk.Button(self.action_frame, text="Save and Close", command=self.save_and_close)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self._update_bindings()

    def _update_bindings(self):
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")

        if self.mode.get() == "points":
            self.canvas.config(cursor="cross")
            self.canvas.bind("<Button-1>", self._handle_point_click)
        elif self.mode.get() == "area":
            self.canvas.config(cursor="tcross")
            self.canvas.bind("<ButtonPress-1>", self._on_area_press)
            self.canvas.bind("<B1-Motion>", self._on_area_drag)
            self.canvas.bind("<ButtonRelease-1>", self._on_area_release)
        elif self.mode.get() == "delete":
            self.canvas.config(cursor="pirate")
            self.canvas.bind("<Button-1>", self._handle_delete_click)

        # Always have zoom and pan enabled
        self.canvas.bind("<Control-MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<ButtonPress-2>", self._on_drag_start)
        self.canvas.bind("<B2-Motion>", self._on_drag_motion)

    def load_config(self):
        if self.config_file_path.exists():
            try:
                with open(self.config_file_path, 'r') as f:
                    config_data = json.load(f)
                for img_config in config_data.get("image_configs", []):
                    if img_config.get("method") == "points":
                        self.image_points[img_config["filename"]] = img_config.get("points", [])
            except json.JSONDecodeError:
                messagebox.showwarning("Config Error", f"Warning: Could not parse {self.config_file_path}. It may be corrupt. A new file will be created on save.")

    def load_image(self):
        image_path = self.image_paths[self.current_image_index]
        image_name = os.path.basename(image_path)
        self.image_label.config(text=f"Image {self.current_image_index + 1} / {len(self.image_paths)}: {image_name}")

        self.current_cv_image = cv2.imread(str(image_path))
        if self.current_cv_image is None:
            messagebox.showerror("Error", f"Could not load image: {image_name}")
            return

        # Calculate initial zoom to fit image in window
        h, w, _ = self.current_cv_image.shape
        scale_w = self.max_canvas_width / w
        scale_h = self.max_canvas_height / h
        self.zoom_level = min(scale_w, scale_h, 1.0)

        self.redraw_canvas()

    def redraw_canvas(self):
        if self.current_cv_image is None: return

        self.canvas.delete("all")

        h, w, _ = self.current_cv_image.shape
        scaled_w, scaled_h = int(w * self.zoom_level), int(h * self.zoom_level)

        # Use PIL for resizing to integrate with Tkinter
        image_rgb = cv2.cvtColor(self.current_cv_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        display_img = pil_img.resize((scaled_w, scaled_h), Image.Resampling.BILINEAR)
        
        self.current_photo = ImageTk.PhotoImage(image=display_img)

        self.canvas.config(scrollregion=(0, 0, scaled_w, scaled_h))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_photo)
        self.draw_points()

    def _zoom(self, factor):
        self.zoom_level *= factor
        self.redraw_canvas()

    def _on_mouse_wheel(self, event):
        factor = 1.1 if (event.delta > 0 or event.num == 4) else 0.9
        self._zoom(factor)
        return "break"

    def _on_drag_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def _on_drag_motion(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _handle_point_click(self, event):
        image_x = int(self.canvas.canvasx(event.x) / self.zoom_level)
        image_y = int(self.canvas.canvasy(event.y) / self.zoom_level)

        image_name = os.path.basename(self.image_paths[self.current_image_index])
        if image_name not in self.image_points:
            self.image_points[image_name] = []
        
        self.image_points[image_name].append({"x": image_x, "y": image_y, "radius": 7})
        self.draw_points()

    def _handle_delete_click(self, event):
        image_x = int(self.canvas.canvasx(event.x) / self.zoom_level)
        image_y = int(self.canvas.canvasy(event.y) / self.zoom_level)
        image_name = os.path.basename(self.image_paths[self.current_image_index])
        
        points = self.image_points.get(image_name, [])
        if not points:
            return

        # Find the closest point to the click
        distances = [np.sqrt((p['x'] - image_x)**2 + (p['y'] - image_y)**2) for p in points]
        closest_index = np.argmin(distances)

        # If the click is within the radius of the point, delete it
        if distances[closest_index] <= points[closest_index].get('radius', 7):
            points.pop(closest_index)
            self.redraw_canvas()

    def _on_area_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', dash=(2,2))

    def _on_area_drag(self, event):
        cur_x, cur_y = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def _on_area_release(self, event):
        if not self.rect: return
        
        # Convert canvas coordinates to image coordinates
        x1_img = int(min(self.start_x, self.canvas.canvasx(event.x)) / self.zoom_level)
        y1_img = int(min(self.start_y, self.canvas.canvasy(event.y)) / self.zoom_level)
        x2_img = int(max(self.start_x, self.canvas.canvasx(event.x)) / self.zoom_level)
        y2_img = int(max(self.start_y, self.canvas.canvasy(event.y)) / self.zoom_level)

        self.canvas.delete(self.rect)
        self.rect = None
        
        if x2_img - x1_img > 5 and y2_img - y1_img > 5:
            self._perform_clustering(x1_img, y1_img, x2_img, y2_img)

    def _perform_clustering(self, x1, y1, x2, y2):
        try:
            K = int(self.num_clusters_var.get())
            if K <= 0:
                messagebox.showerror("Error", "Number of clusters must be a positive integer.")
                return
        except ValueError:
            messagebox.showerror("Error", "Invalid number of clusters. Please enter an integer.")
            return

        roi = self.current_cv_image[y1:y2, x1:x2]
        if roi.size == 0:
            return

        pixels = roi.reshape(-1, 3).astype(np.float32)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        new_points = []
        for center_color in centers:
            distances = np.linalg.norm(pixels - center_color, axis=1)
            closest_pixel_index = np.argmin(distances)
            y_in_roi, x_in_roi = np.unravel_index(closest_pixel_index, (roi.shape[0], roi.shape[1]))
            point = {"x": int(x_in_roi + x1), "y": int(y_in_roi + y1), "radius": 7}
            new_points.append(point)

        image_name = os.path.basename(self.image_paths[self.current_image_index])
        if image_name not in self.image_points:
            self.image_points[image_name] = []
        self.image_points[image_name].extend(new_points)
        self.draw_points()
        messagebox.showinfo("Clustering Complete", f"Added {len(new_points)} points from the selected area.")

    def draw_points(self):
        self.canvas.delete("point")
        image_name = os.path.basename(self.image_paths[self.current_image_index])
        points = self.image_points.get(image_name, [])
        for i, point in enumerate(points):
            x, y, r = point['x'], point['y'], point.get('radius', 7)
            # Scale point coordinates to canvas coordinates
            x_canvas, y_canvas = x * self.zoom_level, y * self.zoom_level
            # Use a unique tag for each point to be able to identify it
            tag = f"point_{i}"
            self.canvas.create_oval(x_canvas - r, y_canvas - r, x_canvas + r, y_canvas + r, fill="red", outline="red", tags=("point", tag))

    def clear_points(self):
        image_name = os.path.basename(self.image_paths[self.current_image_index])
        if image_name in self.image_points:
            self.image_points[image_name] = []
            self.draw_points()

    def next_image(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.load_image()

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image()

    def save_and_close(self):
        image_configs = []
        current_image_names = {os.path.basename(p) for p in self.image_paths}

        if self.config_file_path.exists():
            try:
                with open(self.config_file_path, 'r') as f:
                    existing_config_data = json.load(f)
                for cfg in existing_config_data.get("image_configs", []):
                    if cfg["filename"] not in current_image_names:
                        image_configs.append(cfg)
            except json.JSONDecodeError:
                pass

        for image_path in self.image_paths:
            image_name = os.path.basename(image_path)
            points = self.image_points.get(image_name)

            if points is not None:
                image_configs.append({
                    "filename": image_name,
                    "method": "points",
                    "points": points
                })

        config_data = {"image_configs": image_configs}
        
        temp_path = self.config_file_path.with_suffix('.json.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(config_data, f, indent=4)
            os.replace(temp_path, self.config_file_path)
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save configuration file: {e}")
            if temp_path.exists():
                os.remove(temp_path)
            return
        
        messagebox.showinfo("Success", f"Configuration saved to {self.config_file_path}")
        self.master.destroy()
