import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import argparse
import os
import json
from pathlib import Path
import cv2
from PIL import Image, ImageTk
import pickle
from datetime import datetime

from src.pipeline import run_analysis, Pipeline
from src.project_manager import ProjectManager
from src.create_project import create_project
from src.reporting.generator import ReportGenerator
from src.sample_manager.dataset_gui import DatasetManagerGUI
from src.sample_manager.file_placer_gui import ProjectFilePlacerGUI
from src.config import AUTHOR, DEPARTMENT, REPORT_TITLE


class VisualAnalyzerGUI(tk.Tk):
    def __init__(self, debug_mode=False):
        super().__init__()
        self.debug_mode = debug_mode
        self.title("Visual Analyzer")
        self.geometry("800x850")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(pady=10, padx=10, fill="both", expand=True)

        self.analysis_tab = ttk.Frame(self.notebook)
        self.create_project_tab = ttk.Frame(self.notebook)
        self.manage_dataset_tab = ttk.Frame(self.notebook)
        self.history_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.analysis_tab, text="Run Analysis")
        self.notebook.add(self.history_tab, text="History & Reports")
        self.notebook.add(self.create_project_tab, text="Create Project")
        self.notebook.add(self.manage_dataset_tab, text="Manage Dataset")

        self.project_manager = ProjectManager()
        self.setup_variables()
        self.populate_analysis_tab()
        self.populate_create_project_tab()
        self.populate_manage_dataset_tab()
        self.populate_history_tab()

    def setup_variables(self):
        self.available_projects = self.project_manager.list_projects()
        self.project_var = tk.StringVar()
        self.image_path_var = tk.StringVar()
        self.part_number_var = tk.StringVar()
        self.thickness_var = tk.StringVar()
        self.author_var = tk.StringVar(value=AUTHOR)
        self.department_var = tk.StringVar(value=DEPARTMENT)
        self.report_title_var = tk.StringVar(value=REPORT_TITLE)
        self.color_checker_path_var = tk.StringVar()
        self.color_correction_method_var = tk.StringVar(value="linear")
        self.masking_order_var = tk.StringVar(value="1-2-3")
        self.blur_kernel_var = tk.StringVar(value="5 5")
        self.agg_kernel_size_var = tk.StringVar(value="7")
        self.agg_min_area_var = tk.StringVar(value="0.0005")
        self.agg_density_thresh_var = tk.StringVar(value="0.5")
        self.color_alignment_var = tk.BooleanVar(value=True)
        self.symmetry_var = tk.BooleanVar(value=True)
        self.aggregate_var = tk.BooleanVar(value=True)
        self.debug_var = tk.BooleanVar()
        self.alignment_var = tk.BooleanVar(value=True)
        self.object_alignment_var = tk.BooleanVar(value=True)
        self.object_alignment_shadow_removal_var = tk.BooleanVar(value=False)
        self.apply_mask_var = tk.BooleanVar(value=True)
        self.mask_bg_is_white_var = tk.BooleanVar(value=False)
        self.blur_var = tk.BooleanVar(value=True)
        self.new_project_name_var = tk.StringVar()
        self.manage_project_var = tk.StringVar()
        # History tab variables
        self.history_data = []
        self.history_filter_vars = {
            'date': tk.StringVar(),
            'project': tk.StringVar(),
            'part_number': tk.StringVar(),
            'thickness': tk.StringVar(),
            'percentage': tk.StringVar()
        }

    def populate_analysis_tab(self):
        main_frame = self.analysis_tab
        project_frame = tk.LabelFrame(main_frame, text="Project")
        project_frame.pack(fill=tk.X, pady=5, padx=5)
        tk.Label(project_frame, text="Project Name:").pack(side=tk.LEFT, padx=5)
        self.project_combobox = ttk.Combobox(
            project_frame, textvariable=self.project_var, values=self.available_projects
        )
        self.project_combobox.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        if self.available_projects:
            self.project_combobox.set(self.available_projects[0])

        file_frame = tk.LabelFrame(main_frame, text="Files & Metadata")
        file_frame.pack(fill=tk.X, pady=5, padx=5)
        tk.Button(file_frame, text="Select Image", command=self.select_image).pack(
            fill=tk.X, pady=2
        )
        tk.Label(
            file_frame, textvariable=self.image_path_var, wraplength=500, anchor=tk.W
        ).pack(fill=tk.X, padx=5)

        metadata_frame = ttk.Frame(file_frame)
        metadata_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
        metadata_frame.columnconfigure(1, weight=1)
        metadata_frame.columnconfigure(3, weight=1)

        tk.Label(metadata_frame, text="Part Number:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        tk.Entry(metadata_frame, textvariable=self.part_number_var).grid(row=0, column=1, sticky=tk.EW)

        tk.Label(metadata_frame, text="Thickness:").grid(row=0, column=2, sticky=tk.W, padx=(10, 5))
        tk.Entry(metadata_frame, textvariable=self.thickness_var).grid(row=0, column=3, sticky=tk.EW)

        tk.Label(metadata_frame, text="Author:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5,0))
        tk.Entry(metadata_frame, textvariable=self.author_var).grid(row=1, column=1, columnspan=3, sticky=tk.EW, pady=(5,0))

        tk.Label(metadata_frame, text="Department:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5), pady=(5,0))
        tk.Entry(metadata_frame, textvariable=self.department_var).grid(row=2, column=1, columnspan=3, sticky=tk.EW, pady=(5,0))

        tk.Label(metadata_frame, text="Report Title:").grid(row=3, column=0, sticky=tk.W, padx=(0, 5), pady=(5,0))
        report_title_entry = tk.Entry(metadata_frame, textvariable=self.report_title_var)
        report_title_entry.grid(row=3, column=1, columnspan=3, sticky=tk.EW, pady=(5,0))
        if not self.debug_mode:
            report_title_entry.config(state=tk.DISABLED)

        tk.Button(
            file_frame, text="Select Color Checker", command=self.select_color_checker
        ).pack(fill=tk.X, pady=2)
        tk.Label(
            file_frame,
            textvariable=self.color_checker_path_var,
            wraplength=500,
            anchor=tk.W,
        ).pack(fill=tk.X, padx=5)

        if self.debug_mode:
            options_frame = tk.LabelFrame(main_frame, text="Analysis Steps & Options")
            options_frame.pack(fill=tk.X, pady=5, padx=5)
            grid_frame = ttk.Frame(options_frame)
            grid_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
            grid_frame.columnconfigure(0, weight=1)
            grid_frame.columnconfigure(1, weight=1)
            bool_frame = ttk.Frame(grid_frame)
            bool_frame.grid(row=0, column=0, sticky="new")
            tk.Checkbutton(bool_frame, text="Color Alignment", variable=self.color_alignment_var).pack(anchor=tk.W)
            cc_sub_frame = ttk.Frame(bool_frame)
            cc_sub_frame.pack(anchor=tk.W, fill=tk.X, padx=(20, 0))
            tk.Label(cc_sub_frame, text="Method:").pack(side=tk.LEFT)
            ttk.Combobox(cc_sub_frame, textvariable=self.color_correction_method_var, values=["linear", "polynomial", "hsv", "histogram"], width=15).pack(side=tk.LEFT, padx=5)
            tk.Checkbutton(bool_frame, text="Geometrical Alignment", variable=self.alignment_var).pack(anchor=tk.W)
            tk.Checkbutton(bool_frame, text="Object Alignment", variable=self.object_alignment_var).pack(anchor=tk.W)
            object_alignment_sub_frame = ttk.Frame(bool_frame)
            object_alignment_sub_frame.pack(anchor=tk.W, fill=tk.X, padx=(20, 0))
            tk.Checkbutton(object_alignment_sub_frame, text="Shadow Removal", variable=self.object_alignment_shadow_removal_var).pack(anchor=tk.W)
            tk.Checkbutton(bool_frame, text="Apply Mask", variable=self.apply_mask_var).pack(anchor=tk.W)
            mask_sub_frame = ttk.Frame(bool_frame)
            mask_sub_frame.pack(anchor=tk.W, fill=tk.X, padx=(20, 0))
            tk.Checkbutton(mask_sub_frame, text="Mask BG is White", variable=self.mask_bg_is_white_var).pack(anchor=tk.W)
            tk.Checkbutton(bool_frame, text="Symmetry Analysis", variable=self.symmetry_var).pack(anchor=tk.W)
            param_frame = ttk.Frame(grid_frame)
            param_frame.grid(row=0, column=1, sticky="new")
            tk.Label(param_frame, text="Masking Order (e.g., 1-2-3):").pack(anchor=tk.W, pady=(5, 0))
            tk.Entry(param_frame, textvariable=self.masking_order_var).pack(anchor=tk.W, fill=tk.X)
            blur_frame = ttk.Frame(param_frame)
            blur_frame.pack(fill=tk.X, pady=(5, 0))
            tk.Checkbutton(blur_frame, text="Blur", variable=self.blur_var).pack(anchor=tk.W)
            blur_options_frame = ttk.Frame(blur_frame)
            blur_options_frame.pack(fill=tk.X, padx=(20, 0))
            tk.Label(blur_options_frame, text="Blur Kernel (W H, e.g., 5 5):").pack(anchor=tk.W)
            tk.Entry(blur_options_frame, textvariable=self.blur_kernel_var).pack(anchor=tk.W, fill=tk.X)
            agg_frame = ttk.Frame(param_frame)
            agg_frame.pack(fill=tk.X, pady=(5, 0))
            tk.Checkbutton(agg_frame, text="Aggregate", variable=self.aggregate_var).pack(anchor=tk.W)
            agg_options_frame = ttk.Frame(agg_frame)
            agg_options_frame.pack(fill=tk.X, padx=(20, 0))
            tk.Label(agg_options_frame, text="Agg. Kernel Size:").pack(anchor=tk.W)
            tk.Entry(agg_options_frame, textvariable=self.agg_kernel_size_var).pack(anchor=tk.W, fill=tk.X)
            tk.Label(agg_options_frame, text="Agg. Min Area:").pack(anchor=tk.W)
            tk.Entry(agg_options_frame, textvariable=self.agg_min_area_var).pack(anchor=tk.W, fill=tk.X)
            tk.Label(agg_options_frame, text="Agg. Density Thresh:").pack(anchor=tk.W)
            tk.Entry(agg_options_frame, textvariable=self.agg_density_thresh_var).pack(anchor=tk.W, fill=tk.X)

        tk.Button(main_frame, text="Run Analysis", command=self.run_analysis_dialog).pack(fill=tk.X, pady=10, padx=5)

    def populate_history_tab(self):
        main_frame = self.history_tab
        
        # Top frame for controls
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        scan_button = ttk.Button(controls_frame, text="Scan for Reports", command=self._scan_and_load_history)
        scan_button.pack(side=tk.LEFT)

        # Filters frame
        filters_frame = ttk.LabelFrame(main_frame, text="Filters")
        filters_frame.pack(fill=tk.X, padx=5, pady=5)

        # Treeview for displaying history
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = {
            'date': ("Date", 150),
            'project': ("Project", 100),
            'part_number': ("Part Number", 120),
            'thickness': ("Thickness", 80),
            'percentage': ("Color %", 70)
        }
        self.history_tree = ttk.Treeview(tree_frame, columns=list(columns.keys()), show="headings")
        
        for col, (text, width) in columns.items():
            self.history_tree.heading(col, text=text, command=lambda c=col: self._sort_history_tree(c, False))
            self.history_tree.column(col, width=width, anchor=tk.W)

        # Add filter entries
        for i, col in enumerate(columns.keys()):
            entry = ttk.Entry(filters_frame, textvariable=self.history_filter_vars[col])
            entry.grid(row=0, column=i, padx=2, pady=2, sticky=tk.EW)
            entry.bind("<KeyRelease>", self._apply_history_filters)
            filters_frame.grid_columnconfigure(i, weight=1)

        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.history_tree.bind("<<TreeviewSelect>>", self._on_history_select)

        # Bottom frame for actions
        actions_frame = ttk.Frame(main_frame)
        actions_frame.pack(fill=tk.X, padx=5, pady=5)
        self.recreate_button = ttk.Button(actions_frame, text="Recreate Selected Report", command=self._recreate_report, state=tk.DISABLED)
        self.recreate_button.pack(side=tk.LEFT)

        self.recreate_debug_var = tk.BooleanVar()
        self.recreate_debug_check = ttk.Checkbutton(actions_frame, text="Generate in Debug Mode", variable=self.recreate_debug_var)
        if self.debug_mode:
            self.recreate_debug_check.pack(side=tk.LEFT, padx=10)

    def _scan_and_load_history(self):
        self.history_data.clear()
        output_path = Path("output")
        if not output_path.exists():
            messagebox.showinfo("Info", "The 'output' directory does not exist. No reports found.")
            return

        gri_files = list(output_path.rglob("*.gri"))
        if not gri_files:
            messagebox.showinfo("Info", "No '.gri' report files found in the 'output' directory.")
            return

        progress_win = tk.Toplevel(self)
        progress_win.title("Scanning...")
        progress_bar = ttk.Progressbar(progress_win, orient='horizontal', length=300, mode='determinate')
        progress_bar.pack(padx=10, pady=10)
        progress_bar['maximum'] = len(gri_files)

        for i, file_path in enumerate(gri_files):
            try:
                with open(file_path, 'rb') as f:
                    saved_data = pickle.load(f)

                project_name = 'Unknown'
                metadata = {}
                analysis_results = {}

                # Safely extract data from both new (dict) and old (object) formats
                if isinstance(saved_data, dict):
                    project_name = saved_data.get('project_name', 'Unknown')
                    metadata = saved_data.get('metadata', {})
                    analysis_results = saved_data.get('analysis_results_raw', {})
                else: # Assuming old Pipeline object
                    project_name = getattr(saved_data.args, 'project', 'Unknown') if hasattr(saved_data, 'args') else 'Unknown'
                    metadata = getattr(saved_data, 'metadata', {})
                    analysis_results = getattr(saved_data, 'analysis_results', {})
                
                metadata = metadata if isinstance(metadata, dict) else {}
                analysis_results = analysis_results if isinstance(analysis_results, dict) else {}

                # Calculate percentage manually for robustness
                matched = analysis_results.get('matched_pixels', 0)
                total = analysis_results.get('total_pixels', 0)
                percentage = (matched / total) * 100 if total > 0 else 0.0

                self.history_data.append({
                    'gri_path': file_path,
                    'date': datetime.fromtimestamp(file_path.stat().st_mtime),
                    'project': project_name,
                    'part_number': metadata.get('part_number', 'N/A'),
                    'thickness': metadata.get('thickness', 'N/A'),
                    'percentage': percentage
                })

            except Exception as e:
                print(f"Could not load or parse {file_path}: {e}")
            progress_bar['value'] = i + 1
            progress_win.update_idletasks()
        
        progress_win.destroy()
        self._apply_history_filters()

    def _populate_history_treeview(self, data_to_show):
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        for item in data_to_show:
            values = (
                item['date'].strftime("%Y-%m-%d %H:%M"),
                item['project'],
                item['part_number'],
                item['thickness'],
                f"{item['percentage']:.2f}%"
            )
            self.history_tree.insert("", tk.END, values=values, iid=str(item['gri_path']))

    def _apply_history_filters(self, event=None):
        filters = {key: var.get().lower() for key, var in self.history_filter_vars.items()}
        
        if not any(filters.values()):
            filtered_data = self.history_data
        else:
            filtered_data = []
            for item in self.history_data:
                match = True
                if filters['date'] and filters['date'] not in item['date'].strftime("%Y-%m-%d %H:%M").lower():
                    match = False
                if filters['project'] and filters['project'] not in item['project'].lower():
                    match = False
                if filters['part_number'] and filters['part_number'] not in str(item['part_number']).lower():
                    match = False
                if filters['thickness'] and filters['thickness'] not in str(item['thickness']).lower():
                    match = False
                if filters['percentage'] and filters['percentage'] not in f"{item['percentage']:.2f}%".lower():
                    match = False
                
                if match:
                    filtered_data.append(item)
        
        self._populate_history_treeview(filtered_data)

    def _sort_history_tree(self, col, reverse):
        data = [self.history_tree.set(child, col) for child in self.history_tree.get_children('')]
        
        # A simple string-based sort; can be improved for numeric/date types
        sorted_data = sorted(self.history_data, key=lambda item: str(item.get(col, '')), reverse=reverse)
        
        self._populate_history_treeview(sorted_data)
        self.history_tree.heading(col, command=lambda: self._sort_history_tree(col, not reverse))

    def _on_history_select(self, event=None):
        self.recreate_button.config(state=tk.NORMAL if self.history_tree.selection() else tk.DISABLED)

    def _recreate_report(self):
        selected_items = self.history_tree.selection()
        if not selected_items:
            return
        
        gri_path_str = selected_items[0]
        gri_path = Path(gri_path_str)

        try:
            with open(gri_path, 'rb') as f:
                saved_data = pickle.load(f)

            # Determine a default name for the save dialog
            pn = "report"
            if isinstance(saved_data, dict):
                pn = saved_data.get('metadata', {}).get('part_number', 'report')
            else:
                pn = getattr(saved_data, 'metadata', {}).get('part_number', 'report')
            default_filename = f"regenerated_{pn}.pdf"

            # Ask user where to save the new PDF
            save_path = filedialog.asksaveasfilename(
                title="Save Regenerated PDF Report As...",
                initialfile=default_filename,
                defaultextension=".pdf",
                filetypes=[("PDF Documents", "*.pdf")]
            )

            if not save_path:
                messagebox.showinfo("Cancelled", "Report regeneration was cancelled.")
                return

            messagebox.showinfo("Regenerating Report", f"Regenerating report from {gri_path.name}. This may take a moment.")

            # Handle both old and new .gri formats
            if isinstance(saved_data, dict):
                report_data = saved_data
                report_generator = ReportGenerator(project_name=report_data.get('project_name'), sample_name=report_data.get('part_number'), debug_mode=self.recreate_debug_var.get())
                report_generator.generate_from_archived_data(report_data, base_dir=gri_path.parent.parent, external_pdf_path=save_path)
            else:
                # Old format: data is a full Pipeline object
                pipeline_state = saved_data
                pipeline_state.args.debug = self.recreate_debug_var.get()
                pipeline_state.generate_report(external_pdf_path=save_path)

            messagebox.showinfo("Success", f"Report saved successfully to:\n{save_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to regenerate report: {e}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()

    def populate_create_project_tab(self):
        main_frame = self.create_project_tab
        create_frame = tk.LabelFrame(main_frame, text="Create a New Project")
        create_frame.pack(fill=tk.X, pady=10, padx=10)
        tk.Label(create_frame, text="New Project Name:").pack(side=tk.LEFT, padx=5, pady=5)
        tk.Entry(create_frame, textvariable=self.new_project_name_var, width=50).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, pady=5)
        tk.Button(create_frame, text="Create", command=self.handle_create_project).pack(side=tk.LEFT, padx=5, pady=5)
        self.create_project_output = scrolledtext.ScrolledText(main_frame, height=20)
        self.create_project_output.pack(pady=5, padx=10, fill="both", expand=True)
        self.create_project_output.config(state=tk.DISABLED)

    def populate_manage_dataset_tab(self):
        main_frame = self.manage_dataset_tab
        manage_frame = tk.LabelFrame(main_frame, text="Launch Dataset Manager")
        manage_frame.pack(fill=tk.X, pady=10, padx=10)
        tk.Label(manage_frame, text="Select Project:").pack(side=tk.LEFT, padx=5, pady=5)
        self.manage_project_combobox = ttk.Combobox(manage_frame, textvariable=self.manage_project_var, values=self.available_projects)
        self.manage_project_combobox.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, pady=5)
        if self.available_projects:
            self.manage_project_combobox.set(self.available_projects[0])
        tk.Button(manage_frame, text="Launch Point Selector", command=self.launch_dataset_manager).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(manage_frame, text="Setup Project Files", command=self.launch_file_placer).pack(side=tk.LEFT, padx=5, pady=5)

    def handle_create_project(self):
        project_name = self.new_project_name_var.get()
        if not project_name:
            messagebox.showerror("Error", "Please enter a project name.")
            return
        messages = create_project(project_name)
        self.create_project_output.config(state=tk.NORMAL)
        self.create_project_output.delete("1.0", tk.END)
        self.create_project_output.insert(tk.END, "\n".join(messages))
        self.create_project_output.config(state=tk.DISABLED)
        self.available_projects = self.project_manager.list_projects()
        self.project_combobox["values"] = self.available_projects
        self.manage_project_combobox["values"] = self.available_projects

    def launch_file_placer(self):
        project_name = self.manage_project_var.get()
        if not project_name:
            messagebox.showerror("Error", "Please select a project.")
            return
        file_placer_window = ProjectFilePlacerGUI(self, project_name)
        self.wait_window(file_placer_window)

    def launch_dataset_manager(self):
        project_name = self.manage_project_var.get()
        if not project_name:
            messagebox.showerror("Error", "Please select a project.")
            return
        try:
            project_files = self.project_manager.get_project_file_paths(project_name, debug_mode=self.debug_mode)
            dataset_image_configs = project_files.get("training_image_configs", [])
            dataset_image_paths = [cfg["path"] for cfg in dataset_image_configs]
            if not dataset_image_paths:
                messagebox.showinfo("Info", rf"No training images found in project '{project_name}'. Please add images to the 'training' folder and ensure the 'training_path' in project_config.json is correct.")
                return
            config_file_path = (self.project_manager.projects_root / project_name / "dataset_item_processing_config.json")
            dataset_manager_window = tk.Toplevel(self)
            DatasetManagerGUI(dataset_manager_window, dataset_image_paths, str(config_file_path))
            self.wait_window(dataset_manager_window)
            messagebox.showinfo("Dataset Manager", "Dataset management complete. Remember to re-run analysis to use updated color space.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while trying to open the dataset manager: {e}")

    def select_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.image_path_var.set(path)
            file_name_without_ext = os.path.splitext(os.path.basename(path))[0]
            file_parts = file_name_without_ext.split("_")
            if len(file_parts) >= 3:
                part_number = file_parts[2]
                thickness = file_parts[3] if len(file_parts) >= 4 else "N/A"
            else:
                part_number = file_name_without_ext
                thickness = "N/A"
            self.part_number_var.set(part_number)
            self.thickness_var.set(thickness)

    def select_color_checker(self):
        path = filedialog.askopenfilename()
        if path:
            self.color_checker_path_var.set(path)

    def run_analysis_dialog(self):
        if self.debug_mode:
            debug_choice = messagebox.askyesno("Run Mode", "Do you want to run the pipeline in Debug Mode?")
            self.debug_var.set(debug_choice)
        else:
            self.debug_var.set(False)
        self.run_analysis()

    def run_analysis(self):
        args = argparse.Namespace()
        args.project = self.project_var.get()
        if not args.project:
            messagebox.showerror("Error", "Please select a project.")
            return
        args.image = self.image_path_var.get()
        if not args.image:
            messagebox.showerror("Error", "Please select an image.")
            return
        
        args.part_number = self.part_number_var.get()
        args.thickness = self.thickness_var.get()
        args.author = self.author_var.get()
        args.department = self.department_var.get()
        args.report_title = self.report_title_var.get()

        args.video = None
        args.camera = False
        args.drawing = None
        args.color_alignment = self.color_alignment_var.get()
        args.color_correction_method = self.color_correction_method_var.get()
        args.sample_color_checker = (self.color_checker_path_var.get() if args.color_alignment else None)
        
        if args.color_alignment and not args.sample_color_checker:
            messagebox.showerror("Error", "Please select a color checker image for Color Alignment.")
            return

        args.debug = self.debug_var.get()
        args.aggregate = self.aggregate_var.get()
        args.blur = self.blur_var.get()
        args.alignment = self.alignment_var.get()
        args.object_alignment = self.object_alignment_var.get()
        args.object_alignment_shadow_removal = (self.object_alignment_shadow_removal_var.get())
        args.apply_mask = self.apply_mask_var.get()
        args.mask_bg_is_white = self.mask_bg_is_white_var.get()
        args.symmetry = self.symmetry_var.get()
        args.masking_order = self.masking_order_var.get()

        args.load_state_from = None
        args.save_state_to = None
        args.skip_color_analysis = False
        # --- Defer report generation to a manual step ---
        args.skip_report_generation = True

        try:
            args.agg_kernel_size = (int(self.agg_kernel_size_var.get()) if self.agg_kernel_size_var.get() else 7)
            args.agg_min_area = (float(self.agg_min_area_var.get()) if self.agg_min_area_var.get() else 0.0005)
            args.agg_density_thresh = (float(self.agg_density_thresh_var.get()) if self.agg_density_thresh_var.get() else 0.5)
            if self.blur_kernel_var.get():
                w, h = map(int, self.blur_kernel_var.get().split())
                if w % 2 == 0 or h % 2 == 0:
                    messagebox.showerror("Error", "Blur kernel dimensions must be odd.")
                    return
                args.blur_kernel = [w, h]
            else:
                args.blur_kernel = None
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid numeric value for an option: {e}")
            return
        try:
            messagebox.showinfo("Running Analysis", "Analysis is starting. This may take a moment.")
            
            # Run analysis but don't generate the report yet
            pipeline_instance = run_analysis(args)

            if pipeline_instance:
                messagebox.showinfo("Success", "Analysis pipeline completed. Now, please choose where to save the report.")
                self._prompt_for_report_saving(pipeline_instance)
            else:
                messagebox.showwarning("Analysis Warning", "Analysis finished, but no results were returned to generate a report.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during analysis:{e}")

    def _prompt_for_report_saving(self, pipeline_instance):
        """Asks the user where to save the PDF report and then generates it."""
        # Determine default report name
        pn = pipeline_instance.metadata.get("part_number", "report")
        safe_pn = "".join(c for c in pn if c.isalnum() or c in ('-', '_')).rstrip()
        default_filename = f"{safe_pn}_report.pdf"

        # Ask user for save location
        save_path = filedialog.asksaveasfilename(
            title="Save PDF Report As...",
            initialfile=default_filename,
            defaultextension=".pdf",
            filetypes=[("PDF Documents", "*.pdf")]
        )

        if not save_path:
            messagebox.showinfo("Cancelled", "Report generation was cancelled.")
            return

        # Determine if debug mode should be used for the report
        generate_in_debug = False
        if self.debug_mode:
            generate_in_debug = messagebox.askyesno("Report Type", "Generate a DEBUG report? (Includes extra pages and details)")
        
        pipeline_instance.args.debug = generate_in_debug

        try:
            messagebox.showinfo("Generating Report", "Generating PDF report. This may take a moment.")
            pipeline_instance.generate_report(external_pdf_path=save_path)
            messagebox.showinfo("Success", f"Report saved successfully to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Analyzer GUI.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for GUI layout.")
    gui_args = parser.parse_args()
    app = VisualAnalyzerGUI(debug_mode=gui_args.debug)
    app.mainloop()
