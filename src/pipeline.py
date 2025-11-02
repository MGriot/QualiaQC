"""
This module defines the main analysis pipeline for the Visual Analyzer application.

It contains the `Pipeline` class, which orchestrates the entire sequence of
image processing and analysis tasks. The pipeline is configurable via command-line
arguments and project-specific settings.

The typical execution flow is as follows:
1.  Load project data (color ranges, reference paths).
2.  Load an input image.
3.  Execute a series of optional and required processing steps in order:
    - Color Correction
    - Geometrical (ArUco) Alignment
    - Object Alignment
    - Masking (Background Removal)
    - Blurring
    - Color Analysis
    - Symmetry Analysis
4.  Generate a comprehensive report (HTML/PDF) summarizing the results.

The pipeline is designed to be modular, with each major step encapsulated in its
own method and leveraging a dedicated class from other modules.
"""

import os
import cv2
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pickle
import datetime

# Suppress NumPy warnings that might arise from operations like empty slices
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

# Import core modules for the Visual Analyzer application
from src.project_manager import ProjectManager
from src.color_analysis.analyzer import ColorAnalyzer
from src.color_correction.corrector import ColorCorrector
from src.geometric_alignment.geometric_aligner import Aligner
from src.object_alignment.object_aligner import AdvancedAligner
from src.masking.creator import MaskCreator
from src.reporting.generator import ReportGenerator
from src.utils.image_utils import load_image, save_image, blur_image
from src.utils.video_utils import process_video_stream
from src import config
from src.symmetry_analysis.symmetry import SymmetryAnalyzer

class Pipeline:
    """
    Orchestrates the full image analysis workflow from loading to reporting.

    This class manages the state of an image as it passes through various
    processing steps. It initializes all necessary component classes and uses
    command-line arguments to determine which steps to execute.

    Attributes:
        args (argparse.Namespace): Command-line arguments.
        project_manager (ProjectManager): Manages project configurations and data.
        color_corrector (ColorCorrector): Handles color correction tasks.
        color_analyzer (ColorAnalyzer): Performs color zone analysis.
        project_data (dict): Loaded configuration and data for the current project.
        image_path (str): Path to the current image being processed.
        image_to_be_processed (np.ndarray): The image at the current stage of the
                                            pipeline. This is modified in-place by
                                            each processing step.
        analysis_results (dict): The final results from the color analysis step.
        debug_data_for_report (dict): A collection of intermediate data and stats
                                      for inclusion in debug reports.
        debug_image_pipeline (list): A list of paths and titles for images
                                     generated at each pipeline step, used for
                                     visualizing the workflow in debug reports.
    """
    def __init__(self, args: 'argparse.Namespace'):
        """
        Initializes the Pipeline instance.

        Args:
            args (argparse.Namespace): The command-line arguments passed to the main script.
        """
        self.args = args
        self.project_manager = ProjectManager()
        self.color_corrector = ColorCorrector()
        self.color_analyzer = ColorAnalyzer()
        self.project_data = None
        self.image_path = None
        self.original_input_image_bgr = None
        self.image_to_be_processed = None
        self.analysis_results = None
        self.metadata = None
        self.metadata_overrides = None  # Add override placeholder
        self.debug_data_for_report = {}
        self.debug_image_pipeline = []
        self.masked_image_path = None
        self.pipeline_step_counter = 1
        self.pipeline_image_stages = {} # To store image output of each step

    def load_project_data(self):
        self.project_data = self.project_manager.get_project_data(
            self.args.project, debug_mode=self.args.debug
        )
        if self.args.debug:
            print(
                f"Loaded project '{self.args.project}' with HSV range: {self.project_data['lower_hsv']} - {self.project_data['upper_hsv']}"
            )

    def process_image(self, image_path, part_number=None, thickness=None, author=None, department=None, report_title=None):
        self.image_path = image_path
        self.metadata_overrides = {"part_number": part_number, "thickness": thickness, "author": author, "department": department, "report_title": report_title}

        # --- Step 1: Extract metadata early to determine output path ---
        self._extract_metadata()

        # --- Step 2: Create a unique, descriptive folder name ---
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        pn = self.metadata.get("part_number", "PN-NA")
        thick = self.metadata.get("thickness", "THICK-NA")
        # Sanitize for folder names
        safe_pn = "".join(c for c in pn if c.isalnum() or c in ('-', '_')).rstrip()
        safe_thick = "".join(c for c in thick if c.isalnum() or c in ('-', '_')).rstrip()
        output_folder_name = f"{timestamp}_{safe_pn}_{safe_thick}"

        # --- Step 3: Instantiate the report generator with the new path ---
        self.report_generator = ReportGenerator(
            self.args.project, sample_name=output_folder_name, debug_mode=self.args.debug
        )

        self.original_input_image_bgr, _ = load_image(self.image_path)
        if self.original_input_image_bgr is None:
            raise ValueError(f"Could not load image {self.image_path}")

        # Store the original image as the first stage
        self.pipeline_image_stages["original"] = self.original_input_image_bgr.copy()

        if self.args.debug:
            import shutil
            from pathlib import Path

            # Copy the original image to the report directory to make paths relative and self-contained.
            copied_input_filename = f"00_original_input{Path(self.image_path).suffix}"
            copied_input_path = (
                self.report_generator.project_output_dir / copied_input_filename
            )
            shutil.copy(self.image_path, copied_input_path)

            self.debug_data_for_report["dataset_debug_info"] = self.project_data[
                "dataset_debug_info"
            ]
            self.debug_image_pipeline.append(
                {
                    "title": f"{self.pipeline_step_counter}. Original Input",
                    "path": copied_input_filename,  # Use the relative path (filename)
                }
            )
            self.pipeline_step_counter += 1

        self.image_to_be_processed = self.original_input_image_bgr.copy()

        # The rest of the processing steps from the original process_image function will be here
        # ... (color correction, alignment, masking, etc.)
        self.run_full_pipeline()

    def run_full_pipeline(self):
        if self.args.color_alignment:
            self._perform_color_correction()

        if self.args.alignment:
            self._perform_geometrical_alignment()

        if self.args.object_alignment:
            self._perform_object_alignment()

        if self.args.apply_mask:
            self._apply_masking()

        if self.args.blur:
            self._perform_blur()

        if not self.args.skip_color_analysis:
            self._perform_color_analysis()

        if self.args.symmetry:
            self._perform_symmetry_analysis()

        self._extract_metadata()

        if not self.args.skip_report_generation:
            self.generate_report()


    def _perform_color_correction(self):
        """
        Applies color correction. Prioritizes on-the-fly correction, which includes
        manual fallbacks. Falls back to a pre-calculated project matrix if on-the-fly fails.
        """
        step_dir = self.report_generator.get_step_output_dir("color_correction")

        # --- Scenario 1: On-the-fly correction (Primary) ---
        project_files = self.project_manager.get_project_file_paths(self.args.project, self.args.debug)
        ideal_checker_path = project_files.get("reference_color_checker")

        if ideal_checker_path:
            if self.args.debug:
                print(f"[DEBUG] Attempting on-the-fly correction with ColorCorrectionPipeline.")

            try:
                from src.color_correction.color_correction_pipeline import ColorCorrectionPipeline

                correction_pipeline = ColorCorrectionPipeline(
                    reference_color_checker_path=str(ideal_checker_path),
                    output_dir=step_dir,
                    debug_mode=self.args.debug
                )

                result = correction_pipeline.run(
                    image_to_correct=self.image_to_be_processed,
                    photo_of_checker_path=self.args.sample_color_checker,
                    correction_method=self.args.color_correction_method
                )

                if result and result.get("corrected_image") is not None:
                    self.image_to_be_processed = result["corrected_image"]
                    self.pipeline_image_stages["color_corrected"] = self.image_to_be_processed.copy()

                    self.debug_data_for_report["color_correction_method"] = self.args.color_correction_method
                    correction_model = result.get("correction_model", {})
                    serializable_model = {}
                    if 'matrix' in correction_model and correction_model['matrix'] is not None:
                        serializable_model['matrix'] = correction_model['matrix'].tolist()
                    if 'luts' in correction_model and correction_model['luts'] is not None:
                        serializable_model['luts'] = [lut.tolist() for lut in correction_model['luts']]
                    self.debug_data_for_report["color_correction_model_on_the_fly"] = serializable_model
                    
                    if self.args.debug and result.get("debug_paths"):
                        for key, path in sorted(result["debug_paths"].items()):
                            title = key.replace("_", " ").title()
                            self.debug_image_pipeline.append({
                                "title": f"{self.pipeline_step_counter}. CC: {title}",
                                "path": os.path.relpath(path, self.report_generator.project_output_dir),
                            })
                        self.pipeline_step_counter += 1
                    
                    return  # On-the-fly correction successful

            except Exception as e:
                print(f"[WARNING] On-the-fly color correction failed with an unexpected error: {e}")
                if self.args.debug:
                    import traceback
                    traceback.print_exc()
        else:
            print("[WARNING] Cannot perform on-the-fly correction: ideal reference_color_checker_path not set in project.")

        # --- Scenario 2: Fallback to pre-calculated project matrix ---
        if self.args.debug:
            print("[DEBUG] On-the-fly correction failed or was not possible. Falling back to pre-calculated matrix.")
        
        correction_model = self.project_data.get("correction_matrix")
        if correction_model is not None:
            if self.args.debug:
                print("[DEBUG] Applying pre-calculated project color correction matrix.")
                self.debug_data_for_report["color_correction_model_used"] = correction_model
            
            self.image_to_be_processed = self.color_corrector.apply_correction_model(
                self.image_to_be_processed,
                correction_model,
                method='linear' 
            )
            self.pipeline_image_stages["color_corrected"] = self.image_to_be_processed.copy()

            if self.args.debug:
                path = os.path.join(step_dir, "05_corrected_with_project_matrix.png")
                save_image(path, self.image_to_be_processed)
                self.debug_image_pipeline.append({
                    "title": f"{self.pipeline_step_counter}. After Project Color Correction",
                    "path": os.path.relpath(path, self.report_generator.project_output_dir),
                })
                self.pipeline_step_counter += 1
        elif self.args.debug:
            print("[DEBUG] No color correction matrix available. Skipping color correction entirely.")
    def _perform_geometrical_alignment(self):
        step_dir = self.report_generator.get_step_output_dir("geometrical_alignment")
        aligner = Aligner(debug_mode=self.args.debug, output_dir=step_dir)

        # Get file paths and config objects from the project manager
        project_files = self.project_manager.get_project_file_paths(
            self.args.project, self.args.debug
        )

        # Get the specific config object for this step
        geo_config = project_files.get("geometrical_alignment_config")
        if not geo_config:
            if self.args.debug:
                print(
                    "[DEBUG] No geometrical alignment config found in project_files. Skipping step."
                )
            return

        if self.args.debug:
            print(
                f"[DEBUG] Available keys in project_files: {list(project_files.keys())}"
            )
        aruco_ref_path = project_files.get("geometrical_alignment_reference_path")

        result = aligner.align_image(
            image=self.image_to_be_processed,
            aruco_reference_path=str(aruco_ref_path) if aruco_ref_path else None,
            marker_map=geo_config.marker_map,
            output_size_wh=geo_config.output_size,
        )

        if result and result.get("image") is not None:
            self.image_to_be_processed = result["image"]
            self.pipeline_image_stages["geometrically_aligned"] = (
                self.image_to_be_processed.copy()
            )
            self.debug_data_for_report["geometrical_alignment_data"] = result.get(
                "alignment_data"
            )

            if self.args.debug and result.get("debug_paths"):
                # Define titles for known debug images
                path_titles = {
                    "detected_markers": "Detected ArUco Markers",
                    "final_aligned": "After Geometrical Alignment",
                }
                for key, path in result["debug_paths"].items():
                    title = path_titles.get(key, key.replace("_", " ").title())
                    self.debug_image_pipeline.append(
                        {
                            "title": f"{self.pipeline_step_counter}. GA: {title}",
                            "path": os.path.relpath(
                                path, self.report_generator.project_output_dir
                            ),
                        }
                    )
                    self.pipeline_step_counter += 1
        elif self.args.debug:
            print(
                "[WARNING] Geometrical alignment failed. Proceeding without alignment."
            )

    def _perform_object_alignment(self):
        """
        Performs object alignment by calling the self-contained AdvancedAligner.

        The aligner is responsible for all its own logic and debug image generation.
        This method's role is to call the aligner and integrate its results and
        debug data back into the main pipeline workflow.
        """
        if self.args.debug:
            print("[DEBUG] Entering object alignment step...")

        step_dir = self.report_generator.get_step_output_dir("object_alignment")
        project_files = self.project_manager.get_project_file_paths(
            self.args.project, self.args.debug
        )
        object_ref_path = project_files.get("object_reference_path")

        if not object_ref_path:
            if self.args.debug:
                print(
                    "[DEBUG] No object reference path found in project config. Skipping object alignment."
                )
            return

        ref_image, _ = load_image(str(object_ref_path))
        if ref_image is None:
            if self.args.debug:
                print(
                    f"[WARNING] Could not load object reference image at {object_ref_path}. Skipping object alignment."
                )
            return

        # Instantiate the aligner, which will manage its own state and debug output.
        advanced_aligner = AdvancedAligner(
            debug_mode=self.args.debug, output_dir=step_dir
        )

        # Call the aligner, telling it which shadow method to use.
        # Safely access the shadow removal argument, defaulting to 'none'.
        shadow_removal_method = getattr(self.args, "object_alignment_shadow_removal", "none")
        if self.args.debug and not hasattr(self.args, "object_alignment_shadow_removal"):
            print("[DEBUG] 'object_alignment_shadow_removal' argument not found, defaulting to 'none'.")

        result = advanced_aligner.align(
            self.image_to_be_processed,
            ref_image,
            shadow_removal=shadow_removal_method,
        )

        # Process ALL debug paths returned by the aligner, whatever they may be.
        if self.args.debug and result.get("debug_paths"):
            # Sort by key to maintain the logical numbered order (e.g., "00_", "01_") in the report.
            sorted_paths = sorted(result["debug_paths"].items())
            for key, path in sorted_paths:
                if isinstance(path, str) and os.path.exists(path):
                    # Create a user-friendly title from the filename key
                    title = key.replace("_", " ").title()
                    self.debug_image_pipeline.append(
                        {
                            "title": f"{self.pipeline_step_counter}. OA: {title}",
                            "path": os.path.relpath(
                                path, self.report_generator.project_output_dir
                            ),
                        }
                    )
                    self.pipeline_step_counter += 1

        # Update the pipeline's image and data only on success.
        if result and result.get("image") is not None:
            if self.args.debug:
                print(
                    "[DEBUG] Object alignment successful. Updating image for next step."
                )
            self.image_to_be_processed = result["image"]
            self.pipeline_image_stages["object_aligned"] = (
                self.image_to_be_processed.copy()
            )
            if result.get("transform_matrix") is not None:
                self.debug_data_for_report["object_alignment_matrix"] = result[
                    "transform_matrix"
                ]
        else:
            if self.args.debug:
                print(
                    "[WARNING] Object alignment core logic failed. Proceeding with unaligned image."
                )

    def _apply_masking(self):
        step_dir = self.report_generator.get_step_output_dir("masking")
        project_files = self.project_manager.get_project_file_paths(
            self.args.project, self.args.debug
        )

        # The masking_order argument is now handled inside the new function.
        masking_order = []
        if self.args.masking_order:
            masking_order = [
                layer for layer in self.args.masking_order.split("-") if layer
            ]

        if not masking_order:
            raise ValueError(
                "Masking is enabled (--apply-mask), but no valid layers were specified via --masking-order. Please provide a value like '1' or '1-2'."
            )

        # Defer all logic to the self-contained function in the masking module.
        from src.masking.creator import create_and_apply_mask_from_layers

        # Pass the new technical_drawing_paths dictionary
        result = create_and_apply_mask_from_layers(
            image_to_be_processed=self.image_to_be_processed,
            drawing_paths=project_files.get("technical_drawing_paths", {}),
            masking_order=masking_order,
            mask_bg_is_white=self.args.mask_bg_is_white,
            output_dir=step_dir,
            debug_mode=self.args.debug,
        )

        self.image_to_be_processed = result["image"]
        self.masked_image_path = result.get("path")
        self.pipeline_image_stages["masked"] = self.image_to_be_processed.copy()
        if result.get("stats"):
            self.debug_data_for_report["masking_stats"] = result["stats"]

        if self.args.debug and result["debug_paths"]:
            for debug_info in result["debug_paths"]:
                self.debug_image_pipeline.append(
                    {
                        "title": f"{self.pipeline_step_counter}. {debug_info['title']}",
                        "path": os.path.relpath(
                            debug_info["path"], self.report_generator.project_output_dir
                        ),
                    }
                )
                self.pipeline_step_counter += 1

    def _perform_color_analysis(self):
        step_dir = self.report_generator.get_step_output_dir("color_analysis")

        if self.args.debug:
            debug_path = os.path.join(step_dir, "image_sent_to_analyzer.png")
            save_image(debug_path, self.image_to_be_processed)
            print(
                f"[DEBUG] Diagnostic image saved: the exact image being sent to ColorAnalyzer is at {debug_path}"
            )

        self.analysis_results = self.color_analyzer.process_image(
            image=self.image_to_be_processed,
            original_image_path=self.image_path,
            lower_hsv=self.project_data["lower_hsv"],
            upper_hsv=self.project_data["upper_hsv"],
            center_hsv=self.project_data["center_hsv"],
            output_dir=step_dir,
            debug_mode=self.args.debug,
            aggregate_mode=self.args.aggregate,
            agg_kernel_size=self.args.agg_kernel_size,
            agg_min_area=self.args.agg_min_area,
            agg_density_thresh=self.args.agg_density_thresh,
        )
        self.pipeline_image_stages["color_analyzed"] = self.analysis_results[
            "processed_image"
        ].copy()

        if self.args.debug and self.analysis_results.get("debug_info"):
            for debug_item in self.analysis_results["debug_info"]:
                self.debug_image_pipeline.append(
                    {
                        "title": f"{self.pipeline_step_counter}. CA: {debug_item['title']}",
                        "path": os.path.relpath(
                            debug_item["path"], self.report_generator.project_output_dir
                        ),
                    }
                )
                self.pipeline_step_counter += 1

    def _perform_symmetry_analysis(self):
        """
        Performs symmetry analysis on the binary mask from color analysis.
        """
        if not self.analysis_results or "binary_mask" not in self.analysis_results:
            if self.args.debug:
                print(
                    "[WARNING] Skipping symmetry analysis because color analysis results are not available."
                )
            return

        binary_mask = self.analysis_results["binary_mask"]
        if binary_mask is None or np.count_nonzero(binary_mask) == 0:
            if self.args.debug:
                print(
                    "[WARNING] Skipping symmetry analysis because the binary mask is empty."
                )
            return

        step_dir = self.report_generator.get_step_output_dir("symmetry_analysis")

        symmetry_analyzer = SymmetryAnalyzer(
            binary_mask, output_dir=step_dir, debug_mode=self.args.debug
        )
        symmetry_analyzer.analyze_all()
        self.debug_data_for_report["symmetry_results"] = symmetry_analyzer.results

        if self.args.debug and "visualizations" in symmetry_analyzer.results:
            for viz_info in symmetry_analyzer.results["visualizations"]:
                self.debug_image_pipeline.append(
                    {
                        "title": f"{self.pipeline_step_counter}. {viz_info['title']}",
                        "path": os.path.relpath(
                            viz_info["path"], self.report_generator.project_output_dir
                        ),
                    }
                )
                self.pipeline_step_counter += 1

    def _perform_blur(self):
        step_dir = self.report_generator.get_step_output_dir("blur")

        blur_result = blur_image(
            self.image_to_be_processed, self.args.blur_kernel, output_dir=step_dir
        )

        self.image_to_be_processed = blur_result["image"]
        self.pipeline_image_stages["blurred"] = self.image_to_be_processed.copy()
        self.debug_data_for_report["blur_kernel_used"] = blur_result["kernel_used"]

        if self.args.debug and blur_result["debug_path"]:
            self.debug_image_pipeline.append(
                {
                    "title": f"{self.pipeline_step_counter}. After Blur (Kernel: {blur_result['kernel_used']})",
                    "path": os.path.relpath(
                        blur_result["debug_path"],
                        self.report_generator.project_output_dir,
                    ),
                }
            )
            self.pipeline_step_counter += 1

    def _extract_metadata(self):
        # Prioritize metadata from GUI if available
        if self.metadata_overrides and self.metadata_overrides.get("part_number"):
            self.metadata = self.metadata_overrides
            return

        # Fallback to parsing the filename
        file_name_without_ext = os.path.splitext(os.path.basename(self.image_path))[0]
        file_parts = file_name_without_ext.split("_")
        if len(file_parts) >= 3:
            part_number = file_parts[2]
            thickness = file_parts[3] if len(file_parts) >= 4 else "N/A"
        else:
            part_number = file_name_without_ext
            thickness = "N/A"
        self.metadata = {"part_number": part_number, "thickness": thickness}

    def generate_report(self, external_pdf_path: str = None):
        if self.analysis_results is None:
            print("[WARNING] No analysis results to generate report from.")
            return None

        # Add the collected pipeline images to the main debug data dictionary
        self.debug_data_for_report["image_pipeline"] = self.debug_image_pipeline

        masked_image_path_relative = None
        if self.masked_image_path:
            masked_image_path_relative = os.path.relpath(
                self.masked_image_path, self.report_generator.project_output_dir
            )

        project_files = self.project_manager.get_project_file_paths(self.args.project, self.args.debug)
        logo_path = project_files.get("logo_path")

        report_data = self.report_generator.generate_report(
            self.analysis_results,
            self.metadata,
            debug_data=self.debug_data_for_report,
            external_pdf_path=external_pdf_path,
            masked_image_path=masked_image_path_relative,
            logo_path=logo_path,
        )
        return report_data

    def save_state(self, path):
        """Saves the pipeline state to a file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        if self.args.debug:
            print(f"Pipeline state saved to {path}")

    @staticmethod
    def load_state(path):
        """Loads a pipeline state from a file."""
        with open(path, "rb") as f:
            return pickle.load(f)


def run_analysis(args):
    """
    Main entry point for the analysis pipeline.
    """
    try:
        if args.load_state_from:
            pipeline = Pipeline.load_state(args.load_state_from)
            pipeline.args = args  # Update args in case they changed
        else:
            pipeline = Pipeline(args)
            pipeline.load_project_data()

        if args.image:
            if os.path.isdir(args.image):
                # Batch mode doesn't support interactive report saving
                results = []
                for filename in os.listdir(args.image):
                    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                        image_path = os.path.join(args.image, filename)
                        pipeline.process_image(image_path)
                        results.append(pipeline.analysis_results)
                return None # Return None as batch mode doesn't return a single pipeline
            else:
                part_number = getattr(args, 'part_number', None)
                thickness = getattr(args, 'thickness', None)
                author = getattr(args, 'author', None)
                department = getattr(args, 'department', None)
                report_title = getattr(args, 'report_title', None)
                pipeline.process_image(
                    args.image, part_number=part_number, thickness=thickness, author=author, department=department, report_title=report_title
                )
                if args.save_state_to:
                    pipeline.save_state(args.save_state_to)
                return pipeline # Return the whole instance

        if args.video:
            process_video(args, pipeline)

        return None

    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def process_video(args, pipeline):
    # This function needs to be adapted to the new class structure
    if args.debug:
        print(f"Processing video: {args.video}")
    # ... (original process_video logic adapted to use the pipeline object)
