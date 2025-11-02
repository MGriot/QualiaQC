"""
This module provides the `ColorAnalyzer` class for identifying and quantifying
color zones within images based on specified HSV color ranges.

It includes functionalities for finding color zones, aggregating masks, and
calculating statistics on matched pixels.
"""

import os
import cv2
import numpy as np
from typing import Tuple
import datetime

from src.utils.image_utils import load_image, save_image
from src.geometric_alignment.geometric_aligner import Aligner


class ColorAnalyzer:
    """
    Analyzes images or video frames to find zones matching a specified HSV color range.
    Generates masks, negative images, and calculates statistics on the matched areas.
    """

    def __init__(self):
        """
        Initializes the ColorAnalyzer.
        """
        pass

    def find_color_zones(
        self,
        image: np.ndarray,
        lower_hsv: np.ndarray,
        upper_hsv: np.ndarray,
        alpha_channel: np.ndarray = None,
        debug_mode: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identifies and extracts color zones within an image that fall within a specified HSV range.

        Args:
            image (np.ndarray): The input image in BGR format.
            lower_hsv (np.ndarray): A NumPy array representing the lower bounds of the HSV color range.
            upper_hsv (np.ndarray): A NumPy array representing the upper bounds of the HSV color range.
            alpha_channel (np.ndarray, optional): An optional alpha channel mask. If provided,
                                                 color zone detection will only occur within non-transparent areas.
            debug_mode (bool, optional): If True, prints debug information to the console.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - mask (np.ndarray): A binary mask where pixels within the HSV range are white (255)
                                     and others are black (0).
                - negative_mask (np.ndarray): The inverse of the `mask`.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

        if alpha_channel is not None:
            mask = cv2.bitwise_and(mask, mask, mask=alpha_channel)
            negative_mask = cv2.subtract(alpha_channel, mask)
        else:
            negative_mask = cv2.bitwise_not(mask)

        if debug_mode:
            print(
                f"[DEBUG] Color analysis performed with HSV range: {lower_hsv} - {upper_hsv}"
            )

        return mask, negative_mask

    def _aggregate_mask_improved(
        self,
        mask: np.ndarray,
        kernel_size: int,
        min_area_ratio: float,
        agg_density_thresh: float,
        max_area_ratio: float = None,
        debug_mode: bool = False,
    ) -> np.ndarray:
        """
        Aggregates nearby matched pixel areas in a binary mask to form larger, more coherent regions.

        This method applies dilation to connect close components, then filters these components
        based on their size and the density of original matched pixels within them.

        Args:
            mask (np.ndarray): The input binary mask (single channel, 0 or 255).
            kernel_size (int): The size of the kernel used for dilation. Larger values connect
                               components further apart.
            min_area_ratio (float): The minimum area a connected component must have, as a ratio
                                    of the total image area, to be considered for aggregation.
            agg_density_thresh (float): The minimum density (0.0-1.0) of original matched pixels
                                        within an aggregated component for it to be kept. This
                                        prevents over-aggregation of sparse regions.
            max_area_ratio (float, optional): The maximum area a connected component can have, as a ratio
                                            of the total image area, to be kept. Defaults to 0.1 (10% of image area).
            debug_mode (bool, optional): If True, prints debug information to the console.

        Returns:
            np.ndarray: The aggregated binary mask.
        """
        if debug_mode:
            print(
                f"[DEBUG] Improved aggregating mask with kernel_size={kernel_size}, min_area_ratio={min_area_ratio}, density_thresh={agg_density_thresh}"
            )

        dilate_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_mask = cv2.dilate(mask, dilate_kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dilated_mask, 8, cv2.CV_32S
        )

        aggregated_mask = np.zeros_like(mask)
        total_image_area = mask.shape[0] * mask.shape[1]

        for i in range(1, num_labels):
            component_area = stats[i, cv2.CC_STAT_AREA]

            # Max area check
            if max_area_ratio is not None and component_area > total_image_area * max_area_ratio:
                if debug_mode:
                    print(f"[DEBUG]   Filtered out component {i} with area {component_area} (too large).")
                continue

            if component_area >= total_image_area * min_area_ratio:
                component_mask = (labels == i).astype(np.uint8) * 255

                # Density Check
                original_pixels_in_component = cv2.countNonZero(
                    cv2.bitwise_and(mask, component_mask)
                )
                density = (
                    original_pixels_in_component / component_area
                    if component_area > 0
                    else 0
                )

                if density >= agg_density_thresh:
                    contours, _ = cv2.findContours(
                        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contours:
                        cv2.drawContours(aggregated_mask, contours, -1, 255, cv2.FILLED)
                    if debug_mode:
                        print(
                            f"[DEBUG]   Kept component {i} with area {component_area} and density {density:.2f}."
                        )
                else:
                    if debug_mode:
                        print(
                            f"[DEBUG]   Rejected component {i} with area {component_area} due to low density ({density:.2f} < {agg_density_thresh})."
                        )
            else:
                if debug_mode:
                    print(
                        f"[DEBUG]   Filtered out component {i} with area {component_area} (too small)."
                    )

        final_aggregated_mask = cv2.bitwise_or(aggregated_mask, mask)

        if debug_mode:
            print(
                f"[DEBUG] Improved aggregation complete. Original matched pixels: {cv2.countNonZero(mask)}, Aggregated matched pixels: {cv2.countNonZero(final_aggregated_mask)}"
            )

        return final_aggregated_mask

    def calculate_statistics(
        self, mask: np.ndarray, total_pixels: int, debug_mode: bool = False
    ) -> Tuple[float, int]:
        """
        Calculates the percentage and total count of matched pixels within a given mask.

        Args:
            mask (np.ndarray): The binary mask (single channel, 0 or 255) representing matched pixels.
            total_pixels (int): The total number of pixels in the image (or region of interest).
            debug_mode (bool, optional): If True, prints debug information to the console.

        Returns:
            Tuple[float, int]: A tuple containing:
                - percentage (float): The percentage of matched pixels relative to the total pixels.
                - matched_pixels (int): The absolute count of matched pixels.
        """
        matched_pixels = cv2.countNonZero(mask)
        percentage = (matched_pixels / total_pixels) * 100 if total_pixels > 0 else 0

        if debug_mode:
            print(f"[DEBUG] Matched Pixels: {matched_pixels}")
            print(f"[DEBUG] Total Pixels (non-transparent): {total_pixels}")
            print(f"[DEBUG] Percentage of Matched Pixels: {percentage:.2f}%")

        return percentage, matched_pixels

    def process_image(
        self,
        image: np.ndarray,
        original_image_path: str,
        lower_hsv: np.ndarray,
        upper_hsv: np.ndarray,
        center_hsv: np.ndarray,
        output_dir: str,
        debug_mode: bool = False,
        aggregate_mode: bool = False,
        agg_kernel_size: int = 7,
        agg_min_area: float = 0.0005,
        agg_max_area: float = None,
        agg_density_thresh: float = 0.5,
        use_alpha: bool = True,
    ) -> dict:
        """
        Processes a single image to perform color analysis based on a specified HSV range.

        This function finds color zones, optionally aggregates them, calculates statistics,
        and saves various output images.

        Args:
            image (np.ndarray): The input image in BGR or BGRA format that needs to be analyzed.
            original_image_path (str): The file path to the original input image, used for metadata.
            lower_hsv (np.ndarray): A NumPy array representing the lower bounds of the HSV color range.
            upper_hsv (np.ndarray): A NumPy array representing the upper bounds of the HSV color range.
            center_hsv (np.ndarray): A NumPy array representing the center of the HSV color range.
            output_dir (str): The directory where output images and masks will be saved.
            debug_mode (bool, optional): If True, prints debug information and saves intermediate masks.
            aggregate_mode (bool, optional): If True, aggregates matched pixel areas.
            agg_kernel_size (int, optional): Kernel size for aggregation dilation. Defaults to 7.
            agg_min_area (float, optional): Minimum area ratio for aggregation components. Defaults to 0.0005.
            agg_max_area (float, optional): Maximum area ratio for aggregation components. Defaults to 0.1 (10% of image area).
            agg_density_thresh (float, optional): Minimum density for aggregated areas. Defaults to 0.5.
            use_alpha (bool, optional): If True, handles images with an alpha channel (4 dimensions).
                                        If False, processes as a 3-channel image. Defaults to True.

        Returns:
            dict: A dictionary containing various results of the analysis.

        Raises:
            ValueError: If required arguments like `image`, `lower_hsv`, `upper_hsv`, or `output_dir` are not provided.
        """
        if image is None:
            raise ValueError("ColorAnalyzer.process_image requires a valid image (np.ndarray). It received None.")

        alignment_data = None
        input_image = image
        analysis_mask = None

        # Check for alpha channel and separate it if present and enabled
        has_alpha = use_alpha and image.shape[2] == 4
        if has_alpha:
            alpha_channel = image[:, :, 3]
            image_for_analysis = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
            # Create a mask of only solid pixels (alpha == 255) to exclude non-solid pixels from analysis.
            analysis_mask = (alpha_channel == 255).astype(np.uint8) * 255
            total_pixels = np.count_nonzero(analysis_mask)
        else:
            image_for_analysis = image if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            total_pixels = image_for_analysis.shape[0] * image_for_analysis.shape[1]


        if lower_hsv is None or upper_hsv is None:
            raise ValueError("lower_hsv and upper_hsv must be provided.")
        if output_dir is None:
            raise ValueError("output_dir must be provided.")

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

        input_to_analysis_path = os.path.join(
            output_dir, f"input_to_color_analysis_{timestamp}.png"
        )
        save_image(input_to_analysis_path, image_for_analysis)

        mask, negative_mask = self.find_color_zones(
            image_for_analysis,
            lower_hsv,
            upper_hsv,
            analysis_mask,
            debug_mode=debug_mode,
        )

        mask_pre_aggregation_path = None
        if aggregate_mode:
            mask_pre_aggregation_path = os.path.join(
                output_dir, f"mask_pre_aggregation_{timestamp}.png"
            )
            save_image(mask_pre_aggregation_path, mask)
            if debug_mode:
                print(
                    f"[DEBUG] Mask before aggregation saved to {mask_pre_aggregation_path}"
                )
            mask = self._aggregate_mask_improved(
                mask,
                kernel_size=agg_kernel_size,
                min_area_ratio=agg_min_area,
                agg_density_thresh=agg_density_thresh,
                max_area_ratio=agg_max_area,
                debug_mode=debug_mode,
            )

            # After aggregation, the mask can expand into transparent areas.
            # We must clip it to the analysis_mask to ensure correct statistics.
            if analysis_mask is not None:
                mask = cv2.bitwise_and(mask, analysis_mask)

            # Recalculate the negative mask based on the final, clipped mask.
            if analysis_mask is not None:
                negative_mask = cv2.subtract(analysis_mask, mask)
            else:
                negative_mask = cv2.bitwise_not(mask)

        percentage, matched_pixels = self.calculate_statistics(
            mask, total_pixels, debug_mode=debug_mode
        )

        processed_image_path = os.path.join(
            output_dir, f"processed_image_{timestamp}.png"
        )
        mask_path = os.path.join(output_dir, f"mask_{timestamp}.png")
        negative_mask_path = os.path.join(output_dir, f"negative_mask_{timestamp}.png")

        # Create the blacked-out visualization from the correct input image.
        processed_image = input_image.copy()
        if has_alpha:
            processed_image[mask == 0, :3] = 0  # Blackout BGR, keep alpha
        else:
            processed_image[mask == 0] = [0, 0, 0]

        save_image(processed_image_path, processed_image)
        save_image(mask_path, mask)
        save_image(negative_mask_path, negative_mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours on the correct input image to ensure visualizations are consistent with the pipeline.
        image_with_contours = input_image.copy()
        # Use a 4-component color for BGRA images to be explicit
        contour_color = (0, 0, 255, 255) if has_alpha else (0, 0, 255)
        cv2.drawContours(image_with_contours, contours, -1, contour_color, 2)
        contours_image_path = os.path.join(output_dir, f"contours_{timestamp}.png")
        save_image(contours_image_path, image_with_contours)

        # Convert center HSV to RGB for reporting
        center_rgb = [0, 0, 0]
        if center_hsv is not None:
            center_rgb = cv2.cvtColor(np.uint8([[center_hsv]]), cv2.COLOR_HSV2RGB)[0][0]

        # --- Prepare Debug Information ---
        debug_info_list = []
        if debug_mode:
            # The order here determines the order in the report
            debug_info_list.append({'title': 'Input to Color Analysis', 'path': input_to_analysis_path})
            if mask_pre_aggregation_path:
                debug_info_list.append({'title': 'Mask Before Aggregation', 'path': mask_pre_aggregation_path})
            debug_info_list.append({'title': 'Final Color Mask', 'path': mask_path})
            debug_info_list.append({'title': 'Negative Mask', 'path': negative_mask_path})
            debug_info_list.append({'title': 'Image with Contours', 'path': contours_image_path})
            debug_info_list.append({'title': 'Final Processed Image (Blacked Out)', 'path': processed_image_path})

        return {
            "original_image": input_image,
            "processed_image": processed_image,
            "mask": mask,
            "binary_mask": mask,  # Add explicit key for symmetry analysis consistency
            "negative_mask": negative_mask,
            "original_image_path": original_image_path,
            "input_to_analysis_path": input_to_analysis_path,
            "processed_image_path": processed_image_path,
            "mask_path": mask_path,
            "negative_mask_path": negative_mask_path,
            "contours_image_path": contours_image_path,
            "percentage": percentage,
            "matched_pixels": matched_pixels,
            "total_pixels": total_pixels,
            "mask_pre_aggregation_path": mask_pre_aggregation_path,
            "alignment_data": alignment_data,
            "lower_limit": lower_hsv,
            "upper_limit": upper_hsv,
            "center_color": center_hsv,
            "selected_colors": [
                {
                    "color_name": "Selected Area",
                    "hsv": center_hsv.tolist() if center_hsv is not None else [0, 0, 0],
                    "rgb": (
                        center_rgb.tolist()
                        if isinstance(center_rgb, np.ndarray)
                        else center_rgb
                    ),
                }
            ],
            "debug_info": debug_info_list
        }
