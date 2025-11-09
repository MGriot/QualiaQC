"""
This module provides the AdvancedAligner class for robust image alignment
using various computer vision techniques.

It supports feature-based alignment (ORB/SIFT), a robust method based on
contour pose estimation, and a new default geometric method that prioritizes
a 5-point pentagon and falls back to a 4-point quadrilateral.
"""

import cv2
import numpy as np
import os
import math


def save_image(path, image):
    """Saves an image to a specified path, creating directories if needed."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, image)
    except Exception as e:
        print(f"Error saving image to {path}: {e}")


class AdvancedAligner:
    """
    A comprehensive image aligner that offers various alignment methods.

    The new default method, 'geometric_shape', attempts to align using a 5-point
    pentagon circumscribed around the object, falling back to a 4-point
    quadrilateral if a pentagon is not detected.
    """

    def __init__(
        self,
        max_features=2000,
        min_contour_area=100,
        poly_epsilon_ratio=0.02,
        debug_mode=False,
        output_dir=None,
        default_align_method="geometric_shape",
        shadow_removal_method="clahe",
    ):
        self.max_features = max_features
        self.min_contour_area = min_contour_area
        self.poly_epsilon_ratio = poly_epsilon_ratio
        self.debug_mode = debug_mode
        self.output_dir = output_dir
        self.default_align_method = default_align_method
        self.shadow_removal_method = shadow_removal_method

        if self.debug_mode and not self.output_dir:
            raise ValueError("output_dir must be provided when debug_mode is True.")

        self.orb = (
            cv2.ORB_create(self.max_features) if hasattr(cv2, "ORB_create") else None
        )
        self.sift = (
            cv2.SIFT_create(self.max_features) if hasattr(cv2, "SIFT_create") else None
        )

    def _save_debug_image(self, name, image, debug_paths):
        if self.debug_mode and self.output_dir:
            path = os.path.join(self.output_dir, f"debug_{name}.png")
            save_image(path, image)
            debug_paths[name] = path

    def _apply_clahe_contrast(self, img):
        if len(img.shape) != 3:
            return img
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def _apply_simple_gamma(self, img, gamma=1.5):
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        return cv2.LUT(img, table)

    def _find_largest_contour(self, img):
        """
        Improved contour detection:
        - Removes image border
        - Filters by area, bounding box ratio, and perimeter ratio
        - Falls back to second-largest contour if first is likely a frame
        """
        gray = (
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        )
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Remove border frame
        cv2.rectangle(
            binary, (0, 0), (binary.shape[1] - 1, binary.shape[0] - 1), 0, thickness=5
        )

        # Morphological closing
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        valid_contours = sorted(
            [c for c in contours if cv2.contourArea(c) > self.min_contour_area],
            key=cv2.contourArea,
            reverse=True,
        )

        if not valid_contours:
            return None

        image_area = img.shape[0] * img.shape[1]
        largest = valid_contours[0]
        largest_area = cv2.contourArea(largest)

        # Check if largest contour is likely the frame
        x, y, w, h = cv2.boundingRect(largest)
        if (largest_area / image_area) > 0.95 or (
            w > 0.95 * img.shape[1] and h > 0.95 * img.shape[0]
        ):
            if len(valid_contours) > 1:
                if self.debug_mode:
                    print(
                        "[DEBUG] Largest contour is likely frame. Using second largest."
                    )
                return valid_contours[1]

        return largest

    def _order_points_quad(self, pts):
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _order_polygon_points(self, pts):
        pts = pts.reshape(-1, 2)
        centroid = np.mean(pts, axis=0)
        sorted_pts = sorted(
            pts, key=lambda p: np.arctan2(p[1] - centroid[1], p[0] - centroid[0])
        )
        return np.array(sorted_pts, dtype="float32")

    def align_by_geometric_shape(self, src_processed, ref_processed):
        debug_paths = {}
        src_contour = self._find_largest_contour(src_processed)
        ref_contour = self._find_largest_contour(ref_processed)
        if src_contour is None or ref_contour is None:
            raise RuntimeError(
                "Could not find a dominant contour in one or both images."
            )

        src_epsilon = self.poly_epsilon_ratio * cv2.arcLength(src_contour, True)
        ref_epsilon = self.poly_epsilon_ratio * cv2.arcLength(ref_contour, True)
        src_poly = cv2.approxPolyDP(src_contour, src_epsilon, True)
        ref_poly = cv2.approxPolyDP(ref_contour, ref_epsilon, True)

        print(f"[INFO] Source polygon approximation has {len(src_poly)} vertices.")
        print(f"[INFO] Reference polygon approximation has {len(ref_poly)} vertices.")

        if len(src_poly) == 5 and len(ref_poly) == 5:
            src_pts = self._order_polygon_points(src_poly)
            ref_pts = self._order_polygon_points(ref_poly)
            shape_used = "pentagon"
        else:
            src_rect = cv2.minAreaRect(src_contour)
            ref_rect = cv2.minAreaRect(ref_contour)
            src_pts = self._order_points_quad(cv2.boxPoints(src_rect))
            ref_pts = self._order_points_quad(cv2.boxPoints(ref_rect))
            shape_used = "quadrilateral"

        src_debug, ref_debug = src_processed.copy(), ref_processed.copy()
        cv2.polylines(src_debug, [src_pts.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.polylines(ref_debug, [ref_pts.astype(np.int32)], True, (0, 255, 0), 2)
        self._save_debug_image(f"03_geom_src_{shape_used}", src_debug, debug_paths)
        self._save_debug_image(f"03_geom_ref_{shape_used}", ref_debug, debug_paths)

        M, _ = cv2.findHomography(src_pts, ref_pts, cv2.RANSAC, 5.0)
        if M is None:
            raise RuntimeError("Homography computation failed with geometric points.")
        return M, debug_paths

    def align_by_feature(self, src_processed, ref_processed, use_sift=False):
        debug_paths = {}
        detector = self.sift if use_sift and self.sift else self.orb
        if detector is None:
            raise RuntimeError(
                f"Feature detector {'SIFT' if use_sift else 'ORB'} not available."
            )
        gray_src = cv2.cvtColor(src_processed, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.cvtColor(ref_processed, cv2.COLOR_BGR2GRAY)
        kp1, des1 = detector.detectAndCompute(gray_src, None)
        kp2, des2 = detector.detectAndCompute(gray_ref, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return None, {}

        norm = cv2.NORM_L2 if use_sift and self.sift else cv2.NORM_HAMMING
        matcher = cv2.BFMatcher(norm, crossCheck=False)
        matches = matcher.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        img_matches = cv2.drawMatches(
            src_processed, kp1, ref_processed, kp2, good_matches, None
        )
        self._save_debug_image("02_feature_matches", img_matches, debug_paths)

        if len(good_matches) < 10:
            return None, debug_paths

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return M, debug_paths

    def align_by_contour_pose(self, src_processed, ref_processed):
        debug_paths = {}
        gray_src = cv2.cvtColor(src_processed, cv2.COLOR_BGR2GRAY)
        binary_src = cv2.adaptiveThreshold(
            gray_src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        contours_src, _ = cv2.findContours(
            binary_src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        src_contour = max(
            [c for c in contours_src if cv2.contourArea(c) > self.min_contour_area],
            key=cv2.contourArea,
        )
        M_src = cv2.moments(src_contour)
        src_centroid = (
            int(M_src["m10"] / M_src["m00"]),
            int(M_src["m01"] / M_src["m00"]),
        )

        gray_ref = cv2.cvtColor(ref_processed, cv2.COLOR_BGR2GRAY)
        binary_ref = cv2.adaptiveThreshold(
            gray_ref, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        contours_ref, _ = cv2.findContours(
            binary_ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        ref_contour = max(
            [c for c in contours_ref if cv2.contourArea(c) > self.min_contour_area],
            key=cv2.contourArea,
        )
        M_ref = cv2.moments(ref_contour)
        ref_centroid = (
            int(M_ref["m10"] / M_ref["m00"]),
            int(M_ref["m01"] / M_ref["m00"]),
        )

        src_rect = cv2.minAreaRect(src_contour)
        ref_rect = cv2.minAreaRect(ref_contour)
        src_angle = src_rect[2] + (90 if src_rect[1][0] < src_rect[1][1] else 0)
        ref_angle = ref_rect[2] + (90 if ref_rect[1][0] < ref_rect[1][1] else 0)
        angle_diff = ref_angle - src_angle
        src_area = max(1, src_rect[1][0] * src_rect[1][1])
        ref_area = ref_rect[1][0] * ref_rect[1][1]
        scale = math.sqrt(ref_area / src_area)

        M = cv2.getRotationMatrix2D(src_centroid, angle_diff, scale)
        rotated_src_centroid = M @ np.array([src_centroid[0], src_centroid[1], 1])
        M[0, 2] += ref_centroid[0] - rotated_src_centroid[0]
        M[1, 2] += ref_centroid[1] - rotated_src_centroid[1]

        return M, debug_paths

    def align(self, src, ref, method=None, shadow_removal=None):
        method = method or self.default_align_method
        shadow_method = shadow_removal or self.shadow_removal_method
        debug_paths = {}

        self._save_debug_image("00_input_source", src, debug_paths)
        self._save_debug_image("00_input_reference", ref, debug_paths)

        try:
            src_processed, ref_processed = src.copy(), ref.copy()
            if shadow_method == "clahe":
                src_processed = self._apply_clahe_contrast(src_processed)
                ref_processed = self._apply_clahe_contrast(ref_processed)
            elif shadow_method == "gamma":
                src_processed = self._apply_simple_gamma(src_processed)
                ref_processed = self._apply_simple_gamma(ref_processed)

            M, dbg = None, {}
            print(f"[INFO] Attempting alignment with method: '{method}'")

            if method == "geometric_shape":
                M, dbg = self.align_by_geometric_shape(src_processed, ref_processed)
            elif method == "contour_pose":
                M, dbg = self.align_by_contour_pose(src_processed, ref_processed)
            elif method == "feature_sift":
                M, dbg = self.align_by_feature(
                    src_processed, ref_processed, use_sift=True
                )
            elif method == "feature_orb":
                M, dbg = self.align_by_feature(
                    src_processed, ref_processed, use_sift=False
                )
            else:
                raise ValueError(f"Unknown alignment method: {method}")

            debug_paths.update(dbg)

        except Exception as e:
            print(f"[Alignment Error] Method '{method}' failed: {e}")
            return {"image": None, "debug_paths": debug_paths, "transform_matrix": None}

        if M is None:
            print(
                f"[Alignment Warning] Method '{method}' could not compute a transformation matrix."
            )
            return {"image": None, "debug_paths": debug_paths, "transform_matrix": None}

        h, w = ref.shape[:2]
        warp_func = cv2.warpAffine if M.shape[0] == 2 else cv2.warpPerspective
        aligned_image = warp_func(src, M, (w, h))
        self._save_debug_image(f"04_final_aligned_{method}", aligned_image, debug_paths)

        return {
            "image": aligned_image,
            "debug_paths": debug_paths,
            "transform_matrix": M,
        }
