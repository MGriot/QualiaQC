# PDF/Image Region Color & Vector Editor

This document describes the features and underlying mechanisms of the interactive GUI for editing and coloring regions in PDFs and images.

---

## ✅ What you’ll need

```bash
pip install pymupdf pillow numpy opencv-python
```

> **Note**: This tool heavily leverages `pymupdf`, `Pillow`, `numpy`, and `opencv-python`.

---

## 💻 The Code (save as `drawing_tools/gui_main.py`)

### 🧠 How it works (under the hood)

*   **Render**: Uses PyMuPDF to render selected PDF pages at 2× scale for sharper edges. Images are loaded directly.
*   **Detect lines & regions**:
    *   Converts the image to grayscale.
    *   For **line detection**, it uses the **Line Segment Detector (LSD)** from OpenCV for fast and accurate vector extraction. This replaces older, less precise skeletonization methods.
    *   For **region segmentation**, it applies Otsu thresholding to create a binary mask of lines. These lines are then optionally dilated by a tunable amount to seal small gaps, preventing “leaks.”
    *   The inverted mask (fillable areas) is then processed using **OpenCV's connected components labeling** to identify and map all closed regions.
    *   Regions touching the image border are marked as background (not fillable).
*   **Click-to-fill**:
    *   Maps your click to the underlying region ID and paints that area onto a transparent overlay using your chosen color (RGBA with alpha).
    *   The display is a live composite: base image + overlay.
*   **Vector Editing**: Allows selection, grouping, transformation (move, rotate), and connection of detected line segments. New fillable areas can be created from selected vectors, with an option to define holes.
*   **Export**:
    *   Region PNG: A transparent image with only that region colored.
    *   All regions: Batch-exports every colored region.
    *   Composite PNG: Saves the full page with overlay applied.
    *   Region PDF: Embeds the transparent PNG in a single‑page PDF (same pixel dimensions).

### 🧩 Notes, Tips & Adjustments

*   **Gaps in lines?** Use `Options → Set line dilation` (increase from 2 to 3–4) to seal tiny leaks before segmentation.
*   **White-on-black drawings?** Use `Options → Toggle invert lines` if your source has light lines on a dark background.
*   **Performance**: Significant improvements have been made to line detection and region segmentation using optimized OpenCV algorithms (LSD and `connectedComponents`), making the process much faster and more precise.
*   **Background Visibility**: In "Edit Vectors" mode, use the "Show Background" checkbox in the toolbar to toggle the visibility of the original image/PDF, allowing you to focus solely on the vectors.
*   **Custom Colors**:
    *   `Options → Set Vector Color...`: Customize the colors used for default, selected, and grouped vector lines.
    *   `Options → Set Fill Opacity...`: Set the default opacity (alpha value) for new fill groups.
    *   In the "Color Groups" panel, select a group and click "Set Color" to change its specific color and opacity.

### ▶️ Usage

1.  Run: `python -m src.gui` (or `python src/drawing_tools/gui_main.py` if running directly)
2.  `File → Open PDF...` or `File → Open Image...`
3.  Switch to "Fill Area" mode and click inside any closed area to fill it.
4.  Switch to "Edit Vectors" mode to select, group, and manipulate detected lines.
5.  Customize colors and other options via the `Edit` or `Options` menus, or directly from the UI panels.
6.  Export via `File` menu.
