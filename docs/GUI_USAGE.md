# Visual Analyzer - GUI Usage Guide

This document provides a detailed walkthrough of all features available in the main Visual Analyzer GUI, which is the recommended way to interact with the tool.

To launch the application, run:
```bash
# Make sure your virtual environment is active
.venv\Scripts\activate.bat

# Run the gui module (add --debug to see all options)
python -m src.gui --debug
```

## Main Tabs

The GUI is organized into four main tabs:

1.  **Run Analysis**: The main screen for configuring and executing a single analysis run.
2.  **History & Reports**: A powerful tool for viewing past analyses and regenerating reports.
3.  **Create Project**: A simple utility to scaffold a new project directory.
4.  **Manage Dataset**: Tools for managing project files and defining color sample points.

---

## 1. Run Analysis Tab

This is the primary tab for running the analysis pipeline on a single image.

![Run Analysis Tab](placeholder.png) *<-- Placeholder for a screenshot of the analysis tab -->*

### Configuration

1.  **Project Name**: Select the project you are working on. The dropdown is populated from the folders in `data/projects/`.
2.  **Select Image**: Click to choose the image file you want to analyze.
3.  **Part Number & Thickness**: These fields are automatically filled by parsing the filename of the selected image. You can manually edit them to override the values for the analysis and report.
4.  **Select Color Checker**: If using Color Alignment, you must select the image of the color checker taken under the same lighting conditions as your sample image.

### Analysis Steps & Options

If you run the GUI in `--debug` mode, a comprehensive set of options appears, allowing you to enable or disable every step of the pipeline (e.g., Color Alignment, Masking, Symmetry Analysis) and configure their parameters.

### Aggregation Parameters

These parameters control how detected color regions are grouped together.

*   **Agg. Kernel Size**: The size of the kernel used for dilation during aggregation. Larger values connect components further apart. (Default: 7)
*   **Agg. Min Area**: The minimum area a connected component must have, as a ratio of the total image area, to be considered for aggregation. (Default: 0.0005)
*   **Agg. Max Area**: The maximum area a connected component can have, as a ratio of the total image area, to be kept. This helps prevent over-aggregation of very large, irrelevant areas. (Default: 0.1, or 10% of image area)
*   **Agg. Density Thresh**: The minimum density (0.0-1.0) of original matched pixels within an aggregated component for it to be kept. This prevents over-aggregation of sparse regions. (Default: 0.5)

### Running the Analysis

1.  Click **"Run Analysis"**.
2.  A popup will confirm the analysis is starting.
3.  Once the image processing is complete, a **"Save As..."** dialog will appear, allowing you to choose where to save the final PDF report.
4.  If running in debug mode, you will then be asked if you want a "Normal" or full "Debug" report.

---

## 2. History & Reports Tab

This tab allows you to find, filter, and regenerate reports from all past analyses.

![History Tab](placeholder.png) *<-- Placeholder for a screenshot of the history tab -->*

### Features

1.  **Scan for Reports**: Click this button to scan the entire `output/` directory for analysis archives (`.gri` files). The table will be populated with the findings.
2.  **Filterable Table**: The main view is a table showing key metadata from each analysis. You can filter the results in real-time by typing into the entry boxes at the top of each column.
3.  **Sorting**: Click any column header to sort the results by that column.
4.  **Report Regeneration**:
    *   Select a single row in the table.
    *   The **"Recreate Selected Report"** button will become active.
    *   Click it, and a "Save As..." dialog will appear, allowing you to save a new copy of the PDF report.
    *   If you are in debug mode, you can also choose to regenerate as a "Normal" or "Debug" report.

---

## 3. Create Project Tab

This provides a simple form to create a new, empty project with the correct folder structure and default configuration files in the `data/projects/` directory.

---

## 4. Manage Dataset Tab

This tab contains tools to help set up your project's assets.

1.  **Launch Point Selector**: Opens a dedicated GUI for selecting specific points on your training images to define the target color space. This is an alternative to using the entire image for color calculation.
2.  **Setup Project Files**: Opens the **Project File Placer** GUI.

### Project File Placer Enhancements

The File Placer helps you copy required files (like reference images and drawing layers) into the correct locations. It has been enhanced with a **Training Image Manager**:

*   **Add Images**: Select one or more training images from anywhere on your computer to copy them into the project.
*   **Preview**: See thumbnails of all training images currently in the project.
*   **Delete**: A "Delete" button next to each image allows for easy removal.

---

## 5. PDF Region Color & Vector Editor

This is an interactive tool for segmenting regions, detecting lines, and applying color fills to PDF pages or image files.

### Launching the Editor

The editor is integrated into the main GUI.

### Key Features

*   **Open Files**: Use `File → Open PDF...` to load multi-page PDF documents, or `File → Open Image...` to load single image files (PNG, JPG, BMP, GIF).
*   **Interaction Modes**:
    *   **Fill Area**: Click inside closed regions to apply a color fill.
    *   **Edit Vectors**: Select, group, move, rotate, connect, and delete detected line segments.
*   **Color Groups**:
    *   Organize filled regions into named groups with customizable colors.
    *   `New`: Create a new color group.
    *   `Set Color`: Change the color and opacity of the currently selected group.
    *   `Rename`: Rename the selected group.
    *   `Delete`: Delete the selected group and its associated fills.
    *   `Remove Selected Region(s)`: Remove specific filled areas from their groups.
    *   `Merge Selected Regions`: Combine multiple adjacent filled regions into one.
*   **Vector Actions (in Edit Vectors mode)**:
    *   `Create Area from Selection`: Generate a new fillable region from selected line segments, with an option to `Create with Holes`.
    *   `Connect Endpoints`: Automatically connect nearby endpoints of selected vectors to form closed shapes.
    *   `Delete Selection`: Remove selected vector lines.
    *   `Clear Selection`: Deselect all lines.
    *   `Group Selection` / `Ungroup Selection`: Organize vectors into logical groups.
    *   `Transform Selection`: Move or rotate selected vectors by specified amounts.
*   **Display Options**:
    *   `Show Background`: Toggle the visibility of the original PDF page or image. When off, only the vectors and overlays are shown on a black background.
    *   `Show Overlays`: Toggle the visibility of all colored fill regions.
*   **Customization (Options Menu)**:
    *   `Set Vector Color...`: Customize the colors used for default, selected, and grouped vector lines.
    *   `Set Fill Opacity...`: Set the default opacity (alpha value, 0-255) for newly created fill groups.
*   **Segmentation Settings**:
    *   Adjust the `Threshold` for line detection (manual or auto).
    *   `Set line dilation...`: Control how much lines are thickened to close gaps.
    *   `Toggle invert lines (white-on-black)`: Adapt to documents with light lines on a dark background.
*   **Export Options**:
    *   Export selected regions, all colored regions, or the composited view as PNGs.
    *   Export a single selected region as a PDF.
