# QualiaQC

## Project Description

QualiaQC (Qualia, the qualities of a thing + QC for Quality Control) is a powerful Python-based tool for advanced image and video analysis. It provides a comprehensive suite of features for color correction, color zone analysis, image alignment, and symmetry analysis. The tool is designed to be project-based, allowing users to manage different analysis tasks with specific configurations and reference materials. The application is accessible through a Tkinter-based Graphical User Interface (GUI) and a Command-Line Interface (CLI).

## Core Concepts & Theory

For a deeper, university-level explanation of the core algorithms used in this project, please see the documents below:

-   **[Theory: Color Correction](./docs/THEORY_ColorCorrection.md)**: A detailed guide on the tiered patch detection system and the robust, cost-matrix-based patch matching algorithm.
-   **[Theory: Image Alignment](./docs/THEORY_Alignment.md)**: An in-depth look at the two-stage alignment process, covering both the projective geometry of ArUco markers and the contour-based methods for object alignment.
-   **[Theory: Symmetry Analysis](./docs/THEORY_Symmetry.md)**: An explanation of the geometric principles behind the reflection, rotation, translation, and glide-reflection symmetry analyses.

## Features

*   **Modular Analysis Pipeline**: Sequentially run color correction, geometric alignment, object alignment, masking, and color/symmetry analysis.
*   **Extensible Project Management**: Each analysis is configured via a project-specific `project_config.json`.
*   **Robust Color Correction**: Utilizes a tiered patch detection strategy (ArUco, OpenCV, YOLO) and robust color-matching for stable color correction.
*   **Comprehensive Reporting**: Automatically generates detailed PDF and HTML reports with plots, metadata, and visual pipeline steps.
*   **Full-Featured GUI**: A Tkinter-based GUI to manage projects, datasets, and run the entire analysis pipeline.
*   **Report History & Regeneration**: A "History & Reports" tab in the GUI to scan, view, filter, and regenerate past analysis reports from archives.
*   **Advanced Dataset Management**: Includes a "File Placer" to easily manage project assets and a training image manager with previews and delete functionality.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/VisualAnalyzer.git
    cd VisualAnalyzer
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**
    *   **Windows:**
        ```bash
        .\.venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```

4.  **Install the required dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## Configuration

The Visual Analyzer uses a project-based configuration system. Each project is a directory in the `data/projects` folder. Use the GUI (`python src/gui.py`) to scaffold a new project via the "Create Project" tab.

### `project_config.json`

This file defines the core settings for your project. When a project is created, it is pre-filled with default paths. Paths are relative to the project's root directory.

```json
{
    "training_path": "dataset/training_images",
    "object_reference_path": "dataset/object_reference.png",
    "color_correction": {
        "reference_color_checker_path": "dataset/default_color_checker_reference.png",
        "project_specific_color_checker_path": "dataset/project_color_checker.png"
    },
    "geometrical_alignment": {
        "reference_path": "dataset/default_geometric_align_reference.png",
        "marker_map": {},
        "output_size": [
            1000,
            1000
        ]
    },
    "masking": {
        "drawing_layers": {
            "1": "dataset/drawing_layers/layer1.png",
            "2": "dataset/drawing_layers/layer2.png",
            "3": "dataset/drawing_layers/layer3.png"
        }
    }
}
```

*   `training_path`: Path to the directory of images for calculating the target color range.
*   `object_reference_path`: Path to the reference image for object alignment.
*   `color_correction`: Object containing paths for color correction.
    *   `reference_color_checker_path`: Path to the ideal, canonical color checker image (auto-generated).
    *   `project_specific_color_checker_path`: Path to a photo of the color checker taken in the project's specific lighting. **You must provide this file.**
*   `geometrical_alignment`: Object for ArUco-based alignment settings.
*   `masking`: Object containing paths to drawing layers for masking.

### `dataset_item_processing_config.json`

This file defines how sample images in the `training_path` are processed for color range calculation. You can edit this manually or use the interactive GUI (`python src/gui.py` -> "Manage Dataset" tab).

```json
{
    "image_configs": [
        {
            "filename": "sample1.png",
            "method": "full_average"
        },
        {
            "filename": "sample2.png",
            "method": "points",
            "points": [
                {"x": 100, "y": 150, "radius": 7}
            ]
        }
    ]
}
```

## Usage

### Launching the Application

The application can be launched either via a Graphical User Interface (GUI) or through the Command-Line Interface (CLI).

*   **GUI:** To start the GUI, run `src/gui.py`.
    ```bash
    python src/gui.py

    #or

    .\.venv\Scripts\python.exe -m src.gui --debug
    ```
    The GUI provides an intuitive way to create projects, manage reference files, and configure analysis options.

*   **CLI:** To run the analysis directly from the command line, provide the necessary arguments to `main.py`:
    ```bash
    python src/main.py --project <project_name> --image <path_to_image> [options]
    ```

### Command-Line Options:

*   `--project <project_name>`: **(Required)** The name of the project to use.
*   `--image <path>`: Path to a single image file or a directory of images for analysis.
*   `--video <path>`: Path to a video file for analysis.
*   `--camera`: Use a live camera stream for analysis.
*   `--debug`: Enable debug mode for verbose output and detailed reports.
*   `--color-alignment`: Enable color correction. 
*   `--sample-color-checker <path>`: Path to a color checker image taken with the sample. If provided, it is used for on-the-fly color correction for this specific run.
*   `--color-correction-method <method>`: Specify the algorithm for color correction. Choices: `linear`, `polynomial`, `hsv`, `histogram`. Default is `linear`.
*   `--alignment`: Enable geometrical (ArUco) alignment.
*   `--object-alignment`: Enable object alignment.
*   `--object-alignment-shadow-removal <method>`: Shadow removal method for object alignment. Choices: `clahe` (default), `gamma`, `none`.
*   `--apply-mask`: Enable background removal using technical drawing(s).
*   `--mask-bg-is-white`: When using `--apply-mask`, treat white pixels in the drawing as background.
*   `--masking-order <order>`: Specify the order of masking layers (e.g., '1-2-3', '3-1-2'). Default is '1-2-3'.
*   `--symmetry`: Enable symmetry analysis.
*   `--blur`: Enable blurring of the input image.
*   `--blur-kernel <W> <H>`: Specify a custom kernel size (width height) for blurring. Both values must be odd.
*   `--aggregate`: Enable aggregation of matched pixel areas.
*   `--agg-kernel-size <size>`: Kernel size for the aggregation dilation step (default: 7).
*   `--agg-min-area <ratio>`: Minimum area ratio for keeping a component during aggregation (default: 0.0005).
*   `--agg-density-thresh <threshold>`: Minimum density of original pixels for an aggregated area to be kept (default: 0.5).
*   `--skip-color-analysis`: Skip the color analysis step.
*   `--skip-report-generation`: Skip the final report generation and archiving step.
*   `--save-state-to <path>`: Path to save the entire pipeline state to a `.gri` file for later regeneration.
*   `--load-state-from <path>`: Path to load a previously saved pipeline state from a `.gri` file.
