# Configuration Guide

This document explains the configuration system for the Visual Analyzer application. The system is divided into two main parts: a global configuration for application-wide settings, and project-specific configurations for tailoring the analysis pipeline to different datasets.

---

## Global Configuration (`src/config.py`)

The `src/config.py` file defines static, global settings that apply to the entire application. You can edit this file to change default behaviors and metadata.

Key variables include:

*   **Directory Paths**: Variables like `ROOT_DIR`, `DATA_DIR`, `PROJECTS_DIR`, and `OUTPUT_DIR` define the core directory structure. It is recommended not to change these unless you are restructuring the project.
*   **Report Metadata**: Default author and department information for generated reports.
    *   `AUTHOR`: The default author name (e.g., "Griot Matteo").
    *   `DEPARTMENT`: The default department name.
    *   `REPORT_TITLE`: The default title for analysis reports.
*   **Model Paths**: Paths to machine learning models, such as `YOLO_MODEL_PATH`.

---

## Project-Specific Configuration

Each project, located in its own sub-directory within `data/projects/`, has its own set of configuration files. This allows for highly customized analysis pipelines for different types of images or tasks.

The two main configuration files for a project are:

1.  `project_config.json`: The primary configuration file for the project.
2.  `dataset_item_processing_config.json`: An optional file to specify how individual images in the training dataset should be processed.

### `project_config.json`

This is the most important configuration file for a project. It defines all the necessary paths and parameters for the analysis pipeline. Below is an example with explanations for each field.

**Example `project_config.json`:**
```json
{
  "training_path": "training/",
  "object_reference_path": "reference/object_reference.png",
  "logo_path": "reference/logo.png",
  "color_correction": {
    "reference_color_checker_path": "reference/color_checker.png"
  },
  "geometrical_alignment": {
    "reference_path": "reference/aruco_reference.png",
    "marker_map": {
      "0": [[0, 0], [1000, 0], [1000, 1000], [0, 1000]]
    },
    "output_size": [1000, 1000]
  },
  "masking": {
    "drawing_layers": {
      "layer1": "drawings/layer1.png"
    }
  }
}
```

**Field Explanations:**

*   `training_path` (str): The path (relative to the project directory) to the folder containing the training images used to calculate the target color range.
*   `object_reference_path` (str, optional): Path to a reference image used for object-based alignment.
*   `logo_path` (str, optional): Path to a custom logo to be used in reports for this specific project.
*   **`color_correction`** (object):
    *   `reference_color_checker_path` (str): Path to the reference color checker image.
*   **`geometrical_alignment`** (object, optional):
    *   `reference_path` (str, optional): Path to the reference image containing ArUco markers.
    *   `marker_map` (dict, optional): A dictionary mapping ArUco marker IDs to their corresponding corner coordinates in the output image.
    *   `output_size` (list, optional): The desired output size `[width, height]` of the aligned image.
*   **`masking`** (object, optional):
    *   `drawing_layers` (dict, optional): A dictionary mapping layer names to their image file paths. These layers are used for masking the input image.

### `dataset_item_processing_config.json`

This optional file allows you to specify different color extraction methods for each image in your training dataset. If this file is not present, the application will use the "full_average" method for all training images.

**Example `dataset_item_processing_config.json`:**
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
        {"x": 100, "y": 150, "radius": 10},
        {"x": 250, "y": 300, "radius": 10}
      ]
    }
  ]
}
```

**Field Explanations:**

*   `image_configs` (list): A list of objects, where each object configures one image from the training set.
    *   `filename` (str): The name of the image file.
    *   `method` (str): The color extraction method to use. Can be:
        *   `"full_average"`: Averages the color of the entire image.
        *   `"points"`: Extracts colors from specific points in the image.
    *   `points` (list, optional): If `method` is `"points"`, this is a list of point objects, each with `x`, `y` coordinates and a `radius` defining the sampling area.
