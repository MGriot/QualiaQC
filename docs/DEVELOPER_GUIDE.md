# Developer's Guide

This guide provides instructions for setting up a local development environment to contribute to the Visual Analyzer project.

---

## 1. Setting up the Development Environment

### Prerequisites

*   Python 3.9 or higher
*   Git

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd QualiaQC
    ```

2.  **Create and activate a virtual environment:**

    *   On Windows:
        ```bash
        python -m venv .venv
        .venv\Scripts\activate.bat
        ```
    *   On macOS/Linux:
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```

3.  **Install dependencies:**
    All required packages are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

---

## 2. Running the Application

The main graphical user interface (GUI) is the primary entry point for the application.

*   **To run the GUI:**
    ```bash
    python -m src.gui
    ```
*   **To run in debug mode** (which exposes more options in the GUI):
    ```bash
    python -m src.gui --debug
    ```

---

## 3. Running Tests

This project uses `pytest` for testing. To run the test suite, simply execute the following command from the root of the project:

```bash
pytest
```

Make sure to write new tests for any new features or bug fixes you introduce.

---

## 4. Code Style and Linting

To maintain a consistent code style, this project uses a standard code formatter and linter. It is recommended to use a tool like `black` for formatting and `ruff` or `flake8` for linting.

*   **To format your code (using black):**
    ```bash
    pip install black
    black src/
    ```
*   **To check for linting errors (using a linter like ruff):**
    ```bash
    pip install ruff
    ruff check src/
    ```
