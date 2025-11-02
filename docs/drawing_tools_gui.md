**Planning the code and GUI**I'm designing a Tkinter-based solution for loading PDFs, detecting regions using OpenCV or PIL, and enabling users to fill those regions interactively on a canvas.
Below is a **single-file, end‑to‑end Tkinter app** that lets you:

- **Open a PDF** (line art / black & white drawing)
- **Click to “paint bucket” fill** any closed area bounded by lines
- **Maintain fills as an overlay** (non-destructively)
- **Export**:
  - The **currently selected region** as a **separate transparent PNG “layer”**
  - **All colored regions** as separate transparent PNGs
  - The **composited result** (original + overlay) as a PNG
  - *(optional)* The current region as a one‑page **PDF** with transparent image (via PyMuPDF)

This approach works by rasterizing the PDF page to an image, identifying **closed regions** using connected-components on the **non-line** areas (lines are detected via automated thresholding and dilation), and then mapping clicks to region labels. No OpenCV is required.

---

## ✅ What you’ll need

```bash
pip install pymupdf pillow numpy scipy
```

> **Note**: No OpenCV required. Uses `pymupdf`, `Pillow`, `numpy`, and `scipy.ndimage`.

---

## 💻 The Code (save as `drawing_tools/gui.py`)
🧠 How it works (under the hood)
Render: Uses PyMuPDF to render the selected PDF page at 2× scale for sharper edges.
Detect lines & regions:
Converts to grayscale and applies Otsu thresholding to detect dark lines.
Dilates lines by a tunable amount to seal small gaps, preventing “leaks.”
Inverts to get the fillable areas.
Runs 8-connected components labeling on fillable regions ⇒ region map.
Marks border-touching regions as background (not fillable).
Click-to-fill:
Maps your click to the underlying region ID and paints that area onto a transparent overlay using your chosen color (RGBA with alpha).
The display is a live composite: base image + overlay.
Export:
Region PNG: A transparent image with only that region colored.
All regions: Batch-exports every colored region.
Composite PNG: Saves the full page with overlay applied.
Region PDF: Embeds the transparent PNG in a single‑page PDF (same pixel dimensions).
🧩 Notes, Tips & Adjustments
Gaps in lines? Use Options → Set line dilation (increase from 2 to 3–4) to seal tiny leaks.
White-on-black drawings? Use Options → Toggle invert lines if your source has light lines on dark background.
Performance: Very large pages may take a few seconds to segment. Rendering at 2× is a balance between precision and speed—bump to 3× if needed.
Vector output: Export is image-based (transparent PNG layer). If you need true vector extraction (e.g., SVG paths), we can extend this by tracing region boundaries (Moore neighbor tracing) and writing SVG—happy to add that if you need it.
▶️ Usage
Run: python -m src.drawing_tools.gui
File → Open PDF…
Click inside any closed area to fill it.
Edit → Pick Color to change fill color.
Export via File menu or toolbar buttons.
If you want me to extend this with SVG export (vectorized regions) or zoom/pan, say the word and I’ll add it.