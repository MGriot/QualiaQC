ToDo List

1. good, thank you! now work as professionist pythonist with chirurgical intervent in code touching only the
  essential parts without deleting or modify too much the scripts. asure that the .gri contain all the image
  and infos as array, then update the report re generation script for work without path but with arrays images.
   so less memory need to use for save a already done report and less files

✦ Excellent, thank you for the clear instructions. I understand the goal is to make the .gri report archives
  fully self-contained by embedding all necessary images as arrays, and then updating the report regeneration
   logic to use these arrays instead of relying on external file paths. This will improve portability and
  reduce file clutter.

  I will perform this "chirurgical" intervention in two main steps:

   1. Enhance Archiving: I will modify the generate_report method in src/reporting/generator.py to ensure that
       all images displayed in the report (including generated plots and debug images) are read from disk and
      saved as NumPy arrays within the .gri archive file.
   2. Update Regeneration: I will then modify the _generate_reportlab_pdf method in the same file. The
      add_image helper function within it will be updated to first check for these embedded image arrays when
      regenerating a report. If an array is found, it will be used directly; otherwise, it will fall back to
      the original file path logic for backward compatibility.

  I will begin with the first step to enhance the archiving process.
