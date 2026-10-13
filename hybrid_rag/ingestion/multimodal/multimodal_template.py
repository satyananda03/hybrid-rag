"""Template perakit chunk multimodal — bukan prompt.

Isinya format string yang menyusun hasil analisis gambar/tabel/persamaan jadi
teks chunk yang disimpan ke storage. Tidak ada yang dikirim ke model: prompt
analisis VLM ada di ``multimodal.yaml``, dirakit jadi chain oleh
``services/vlm_service.py``.

Karena itu namanya ``TEMPLATES``, bukan ``PROMPTS``.
"""

from typing import Any

TEMPLATES: dict[str, Any] = {}

# Template chunk per modalitas
TEMPLATES["image_chunk"] = """
Image Content Analysis:
- Section Path: {section_path}
- Neighbor Text: {neighbor_text}
Image Path: {image_path}
Captions: {captions}
Footnotes: {footnotes}

Visual Analysis: {enhanced_caption}
"""

# ``image_path_line`` is "Image Path: <url>\n", or empty for tables without an image
# (Textract, Excel) so the chunk carries no blank field.
TEMPLATES["table_chunk"] = """
Table Analysis:
{image_path_line}Caption: {table_caption}
Structure: {table_body}
Footnotes: {table_footnote}

Analysis: {enhanced_caption}
"""

TEMPLATES["equation_chunk"] = """
Mathematical Equation Analysis:
Equation: {equation_text}
Format: {equation_format}

Mathematical Analysis: {enhanced_caption}
"""

