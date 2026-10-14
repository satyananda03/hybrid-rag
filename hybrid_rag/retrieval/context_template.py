"""Template perakit konteks retrieval — bukan prompt.

Isinya format string yang menyusun entity/relation/chunk hasil retrieval jadi
satu string konteks untuk dikembalikan ke pemanggil. Tidak ada yang dikirim ke
model: satu-satunya prompt yang bicara ke LLM di modul ini ada di
``retrieval/keywords/keywords.yaml``, dirakit jadi chain oleh ``retrieval/keywords/chain.py``.

Karena itu namanya ``TEMPLATES``, bukan ``PROMPTS``.
"""

from typing import Any

TEMPLATES: dict[str, Any] = {}


# Tag XML membatasi bagian dan chunk; isinya teks biasa supaya mudah dibaca LLM.
TEMPLATES["kg_query_context"] = """<context>
<knowledge_graph>
### Entities
{entities_str}

### Relationships
{relations_str}
</knowledge_graph>

<document_chunks>
{text_chunks_str}
</document_chunks>
{references_block}</context>
"""


# Rendered only when ``QueryParam.include_citations``: one entry per (file, location).
TEMPLATES["kg_query_references"] = """
<references>
{reference_list_str}
</references>
"""
