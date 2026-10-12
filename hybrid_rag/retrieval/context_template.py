"""Template perakit konteks retrieval — bukan prompt.

Isinya format string yang menyusun entity/relation/chunk hasil retrieval jadi
satu string konteks untuk dikembalikan ke pemanggil. Tidak ada yang dikirim ke
model: satu-satunya prompt yang bicara ke LLM di modul ini ada di
``retrieval/keywords/keywords.yaml``, dirakit jadi chain oleh ``retrieval/keywords/chain.py``.

Karena itu namanya ``TEMPLATES``, bukan ``PROMPTS``.
"""

from typing import Any

TEMPLATES: dict[str, Any] = {}


TEMPLATES["kg_query_context"] = """
Knowledge Graph (Entity) :

```json
{entities_str}
```

Knowledge Graph (Relationship) :

```json
{relations_str}
```

Related Document Chunks :

```json
{text_chunks_str}
```

"""


# Appended only when ``QueryParam.include_citations``: one entry per (file, location).
TEMPLATES["kg_query_references"] = """Reference Document List (cite a chunk with its [reference_id]; each entry gives the source file and its location, page or sheet):

```
{reference_list_str}
```

"""
