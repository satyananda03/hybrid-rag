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

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""
