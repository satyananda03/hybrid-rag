"""What the Neo4j storages must agree on: names, the workspace key, the chunk hit, the indexes.

Each query lives next to the code that runs it (``graph.py``, ``vector.py``, ``bm25.py``);
only what more than one of them depends on is here. Needs Neo4j >= 2026.09: vector search uses
the Cypher 25 ``SEARCH`` clause (2026.01) and BM25 its ``FULLTEXT`` form (2026.09).

Where the vectors live (the layout the Neo4j hybrid-search guide recommends: text,
embedding and both indexes on the node that is searched):

* chunks    — ``(:HybridRagChunk)`` nodes: text, BM25 text and embedding;
* entities  — the graph's own ``(:HybridRagEntity)`` nodes;
* relations — the graph's own ``[:DIRECTED]`` edges.

Every workspace shares one graph, so every search filters by workspace *inside* the
index: a filter applied after it would let a large workspace crowd a small one out of
the top-k. Vector search does that with ``SEARCH ... WHERE x.workspace = $workspace``
(the ``workspace`` property is declared on each vector index for this). Full-text
``SEARCH`` takes no ``WHERE``, so the workspace is a required term of the Lucene query
instead (``bm25.bm25_query``).

Scores: Neo4j's cosine score is ``(1 + cos) / 2``, in [0, 1]. pgvector's ``<=>`` was the
distance ``1 - cos``. ``vector.score_floor`` and ``vector.distance`` convert between the two.
"""

from hybrid_rag.config import settings
from hybrid_rag.core.base import collection_suffix
from hybrid_rag.core.utils import compute_mdhash_id

LABEL = "HybridRagEntity"
CHUNK_LABEL = "HybridRagChunk"
EDGE_TYPE = "DIRECTED"

# One embedding property per model and dimension (the vector tables used to carry the
# same suffix), so switching EMBEDDING_MODEL never mixes vectors of two models.
SUFFIX = collection_suffix(settings.embedding_model, settings.embedding_dim) or "default"
EMBEDDING = f"embedding_{SUFFIX}"
CHUNK_VECTOR_INDEX = f"hybrid_rag_chunk_vector_{SUFFIX}"
ENTITY_VECTOR_INDEX = f"hybrid_rag_entity_vector_{SUFFIX}"
RELATION_VECTOR_INDEX = f"hybrid_rag_relation_vector_{SUFFIX}"
CHUNK_BM25_INDEX = "hybrid_rag_chunk_bm25"
ENTITY_FULLTEXT_INDEX = "hybrid_rag_entity_id_fulltext"  # entity-name search (graph labels)

# Search-only properties on entity nodes and edges. Graph reads hide them: callers see
# the same properties as before, and the 1024 floats never travel with a node read.
VDB_ID, VDB_CONTENT = "vdb_id", "vdb_content"
HIDDEN = (EMBEDDING, VDB_ID, VDB_CONTENT)


# The writer (vector.py) and the BM25 query (bm25.py) must derive the same key.
def ws_key(workspace: str) -> str:
    """The workspace as one plain token for the full-text index (names may hold ``.`` or ``-``)."""
    return compute_mdhash_id(workspace, prefix="ws")


# A chunk hit, as both chunk searches return it: best first, ties broken by id so
# ranks (and the RRF built on them) are reproducible.
CHUNK_HIT = """
    RETURN c.id AS id, c.content AS content, c.file_path AS file_path,
           c.created_at AS created_at, score
    ORDER BY score DESC, c.id
"""


# ── Schema (``make db-init``) ─────────────────────────────────────────────
_VECTOR_OPTIONS = (
    "OPTIONS {indexConfig: {"
    f"`vector.dimensions`: {settings.embedding_dim}, "
    "`vector.similarity_function`: 'cosine', "
    # Default since 2026.08 is binary quantization: faster, less accurate.
    "`vector.quantization.type`: 'none'}}"
)

SCHEMA = [
    f"CREATE CONSTRAINT hybrid_rag_entity_key IF NOT EXISTS "
    f"FOR (n:{LABEL}) REQUIRE (n.workspace, n.entity_id) IS UNIQUE",
    f"CREATE FULLTEXT INDEX {ENTITY_FULLTEXT_INDEX} IF NOT EXISTS "
    f"FOR (n:{LABEL}) ON EACH [n.entity_id] "
    "OPTIONS {indexConfig: {`fulltext.analyzer`: 'cjk', `fulltext.eventually_consistent`: true}}",
    f"CREATE CONSTRAINT hybrid_rag_chunk_key IF NOT EXISTS "
    f"FOR (c:{CHUNK_LABEL}) REQUIRE (c.workspace, c.id) IS UNIQUE",
    # Deletion resolves a document's chunks through full_doc_id.
    f"CREATE INDEX hybrid_rag_chunk_doc IF NOT EXISTS FOR (c:{CHUNK_LABEL}) ON (c.workspace, c.full_doc_id)",
    # Vector deletes and reads arrive by vdb id ("ent-<md5>", "rel-<md5>").
    f"CREATE INDEX hybrid_rag_entity_vdb_id IF NOT EXISTS FOR (n:{LABEL}) ON (n.workspace, n.{VDB_ID})",
    f"CREATE INDEX hybrid_rag_relation_vdb_id IF NOT EXISTS "
    f"FOR ()-[r:{EDGE_TYPE}]-() ON (r.workspace, r.{VDB_ID})",
    f"CYPHER 25 CREATE VECTOR INDEX {CHUNK_VECTOR_INDEX} IF NOT EXISTS "
    f"FOR (c:{CHUNK_LABEL}) ON c.`{EMBEDDING}` WITH [c.workspace] {_VECTOR_OPTIONS}",
    f"CYPHER 25 CREATE VECTOR INDEX {ENTITY_VECTOR_INDEX} IF NOT EXISTS "
    f"FOR (n:{LABEL}) ON n.`{EMBEDDING}` WITH [n.workspace] {_VECTOR_OPTIONS}",
    f"CYPHER 25 CREATE VECTOR INDEX {RELATION_VECTOR_INDEX} IF NOT EXISTS "
    f"FOR ()-[r:{EDGE_TYPE}]-() ON r.`{EMBEDDING}` WITH [r.workspace] {_VECTOR_OPTIONS}",
    # Text is normalised in Python (preprocess_indonesian_fast) on both the stored and the
    # query side, so the analyzer only splits on whitespace.
    f"CREATE FULLTEXT INDEX {CHUNK_BM25_INDEX} IF NOT EXISTS "
    f"FOR (c:{CHUNK_LABEL}) ON EACH [c.content_cleaned, c.ws_key] "
    "OPTIONS {indexConfig: {`fulltext.analyzer`: 'whitespace'}}",
    # Edges written before edges carried ``workspace`` (the relation vector index filters on it).
    f"MATCH (a:{LABEL})-[r:{EDGE_TYPE}]->() WHERE r.workspace IS NULL "
    "CALL (a, r) { SET r.workspace = a.workspace } IN TRANSACTIONS OF 10000 ROWS",
    "CALL db.awaitIndexes(600)",
]
