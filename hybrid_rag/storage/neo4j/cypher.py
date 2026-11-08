"""Search schema and query templates — the Cypher, kept out of the Python.

Counterpart of ``storage/postgres/sql.py``. Needs Neo4j >= 2026.09: vector search uses
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
instead (:func:`bm25_query`).

Scores: Neo4j's cosine score is ``(1 + cos) / 2``, in [0, 1]. pgvector's ``<=>`` was the
distance ``1 - cos``. :func:`score_floor` and :func:`distance` convert between the two.
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

def hidden_projection(var: str) -> str:
    """``n {.*, hidden: null, ...}``: all properties of ``var`` except the search ones."""
    return f"{var} {{.*, " + ", ".join(f"`{key}`: null" for key in ("workspace", *HIDDEN)) + "}"


def score_floor(max_distance: float) -> float:
    """The Neo4j score above which the cosine distance is below ``max_distance``."""
    return 1 - max_distance / 2


def distance(score: float) -> float:
    """Cosine distance (``1 - cos``) of a Neo4j cosine score."""
    return 2 * (1 - score)


def ws_key(workspace: str) -> str:
    """The workspace as one plain token for the full-text index (names may hold ``.`` or ``-``)."""
    return compute_mdhash_id(workspace, prefix="ws")


def bm25_query(workspace: str, cleaned_text: str) -> str:
    """Lucene query: the workspace term is required, any of the words may match.

    ``cleaned_text`` comes from ``preprocess_indonesian_fast``: lowercase ``\\w+`` words
    only, so nothing in it is Lucene syntax. The required ``ws_key`` term adds the same
    score to every chunk of the workspace, so it does not change their order.
    """
    return f"+ws_key:{ws_key(workspace)} +content_cleaned:({cleaned_text})"


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


# ── Vector storage: writes ─────────────────────────────────────────────────
# Every upsert returns the ids it wrote. Chunks are created here; entity and relation
# vectors attach to a graph element that the graph storage owns, so a row whose
# element does not exist (yet) writes nothing and is not returned.
CHUNK_UPSERT = f"""
    UNWIND $rows AS row
    MERGE (c:{CHUNK_LABEL} {{workspace: $workspace, id: row.id}})
    ON CREATE SET c.created_at = row.created_at
    SET c += row.props, c.`{EMBEDDING}` = row.vector
    RETURN row.id AS id
"""
ENTITY_UPSERT = f"""
    UNWIND $rows AS row
    MATCH (n:{LABEL} {{workspace: $workspace, entity_id: row.entity_name}})
    SET n.{VDB_ID} = row.id, n.{VDB_CONTENT} = row.content, n.`{EMBEDDING}` = row.vector
    RETURN row.id AS id
"""
RELATION_UPSERT = f"""
    UNWIND $rows AS row
    MATCH (:{LABEL} {{workspace: $workspace, entity_id: row.src_id}})
          -[r:{EDGE_TYPE}]-(:{LABEL} {{workspace: $workspace, entity_id: row.tgt_id}})
    SET r.{VDB_ID} = row.id, r.{VDB_CONTENT} = row.content, r.`{EMBEDDING}` = row.vector,
        r.workspace = $workspace
    RETURN DISTINCT row.id AS id
"""

CHUNK_DELETE = f"""
    UNWIND $ids AS id
    MATCH (c:{CHUNK_LABEL} {{workspace: $workspace, id: id}})
    DETACH DELETE c
"""
# The node or edge itself belongs to the graph storage: only its vector goes.
ENTITY_DELETE = f"""
    UNWIND $ids AS id
    MATCH (n:{LABEL} {{workspace: $workspace, {VDB_ID}: id}})
    REMOVE n.{VDB_ID}, n.{VDB_CONTENT}, n.`{EMBEDDING}`
"""
RELATION_DELETE = f"""
    UNWIND $ids AS id
    MATCH ()-[r:{EDGE_TYPE} {{workspace: $workspace, {VDB_ID}: id}}]->()
    REMOVE r.{VDB_ID}, r.{VDB_CONTENT}, r.`{EMBEDDING}`
"""


# ── Vector storage: reads by id ────────────────────────────────────────────
# ``record``: the stored row without its vector; ``vector``: id, content and vector.
CHUNK_BY_ID = f"""
    UNWIND $ids AS id
    MATCH (c:{CHUNK_LABEL} {{workspace: $workspace, id: id}})
    RETURN c.id AS id, c.content AS content, c.`{EMBEDDING}` AS vector,
           c {{.full_doc_id, .chunk_order_index, .tokens, .file_path, .created_at}} AS fields
"""
ENTITY_BY_ID = f"""
    UNWIND $ids AS id
    MATCH (n:{LABEL} {{workspace: $workspace, {VDB_ID}: id}})
    RETURN n.{VDB_ID} AS id, n.{VDB_CONTENT} AS content, n.`{EMBEDDING}` AS vector,
           {{entity_name: n.entity_id, file_path: n.file_path, created_at: n.created_at}} AS fields
"""
RELATION_BY_ID = f"""
    UNWIND $ids AS id
    MATCH (a)-[r:{EDGE_TYPE} {{workspace: $workspace, {VDB_ID}: id}}]->(b)
    RETURN r.{VDB_ID} AS id, r.{VDB_CONTENT} AS content, r.`{EMBEDDING}` AS vector,
           {{source_id: a.entity_id, target_id: b.entity_id, file_path: r.file_path,
             created_at: r.created_at}} AS fields
"""

CHUNK_IDS_OF_DOC = f"""
    MATCH (c:{CHUNK_LABEL} {{workspace: $workspace, full_doc_id: $doc_id}})
    RETURN c.id AS id
"""


# ── Vector storage: search ─────────────────────────────────────────────────
def _vector_search(binding: str, pattern: str, index: str) -> str:
    """Top ``$top_k`` of the workspace by cosine, above ``$min_score``, best first."""
    return f"""
        CYPHER 25
        MATCH {pattern}
        SEARCH {binding} IN (
            VECTOR INDEX {index} FOR $embedding
            WHERE {binding}.workspace = $workspace
            LIMIT $top_k
        ) SCORE AS score
        WITH {binding}, score WHERE score > $min_score
    """


ENTITY_SEARCH = _vector_search("n", f"(n:{LABEL})", ENTITY_VECTOR_INDEX) + """
    RETURN n.entity_id AS entity_name, n.created_at AS created_at, score
    ORDER BY score DESC
"""
# Directed pattern: each edge once (an undirected one would yield it twice).
RELATION_SEARCH = _vector_search("r", f"()-[r:{EDGE_TYPE}]->()", RELATION_VECTOR_INDEX) + """
    RETURN startNode(r).entity_id AS src_id, endNode(r).entity_id AS tgt_id,
           r.created_at AS created_at, score
    ORDER BY score DESC
"""

# A chunk hit, as both chunk searches return it: best first, ties broken by id so
# ranks (and the RRF built on them) are reproducible.
_CHUNK_HIT = """
    RETURN c.id AS id, c.content AS content, c.file_path AS file_path,
           c.created_at AS created_at, score
    ORDER BY score DESC, c.id
"""
CHUNK_SEARCH = _vector_search("c", f"(c:{CHUNK_LABEL})", CHUNK_VECTOR_INDEX) + _CHUNK_HIT
# BM25 over ``content_cleaned``; ``$bm25_query`` comes from :func:`bm25_query`.
CHUNK_BM25_SEARCH = f"""
    CYPHER 25
    MATCH (c:{CHUNK_LABEL})
    SEARCH c IN (
        FULLTEXT INDEX {CHUNK_BM25_INDEX} FOR $bm25_query
        LIMIT $top_k
    ) SCORE AS score
""" + _CHUNK_HIT
