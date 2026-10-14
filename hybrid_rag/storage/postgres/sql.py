"""Table definitions and query templates — the SQL, kept out of the Python.

``TABLES`` holds the DDL (including the BM25 and HNSW index creation),
``SQL_TEMPLATES`` every parameterized statement, and ``NAMESPACE_TABLE_MAP``
resolves a storage namespace to its table.
"""



from hybrid_rag.core.constants import NameSpace, is_namespace






# Note: Order matters! More specific namespaces (e.g., "full_entities") must come before
# more general ones (e.g., "entities") because is_namespace() uses endswith() matching
NAMESPACE_TABLE_MAP = {
    NameSpace.KV_STORE_FULL_DOCS: "hybrid_rag_doc_full",
    NameSpace.KV_STORE_TEXT_CHUNKS: "hybrid_rag_doc_chunks",
    NameSpace.KV_STORE_FULL_ENTITIES: "hybrid_rag_doc_entities",
    NameSpace.KV_STORE_FULL_RELATIONS: "hybrid_rag_doc_relations",
    NameSpace.KV_STORE_ENTITY_CHUNKS: "hybrid_rag_entity_chunks",
    NameSpace.KV_STORE_RELATION_CHUNKS: "hybrid_rag_relation_chunks",
    NameSpace.KV_STORE_LLM_RESPONSE_CACHE: "hybrid_rag_llm_cache",
    NameSpace.VECTOR_STORE_CHUNKS: "hybrid_rag_vdb_chunks",
    NameSpace.VECTOR_STORE_ENTITIES: "hybrid_rag_vdb_entities",
    NameSpace.VECTOR_STORE_RELATIONSHIPS: "hybrid_rag_vdb_relations",
    NameSpace.DOC_STATUS: "hybrid_rag_doc_status",
}


def namespace_to_table_name(namespace: str) -> str:
    for k, v in NAMESPACE_TABLE_MAP.items():
        if is_namespace(namespace, k):
            return v


TABLES = {
    "hybrid_rag_doc_full": {
        "ddl": """CREATE TABLE hybrid_rag_doc_full (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    doc_name VARCHAR(1024),
                    content TEXT,
                    meta JSONB,
                    sidecar_location TEXT NULL,
                    parse_format VARCHAR(32) NULL DEFAULT 'raw',
                    -- content_hash is TEXT (not VARCHAR(N)) so the column is
                    -- agnostic to the hash algorithm. Today's pipeline writes
                    -- 64-char SHA-256 hex; future algos (SHA-512, base64) do
                    -- not require a schema change.
                    content_hash TEXT NULL,
                    -- process_options is an opaque selector string emitted by
                    -- sanitize_process_options() (e.g. "Fi").
                    process_options TEXT NULL,
                    chunk_options JSONB NULL DEFAULT '{}'::jsonb,
                    parse_engine TEXT NULL,
                    -- Multimodal blocks (image / table / equation) split out of
                    -- the parsed content list at enqueue. The text half lives in
                    -- ``content``; these are what process_single_document captions.
                    multimodal_items JSONB NULL DEFAULT '[]'::jsonb,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
	                CONSTRAINT hybrid_rag_doc_full_pk PRIMARY KEY (workspace, id)
                    )"""
    },
    "hybrid_rag_doc_chunks": {
        "ddl": """CREATE TABLE hybrid_rag_doc_chunks (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    full_doc_id VARCHAR(256),
                    chunk_order_index INTEGER,
                    tokens INTEGER,
                    content TEXT,
                    file_path TEXT NULL,
                    llm_cache_list JSONB NULL DEFAULT '[]'::jsonb,
                    heading JSONB NULL DEFAULT '{}'::jsonb,
                    sidecar JSONB NULL DEFAULT '{}'::jsonb,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
	                CONSTRAINT hybrid_rag_doc_chunks_pk PRIMARY KEY (workspace, id)
                    )"""
    },
    "hybrid_rag_vdb_chunks": {
        "ddl": """CREATE TABLE hybrid_rag_vdb_chunks (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    full_doc_id VARCHAR(256),
                    chunk_order_index INTEGER,
                    tokens INTEGER,
                    content TEXT,
                    content_vector VECTOR(dimension),
                    file_path TEXT NULL,
                    content_cleaned TEXT NULL,
                    -- BM25 key_field: unique, unlike id, which repeats across workspaces
                    bm25_key TEXT GENERATED ALWAYS AS (workspace || ':' || id) STORED,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT hybrid_rag_vdb_chunks_pk PRIMARY KEY (workspace, id)
                    )"""
    },
    "hybrid_rag_vdb_entities": {
        "ddl": """CREATE TABLE hybrid_rag_vdb_entities (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    entity_name VARCHAR(512),
                    content TEXT,
                    content_vector VECTOR(dimension),
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    chunk_ids VARCHAR(255)[] NULL,
                    file_path TEXT NULL,
	                CONSTRAINT hybrid_rag_vdb_entities_pk PRIMARY KEY (workspace, id)
                    )"""
    },
    "hybrid_rag_vdb_relations": {
        "ddl": """CREATE TABLE hybrid_rag_vdb_relations (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    source_id VARCHAR(512),
                    target_id VARCHAR(512),
                    content TEXT,
                    content_vector VECTOR(dimension),
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    chunk_ids VARCHAR(255)[] NULL,
                    file_path TEXT NULL,
	                CONSTRAINT hybrid_rag_vdb_relations_pk PRIMARY KEY (workspace, id)
                    )"""
    },
    "hybrid_rag_llm_cache": {
        "ddl": """CREATE TABLE hybrid_rag_llm_cache (
	                workspace varchar(255) NOT NULL,
	                id varchar(255) NOT NULL,
                    original_prompt TEXT,
                    return_value TEXT,
                    chunk_id VARCHAR(255) NULL,
                    cache_type VARCHAR(32),
                    queryparam JSONB NULL,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	                CONSTRAINT hybrid_rag_llm_cache_pk PRIMARY KEY (workspace, id)
                    )"""
    },
    "hybrid_rag_doc_status": {
        "ddl": """CREATE TABLE hybrid_rag_doc_status (
	               workspace varchar(255) NOT NULL,
	               id varchar(255) NOT NULL,
	               content_summary varchar(255) NULL,
	               content_length int4 NULL,
	               chunks_count int4 NULL,
	               status varchar(64) NULL,
	               file_path TEXT NULL,
	               chunks_list JSONB NULL DEFAULT '[]'::jsonb,
	               track_id varchar(255) NULL,
	               metadata JSONB NULL DEFAULT '{}'::jsonb,
	               error_msg TEXT NULL,
	               -- content_hash is TEXT (not VARCHAR(N)) so the column is
	               -- agnostic to the hash algorithm. Today's pipeline writes
	               -- 64-char SHA-256 hex; future algos (SHA-512, base64) do
	               -- not require a schema change.
	               content_hash TEXT NULL,
	               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	               updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	               CONSTRAINT hybrid_rag_doc_status_pk PRIMARY KEY (workspace, id)
	              )"""
    },
    "hybrid_rag_doc_entities": {
        "ddl": """CREATE TABLE hybrid_rag_doc_entities (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    entity_names JSONB,
                    count INTEGER,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT hybrid_rag_doc_entities_pk PRIMARY KEY (workspace, id)
                    )"""
    },
    "hybrid_rag_doc_relations": {
        "ddl": """CREATE TABLE hybrid_rag_doc_relations (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    relation_pairs JSONB,
                    count INTEGER,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT hybrid_rag_doc_relations_pk PRIMARY KEY (workspace, id)
                    )"""
    },
    "hybrid_rag_entity_chunks": {
        "ddl": """CREATE TABLE hybrid_rag_entity_chunks (
                    id VARCHAR(512),
                    workspace VARCHAR(255),
                    chunk_ids JSONB,
                    count INTEGER,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT hybrid_rag_entity_chunks_pk PRIMARY KEY (workspace, id)
                    )"""
    },
    "hybrid_rag_relation_chunks": {
        "ddl": """CREATE TABLE hybrid_rag_relation_chunks (
                    id VARCHAR(512),
                    workspace VARCHAR(255),
                    chunk_ids JSONB,
                    count INTEGER,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT hybrid_rag_relation_chunks_pk PRIMARY KEY (workspace, id)
                    )"""
    },
}


SQL_TEMPLATES = {
    # SQL for KVStorage
    "get_by_id_full_docs": """SELECT id, COALESCE(content, '') as content,
                                COALESCE(doc_name, '') as file_path,
                                sidecar_location,
                                parse_format,
                                content_hash,
                                process_options,
                                COALESCE(chunk_options, '{}'::jsonb) as chunk_options,
                                parse_engine,
                                COALESCE(multimodal_items, '[]'::jsonb) as multimodal_items
                                FROM hybrid_rag_doc_full WHERE workspace=$1 AND id=$2
                            """,
    "get_by_id_text_chunks": """SELECT id, tokens, COALESCE(content, '') as content,
                                chunk_order_index, full_doc_id, file_path,
                                COALESCE(llm_cache_list, '[]'::jsonb) as llm_cache_list,
                                COALESCE(heading, '{}'::jsonb) as heading,
                                COALESCE(sidecar, '{}'::jsonb) as sidecar,
                                EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                FROM hybrid_rag_doc_chunks WHERE workspace=$1 AND id=$2
                            """,
    "get_by_id_llm_response_cache": """SELECT id, original_prompt, return_value, chunk_id, cache_type, queryparam,
                                EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                FROM hybrid_rag_llm_cache WHERE workspace=$1 AND id=$2
                               """,
    "get_by_ids_full_docs": """SELECT id, COALESCE(content, '') as content,
                                 COALESCE(doc_name, '') as file_path,
                                 sidecar_location,
                                 parse_format,
                                 content_hash,
                                 process_options,
                                 COALESCE(chunk_options, '{}'::jsonb) as chunk_options,
                                 parse_engine,
                                 COALESCE(multimodal_items, '[]'::jsonb) as multimodal_items
                                 FROM hybrid_rag_doc_full WHERE workspace=$1 AND id = ANY($2)
                            """,
    "get_by_ids_text_chunks": """SELECT id, tokens, COALESCE(content, '') as content,
                                  chunk_order_index, full_doc_id, file_path,
                                  COALESCE(llm_cache_list, '[]'::jsonb) as llm_cache_list,
                                  COALESCE(heading, '{}'::jsonb) as heading,
                                  COALESCE(sidecar, '{}'::jsonb) as sidecar,
                                  EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                  EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                   FROM hybrid_rag_doc_chunks WHERE workspace=$1 AND id = ANY($2)
                                """,
    "get_by_ids_llm_response_cache": """SELECT id, original_prompt, return_value, chunk_id, cache_type, queryparam,
                                 EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                 EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                 FROM hybrid_rag_llm_cache WHERE workspace=$1 AND id = ANY($2)
                                """,
    "get_by_id_full_entities": """SELECT id, entity_names, count,
                                EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                FROM hybrid_rag_doc_entities WHERE workspace=$1 AND id=$2
                               """,
    "get_by_id_full_relations": """SELECT id, relation_pairs, count,
                                EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                FROM hybrid_rag_doc_relations WHERE workspace=$1 AND id=$2
                               """,
    "get_by_ids_full_entities": """SELECT id, entity_names, count,
                                 EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                 EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                 FROM hybrid_rag_doc_entities WHERE workspace=$1 AND id = ANY($2)
                                """,
    "get_by_ids_full_relations": """SELECT id, relation_pairs, count,
                                 EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                 EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                 FROM hybrid_rag_doc_relations WHERE workspace=$1 AND id = ANY($2)
                                """,
    "get_by_id_entity_chunks": """SELECT id, chunk_ids, count,
                                EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                FROM hybrid_rag_entity_chunks WHERE workspace=$1 AND id=$2
                               """,
    "get_by_id_relation_chunks": """SELECT id, chunk_ids, count,
                                EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                FROM hybrid_rag_relation_chunks WHERE workspace=$1 AND id=$2
                               """,
    "get_by_ids_entity_chunks": """SELECT id, chunk_ids, count,
                                 EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                 EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                 FROM hybrid_rag_entity_chunks WHERE workspace=$1 AND id = ANY($2)
                                """,
    "get_by_ids_relation_chunks": """SELECT id, chunk_ids, count,
                                 EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                 EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                 FROM hybrid_rag_relation_chunks WHERE workspace=$1 AND id = ANY($2)
                                """,
    # Pipeline-derived columns (sidecar_location / parse_format / content_hash /
    # process_options / chunk_options / parse_engine) are guarded with COALESCE
    # so a partial upsert (e.g. a caller writing only ``content`` + ``doc_name``)
    # does not silently overwrite metadata recorded by an earlier full upsert.
    # ``content`` and ``doc_name`` themselves are always overwritten — they are
    # the primary payload, never a candidate for preservation.
    # For the string columns we use NULLIF('', ...) so that an empty string from
    # a default-bearing caller is treated as "no value, preserve existing".
    # For chunk_options (JSONB) we treat NULL or the empty-object literal as
    # "no value, preserve existing".
    "upsert_doc_full": """INSERT INTO hybrid_rag_doc_full (id, content, doc_name, workspace,
                            sidecar_location, parse_format, content_hash,
                            process_options, chunk_options, parse_engine,
                            multimodal_items)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        ON CONFLICT (workspace,id) DO UPDATE
                           SET content = EXCLUDED.content,
                               doc_name = EXCLUDED.doc_name,
                               sidecar_location = COALESCE(
                                   NULLIF(EXCLUDED.sidecar_location, ''),
                                   hybrid_rag_doc_full.sidecar_location
                               ),
                               parse_format = COALESCE(
                                   NULLIF(EXCLUDED.parse_format, ''),
                                   hybrid_rag_doc_full.parse_format
                               ),
                               content_hash = COALESCE(
                                   NULLIF(EXCLUDED.content_hash, ''),
                                   hybrid_rag_doc_full.content_hash
                               ),
                               process_options = COALESCE(
                                   NULLIF(EXCLUDED.process_options, ''),
                                   hybrid_rag_doc_full.process_options
                               ),
                               chunk_options = CASE
                                   WHEN EXCLUDED.chunk_options IS NULL
                                     OR EXCLUDED.chunk_options = '{}'::jsonb
                                   THEN hybrid_rag_doc_full.chunk_options
                                   ELSE EXCLUDED.chunk_options
                               END,
                               parse_engine = COALESCE(
                                   NULLIF(EXCLUDED.parse_engine, ''),
                                   hybrid_rag_doc_full.parse_engine
                               ),
                               -- Same guard as chunk_options: an empty list
                               -- means "caller did not supply", not "clear it".
                               multimodal_items = CASE
                                   WHEN EXCLUDED.multimodal_items IS NULL
                                     OR EXCLUDED.multimodal_items = '[]'::jsonb
                                   THEN hybrid_rag_doc_full.multimodal_items
                                   ELSE EXCLUDED.multimodal_items
                               END,
                               update_time = CURRENT_TIMESTAMP
                       """,
    "upsert_llm_response_cache": """INSERT INTO hybrid_rag_llm_cache(workspace,id,original_prompt,return_value,chunk_id,cache_type,queryparam)
                                      VALUES ($1, $2, $3, $4, $5, $6, $7)
                                      ON CONFLICT (workspace,id) DO UPDATE
                                      SET original_prompt = EXCLUDED.original_prompt,
                                      return_value=EXCLUDED.return_value,
                                      chunk_id=EXCLUDED.chunk_id,
                                      cache_type=EXCLUDED.cache_type,
                                      queryparam=EXCLUDED.queryparam,
                                      update_time = CURRENT_TIMESTAMP
                                     """,
    "upsert_text_chunk": """INSERT INTO hybrid_rag_doc_chunks (workspace, id, tokens,
                      chunk_order_index, full_doc_id, content, file_path, llm_cache_list,
                      heading, sidecar, create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET tokens=EXCLUDED.tokens,
                      chunk_order_index=EXCLUDED.chunk_order_index,
                      full_doc_id=EXCLUDED.full_doc_id,
                      content = EXCLUDED.content,
                      file_path=EXCLUDED.file_path,
                      llm_cache_list=EXCLUDED.llm_cache_list,
                      heading=EXCLUDED.heading,
                      sidecar=EXCLUDED.sidecar,
                      update_time = EXCLUDED.update_time
                     """,
    "upsert_full_entities": """INSERT INTO hybrid_rag_doc_entities (workspace, id, entity_names, count,
                      create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET entity_names=EXCLUDED.entity_names,
                      count=EXCLUDED.count,
                      update_time = EXCLUDED.update_time
                     """,
    "upsert_full_relations": """INSERT INTO hybrid_rag_doc_relations (workspace, id, relation_pairs, count,
                      create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET relation_pairs=EXCLUDED.relation_pairs,
                      count=EXCLUDED.count,
                      update_time = EXCLUDED.update_time
                     """,
    "upsert_entity_chunks": """INSERT INTO hybrid_rag_entity_chunks (workspace, id, chunk_ids, count,
                      create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET chunk_ids=EXCLUDED.chunk_ids,
                      count=EXCLUDED.count,
                      update_time = EXCLUDED.update_time
                     """,
    "upsert_relation_chunks": """INSERT INTO hybrid_rag_relation_chunks (workspace, id, chunk_ids, count,
                      create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET chunk_ids=EXCLUDED.chunk_ids,
                      count=EXCLUDED.count,
                      update_time = EXCLUDED.update_time
                     """,
    # SQL for VectorStorage
    "upsert_chunk": """INSERT INTO {table_name} (workspace, id, tokens,
                      chunk_order_index, full_doc_id, content, content_vector, file_path,
                      content_cleaned, create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET tokens=EXCLUDED.tokens,
                      chunk_order_index=EXCLUDED.chunk_order_index,
                      full_doc_id=EXCLUDED.full_doc_id,
                      content = EXCLUDED.content,
                      content_vector=EXCLUDED.content_vector,
                      file_path=EXCLUDED.file_path,
                      content_cleaned=EXCLUDED.content_cleaned,
                      update_time = EXCLUDED.update_time
                     """,
    "upsert_entity": """INSERT INTO {table_name} (workspace, id, entity_name, content,
                      content_vector, chunk_ids, file_path, create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6::varchar[], $7, $8, $9)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET entity_name=EXCLUDED.entity_name,
                      content=EXCLUDED.content,
                      content_vector=EXCLUDED.content_vector,
                      chunk_ids=EXCLUDED.chunk_ids,
                      file_path=EXCLUDED.file_path,
                      update_time=EXCLUDED.update_time
                     """,
    "upsert_relationship": """INSERT INTO {table_name} (workspace, id, source_id,
                      target_id, content, content_vector, chunk_ids, file_path, create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6, $7::varchar[], $8, $9, $10)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET source_id=EXCLUDED.source_id,
                      target_id=EXCLUDED.target_id,
                      content=EXCLUDED.content,
                      content_vector=EXCLUDED.content_vector,
                      chunk_ids=EXCLUDED.chunk_ids,
                      file_path=EXCLUDED.file_path,
                      update_time = EXCLUDED.update_time
                     """,
    # Read path (retrieval)
    # The "+ 0" in every ORDER BY below keeps the HNSW index out of these queries on
    # purpose. HNSW takes the ef_search nearest rows of the WHOLE shared table and only
    # then applies `workspace = $1`, so a workspace that is a small share of the table
    # gets few or none of its true nearest rows (probe: recall 0-34%; with
    # hnsw.iterative_scan precision@k drops to ~50%). The exact path (workspace index +
    # sort) is always correct; its cost grows with the workspace's own row count.
    # ponytail: linear in workspace size; partition per workspace if one gets large.
    "relationships": """
                     SELECT source_id AS src_id,
                            target_id AS tgt_id,
                            EXTRACT(EPOCH FROM create_time)::BIGINT AS created_at
                     FROM {table_name}
                     WHERE workspace = $1
                       AND content_vector <=> $4::{vector_cast} < $2
                     ORDER BY (content_vector <=> $4::{vector_cast}) + 0
                     LIMIT $3;
                     """,
    "entities": """
                SELECT entity_name,
                       EXTRACT(EPOCH FROM create_time)::BIGINT AS created_at
                FROM {table_name}
                WHERE workspace = $1
                  AND content_vector <=> $4::{vector_cast} < $2
                ORDER BY (content_vector <=> $4::{vector_cast}) + 0
                LIMIT $3;
                """,
    # Each arm takes its top $3 with ORDER BY ... LIMIT, then ranks only those rows:
    # that is what lets ParadeDB run a top-K scan, and makes the top $3 guaranteed.
    # ``+ 0`` keeps the vector arm an exact scan of the workspace rather than an
    # HNSW scan filtered afterwards. The BM25 index holds ``workspace`` as a
    # literal, so ``workspace = $1`` is a term in the index, not a heap filter.
    "chunks_hybrid_bm25": """
            WITH semantic_search AS (
                SELECT *, ROW_NUMBER() OVER (ORDER BY vector_distance) AS rank
                FROM (
                    SELECT
                        id,
                        content,
                        file_path,
                        EXTRACT(EPOCH FROM create_time)::BIGINT AS created_at,
                        (content_vector <=> $4::{vector_cast}) AS vector_distance
                    FROM {table_name}
                    WHERE workspace = $1
                      AND content_vector <=> $4::{vector_cast} < $2
                    ORDER BY (content_vector <=> $4::{vector_cast}) + 0
                    LIMIT $3
                ) top_semantic
            ),
            bm25_search AS (
                -- id breaks score ties (common in BM25) so ranks are reproducible; it
                -- stays out of the inner ORDER BY, where it would disable the top-K scan.
                SELECT *, ROW_NUMBER() OVER (ORDER BY bm25_score DESC, id) AS rank
                FROM (
                    SELECT
                        id,
                        content,
                        file_path,
                        EXTRACT(EPOCH FROM create_time)::BIGINT AS created_at,
                        pdb.score(bm25_key) AS bm25_score
                    FROM {table_name}
                    WHERE workspace = $1
                      AND content_cleaned ||| $5
                    ORDER BY pdb.score(bm25_key) DESC
                    LIMIT $3
                ) top_bm25
            )
            SELECT
                COALESCE(s.id,         b.id)         AS id,
                COALESCE(s.content,    b.content)    AS content,
                COALESCE(s.file_path,  b.file_path)  AS file_path,
                COALESCE(s.created_at, b.created_at) AS created_at,
                s.vector_distance                    AS vector_distance,
                b.bm25_score                         AS bm25_score,
                COALESCE($6 / (60.0 + s.rank), 0.0) +
                COALESCE($7 / (60.0 + b.rank), 0.0) AS rrf_score
            FROM semantic_search s
            FULL OUTER JOIN bm25_search b ON s.id = b.id
            ORDER BY rrf_score DESC
            LIMIT $3;
        """,
}
