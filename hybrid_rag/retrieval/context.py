"""Stages 2-4 — turn search results into the context string.

* :func:`_apply_token_truncation` — fit entities and relations into
  ``max_entity_tokens`` / ``max_relation_tokens``.
* :func:`_merge_all_chunks` — collect the chunks the surviving entities and
  relations point at, merge with the vector chunks, then hand them to
  :mod:`.reranking`.
* :func:`_build_context_str` — render entities / relations / chunks and the
  reference list into the string the caller receives.
"""


import html
import json
import re
from typing import Any

from hybrid_rag.retrieval.utils import convert_to_user_format, generate_reference_list_from_chunks, pick_by_vector_similarity, pick_by_weighted_polling
from hybrid_rag.core.utils import split_string_by_multi_markers, truncate_list_by_token_size
from hybrid_rag.core.base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage
from hybrid_rag.retrieval.types import QueryParam
from hybrid_rag.core.chunk_schema import format_parent_headings
# Every prompt that reaches a model lives in retrieval/keywords/keywords.yaml and is wired
# into chains by keywords.chain. What is left in TEMPLATES is the two context templates
# that serialize retrieved entities/relations/chunks — formatting, not
# instructions.
from hybrid_rag.retrieval.context_template import TEMPLATES
from hybrid_rag.core.constants import GRAPH_FIELD_SEP, DEFAULT_MAX_SECTION_CONTEXT_TOKENS
from hybrid_rag.retrieval.rerank import process_chunks_unified
from hybrid_rag.core.chunk_schema import _truncate_section_context
import time
from hybrid_rag.config import settings
import logging

logger = logging.getLogger(__name__)


async def _apply_token_truncation(
    search_result: dict[str, Any],
    query_param: QueryParam,
    global_config: dict[str, str],
) -> dict[str, Any]:
    """
    Apply token-based truncation to entities and relations for LLM efficiency.
    """
    tokenizer = global_config.get("tokenizer")
    if not tokenizer:
        logger.warning("No tokenizer found, skipping truncation")
        return {
            "entities_context": [],
            "relations_context": [],
            "filtered_entities": search_result["final_entities"],
            "filtered_relations": search_result["final_relations"],
            "entity_id_to_original": {},
            "relation_id_to_original": {},
        }

    # Get token limits from query_param with fallbacks
    max_entity_tokens = getattr(
        query_param,
        "max_entity_tokens",
        global_config.get("max_entity_tokens", settings.max_entity_tokens),
    )
    max_relation_tokens = getattr(
        query_param,
        "max_relation_tokens",
        global_config.get("max_relation_tokens", settings.max_relation_tokens),
    )

    final_entities = search_result["final_entities"]
    final_relations = search_result["final_relations"]

    # Create mappings from entity/relation identifiers to original data
    entity_id_to_original = {}
    relation_id_to_original = {}

    # Generate entities context for truncation
    entities_context = []
    for i, entity in enumerate(final_entities):
        entity_name = entity["entity_name"]
        created_at = entity.get("created_at", "UNKNOWN")
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))

        # Store mapping from entity name to original data
        entity_id_to_original[entity_name] = entity

        entities_context.append(
            {
                "entity": entity_name,
                "type": entity.get("entity_type", "UNKNOWN"),
                "description": entity.get("description", "UNKNOWN"),
                "created_at": created_at,
                "file_path": entity.get("file_path", "unknown_source"),
            }
        )

    # Generate relations context for truncation
    relations_context = []
    for i, relation in enumerate(final_relations):
        created_at = relation.get("created_at", "UNKNOWN")
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))

        # Handle different relation data formats
        if "src_tgt" in relation:
            entity1, entity2 = relation["src_tgt"]
        else:
            entity1, entity2 = relation.get("src_id"), relation.get("tgt_id")

        # Store mapping from relation pair to original data
        relation_key = (entity1, entity2)
        relation_id_to_original[relation_key] = relation

        relations_context.append(
            {
                "entity1": entity1,
                "entity2": entity2,
                "description": relation.get("description", "UNKNOWN"),
                "created_at": created_at,
                "file_path": relation.get("file_path", "unknown_source"),
            }
        )

    logger.debug(
        f"Before truncation: {len(entities_context)} entities, {len(relations_context)} relations"
    )

    # Apply token-based truncation
    if entities_context:
        # Remove file_path and created_at for token calculation
        entities_context_for_truncation = []
        for entity in entities_context:
            entity_copy = entity.copy()
            entity_copy.pop("file_path", None)
            entity_copy.pop("created_at", None)
            entities_context_for_truncation.append(entity_copy)

        entities_context = truncate_list_by_token_size(
            entities_context_for_truncation,
            key=lambda x: "\n".join(
                json.dumps(item, ensure_ascii=False) for item in [x]
            ),
            max_token_size=max_entity_tokens,
            tokenizer=tokenizer,
        )

    if relations_context:
        # Remove file_path and created_at for token calculation
        relations_context_for_truncation = []
        for relation in relations_context:
            relation_copy = relation.copy()
            relation_copy.pop("file_path", None)
            relation_copy.pop("created_at", None)
            relations_context_for_truncation.append(relation_copy)

        relations_context = truncate_list_by_token_size(
            relations_context_for_truncation,
            key=lambda x: "\n".join(
                json.dumps(item, ensure_ascii=False) for item in [x]
            ),
            max_token_size=max_relation_tokens,
            tokenizer=tokenizer,
        )

    logger.info(
        f"After truncation: {len(entities_context)} entities, {len(relations_context)} relations"
    )

    # Create filtered original data based on truncated context
    filtered_entities = []
    filtered_entity_id_to_original = {}
    if entities_context:
        final_entity_names = {e["entity"] for e in entities_context}
        seen_nodes = set()
        for entity in final_entities:
            name = entity.get("entity_name")
            if name in final_entity_names and name not in seen_nodes:
                filtered_entities.append(entity)
                filtered_entity_id_to_original[name] = entity
                seen_nodes.add(name)

    filtered_relations = []
    filtered_relation_id_to_original = {}
    if relations_context:
        final_relation_pairs = {(r["entity1"], r["entity2"]) for r in relations_context}
        seen_edges = set()
        for relation in final_relations:
            src, tgt = relation.get("src_id"), relation.get("tgt_id")
            if src is None or tgt is None:
                src, tgt = relation.get("src_tgt", (None, None))

            pair = (src, tgt)
            if pair in final_relation_pairs and pair not in seen_edges:
                filtered_relations.append(relation)
                filtered_relation_id_to_original[pair] = relation
                seen_edges.add(pair)

    return {
        "entities_context": entities_context,
        "relations_context": relations_context,
        "filtered_entities": filtered_entities,
        "filtered_relations": filtered_relations,
        "entity_id_to_original": filtered_entity_id_to_original,
        "relation_id_to_original": filtered_relation_id_to_original,
    }

async def _attach_content_headings(
    chunks: list[dict], text_chunks_db: BaseKVStorage | None
) -> None:
    """Backfill the ``content_headings`` field onto chunks in place.

    Vector chunks never carry a ``heading`` field (chunks_vdb does not store it),
    and the round-robin merge drops the entity/relation headings too. So we look
    the chunks up by ``chunk_id`` in text_chunks storage and attach the parent
    heading path. Only chunks that actually have parent headings get the field,
    so empty paths are omitted from the JSON sent to the LLM.

    Each level is char-capped inside ``format_parent_headings`` and the joined
    path is token-budgeted against ``DEFAULT_MAX_SECTION_CONTEXT_TOKENS`` here —
    mirroring the extraction breadcrumb (see ``_truncate_section_context``). The
    query-context token truncation only keeps/drops whole chunks; it never trims
    this metadata inside a retained chunk, so a deep heading chain (the per-level
    cap bounds each level's length, not the number of levels) is bounded here.
    """
    if not text_chunks_db or not chunks:
        return
    tokenizer = text_chunks_db.global_config.get("tokenizer")
    chunk_ids = [c.get("chunk_id") for c in chunks]
    chunk_data_list = await text_chunks_db.get_by_ids(chunk_ids)
    for chunk, data in zip(chunks, chunk_data_list):
        if not isinstance(data, dict):
            continue
        headings = _truncate_section_context(
            format_parent_headings(data),
            tokenizer,
            DEFAULT_MAX_SECTION_CONTEXT_TOKENS,
        )
        if headings:
            chunk["content_headings"] = headings

async def _attach_sources(
    chunks: list[dict],
    text_chunks_db: BaseKVStorage | None,
    *,
    return_parent: bool,
    include_citations: bool,
    limit: int | None,
) -> list[dict]:
    """Final body and location of each chunk, in one lookup of the stored rows.

    ``return_parent``: isi tiap chunk diganti teks parent-nya dan child seinduk
    dilebur jadi satu, di peringkat child terbaiknya. Pencarian dan rerank tetap
    berjalan di atas child, jadi presisi tidak berubah — yang berubah hanya badan
    buktinya. Chunk tanpa ``parent_id`` (``save_parent=False``, multimodal) lewat
    apa adanya.

    ``include_citations``: tiap chunk mendapat ``page_start`` / ``page_end`` dari
    baris yang **isinya ditampilkan** — sidecar parent bila di-swap, sidecar child
    bila tidak — karena sitasi harus menunjuk teks yang benar-benar dibaca LLM.
    ``chunk_id`` / ``file_path`` tetap milik child.

    Gambar slide (``sidecar.images``, dek slide) ikut ke chunk sebagai ``images`` setiap
    kali baris tersimpannya dibaca.

    ``limit`` memotong hasil (``parent_top_k`` saat ``return_parent``; batas jumlah,
    bukan token — parent juga teks asli).
    """
    if not chunks or not text_chunks_db or not (return_parent or include_citations):
        return chunks[:limit]

    rows = [
        row if isinstance(row, dict) else {}
        for row in await text_chunks_db.get_by_ids([c.get("chunk_id") for c in chunks])
    ]
    parent_ids = [(row.get("sidecar") or {}).get("parent_id") for row in rows]

    parents: dict[str, dict] = {}
    wanted = list(dict.fromkeys(pid for pid in parent_ids if pid)) if return_parent else []
    if wanted:
        fetched = await text_chunks_db.get_by_ids(wanted)
        parents = {
            pid: row
            for pid, row in zip(wanted, fetched)
            if isinstance(row, dict) and row.get("content")
        }

    out: list[dict] = []
    seen: set[str] = set()
    for chunk, row, pid in zip(chunks, rows, parent_ids):
        parent = parents.get(pid) if pid else None
        if parent is not None:
            if pid in seen:
                continue
            seen.add(pid)
            chunk, row = {**chunk, "content": parent["content"]}, parent
        sidecar = row.get("sidecar") or {}
        if include_citations:
            chunk = {**chunk, "page_start": sidecar.get("page_start"), "page_end": sidecar.get("page_end")}
        if sidecar.get("images"):
            chunk = {**chunk, "images": sidecar["images"]}  # gambar slide halaman ini (slides.py)
        out.append(chunk)

    if return_parent:
        logger.info(
            f"[return_parent] {len(chunks)} child -> {len(out)} bukti "
            f"({len(seen)} parent), dipotong ke {limit}"
        )
    return out[:limit]


async def _merge_all_chunks(
    filtered_entities: list[dict],
    filtered_relations: list[dict],
    vector_chunks: list[dict],
    query: str = "",
    knowledge_graph_inst: BaseGraphStorage = None,
    text_chunks_db: BaseKVStorage = None,
    query_param: QueryParam = None,
    chunks_vdb: BaseVectorStorage = None,
    chunk_tracking: dict = None,
    query_embedding: list[float] = None,
) -> list[dict]:
    """
    Merge chunks from different sources: vector_chunks + entity_chunks + relation_chunks.
    """
    if chunk_tracking is None:
        chunk_tracking = {}

    # Get chunks from entities
    entity_chunks = []
    if filtered_entities and text_chunks_db:
        entity_chunks = await _find_related_text_unit_from_entities(
            filtered_entities,
            query_param,
            text_chunks_db,
            knowledge_graph_inst,
            query,
            chunks_vdb,
            chunk_tracking=chunk_tracking,
            query_embedding=query_embedding,
        )

    # Get chunks from relations
    relation_chunks = []
    if filtered_relations and text_chunks_db:
        relation_chunks = await _find_related_text_unit_from_relations(
            filtered_relations,
            query_param,
            text_chunks_db,
            entity_chunks,  # For deduplication
            query,
            chunks_vdb,
            chunk_tracking=chunk_tracking,
            query_embedding=query_embedding,
        )

    # Round-robin merge chunks from different sources with deduplication
    merged_chunks = []
    seen_chunk_ids = set()
    max_len = max(len(vector_chunks), len(entity_chunks), len(relation_chunks))
    origin_len = len(vector_chunks) + len(entity_chunks) + len(relation_chunks)

    for i in range(max_len):
        # Add from vector chunks first (Naive mode)
        if i < len(vector_chunks):
            chunk = vector_chunks[i]
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                merged_chunks.append(
                    {
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                        "chunk_id": chunk_id,
                    }
                )

        # Add from entity chunks (Local mode)
        if i < len(entity_chunks):
            chunk = entity_chunks[i]
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                merged_chunks.append(
                    {
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                        "chunk_id": chunk_id,
                    }
                )

        # Add from relation chunks (Global mode)
        if i < len(relation_chunks):
            chunk = relation_chunks[i]
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                merged_chunks.append(
                    {
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                        "chunk_id": chunk_id,
                    }
                )

    logger.info(
        f"Round-robin merged chunks: {origin_len} -> {len(merged_chunks)} (deduplicated {origin_len - len(merged_chunks)})"
    )

    # Backfill heading path (breadcrumb) onto each chunk
    if text_chunks_db and text_chunks_db.global_config.get(
        "enable_content_headings", False
    ):
        await _attach_content_headings(merged_chunks, text_chunks_db)

    return merged_chunks

# Image chunk as built by ``multimodal_template.py`` (``image_chunk``).
_IMAGE_CHUNK = re.compile(
    r"- Section Path: (?P<section>.*?)\n- Neighbor Text: (?P<neighbor>.*?)\n"
    r"Image Path: (?P<url>[^\n]*)\nCaptions: (?P<captions>[^\n]*)\n"
    r"Footnotes: (?P<footnotes>[^\n]*)\n\nVisual Analysis: (?P<analysis>.*)",
    re.S,
)


def _one_line(text: str) -> str:
    """Merged KG description as one line: ``<SEP>`` and paragraph breaks become spaces."""
    return " ".join(str(text).replace(GRAPH_FIELD_SEP, " ").split())


def _pages(chunk: dict) -> set:
    """Pages a chunk spans; a sheet name stands for itself. Empty without citations."""
    start, end = chunk.get("page_start"), chunk.get("page_end")
    if isinstance(start, int) and isinstance(end, int):
        return set(range(start, end + 1))
    return {start} if start is not None else set()


def _render_chunk(chunk: dict, source: str, covered: set) -> str:
    attrs = [("ref_id", chunk.get("reference_id")), ("source", source)]
    image = _IMAGE_CHUNK.search(chunk["content"])
    if image is None:
        body = chunk["content"].strip()
        if chunk.get("content_headings"):
            body = f"[Section: {chunk['content_headings']}]\n{body}"
    else:
        attrs += [("type", "image"), ("image_url", image["url"].strip())]
        page = chunk.get("page_start")
        # Teks tetangga dibuang hanya bila halamannya sudah dibawa text chunk lain di konteks ini.
        keep_neighbor = page is None or (chunk.get("file_path"), page) not in covered
        lines = [
            f"[Section: {image['section'].strip()}]" if image["section"].strip() != "None" else "",
            f"Neighbor Text: {image['neighbor'].strip()}" if keep_neighbor and image["neighbor"].strip() != "None" else "",
            f"Captions: {image['captions'].strip()}" if image["captions"].strip() != "None" else "",
            f"Footnotes: {image['footnotes'].strip()}" if image["footnotes"].strip() != "None" else "",
            f"Visual Analysis: {image['analysis'].strip()}",
        ]
        body = "\n".join(line for line in lines if line)
    attr_str = "".join(f' {k}="{html.escape(str(v))}"' for k, v in attrs if v)
    return f"<chunk{attr_str}>\n{body}\n</chunk>"


async def _build_context_str(
    entities_context: list[dict],
    relations_context: list[dict],
    merged_chunks: list[dict],
    query: str,
    query_param: QueryParam,
    global_config: dict[str, str],
    chunk_tracking: dict = None,
    entity_id_to_original: dict = None,
    relation_id_to_original: dict = None,
    text_chunks_db: BaseKVStorage | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Build the final LLM context string. Entities/relations were already capped
    by token in ``_apply_token_truncation``; chunks are only capped by count.
    """
    kg_context_template = TEMPLATES["chunk_only_context" if query_param.return_chunk_only else "kg_query_context"]

    # Chunk = teks asli: hanya rerank + chunk_top_k, tidak pernah dipotong per token.
    truncated_chunks = await process_chunks_unified(
        query=query,
        unique_chunks=merged_chunks,
        query_param=query_param,
        global_config=global_config,
    )

    include_citations = query_param.include_citations
    truncated_chunks = await _attach_sources(
        truncated_chunks,
        text_chunks_db,
        return_parent=query_param.return_parent,
        include_citations=include_citations,
        limit=settings.parent_top_k if query_param.return_parent else None,
    )

    reference_list: list[dict] = []
    if include_citations:
        reference_list, truncated_chunks = generate_reference_list_from_chunks(truncated_chunks)

    sources = {
        ref["reference_id"]: ref["file_path"] + (f", {ref['location']}" if ref.get("location") else "")
        for ref in reference_list
    }
    covered = {
        (chunk.get("file_path"), page)
        for chunk in truncated_chunks
        if not _IMAGE_CHUNK.search(chunk["content"])
        for page in _pages(chunk)
    }
    chunks_context = [
        _render_chunk(chunk, sources.get(chunk.get("reference_id"), ""), covered)
        for chunk in truncated_chunks
    ]

    # [Ref] only points at chunks actually in this context, so a KG fact is never cited to a missing source.
    chunk_refs = {c.get("chunk_id"): c["reference_id"] for c in truncated_chunks if c.get("reference_id")}

    def cite(original: dict | None) -> str:
        source_id = (original or {}).get("source_id") or ""
        ids = dict.fromkeys(
            chunk_refs[c] for c in split_string_by_multi_markers(source_id, [GRAPH_FIELD_SEP]) if c in chunk_refs
        )
        return f" [Ref: {', '.join(sorted(ids, key=int))}]" if ids else ""

    entities_str = "\n".join(
        f"- **{e['entity']}** ({e['type']}): {_one_line(e['description'])}"
        + cite((entity_id_to_original or {}).get(e["entity"]))
        for e in entities_context
    )
    # Edges are undirected, so no arrow.
    relations_str = "\n".join(
        f"- **{r['entity1']}** — **{r['entity2']}**: {_one_line(r['description'])}"
        + cite((relation_id_to_original or {}).get((r["entity1"], r["entity2"])))
        for r in relations_context
    )
    text_units_str = "\n\n".join(chunks_context)
    reference_list_str = "\n".join(f"[{ref_id}] {source}" for ref_id, source in sources.items())

    logger.info(
        f"Final context: {len(entities_context)} entities, {len(relations_context)} relations, {len(chunks_context)} chunks"
    )

    # not necessary to use LLM to generate a response
    # Mode chunk-only tanpa chunk = kosong, walau KG menemukan entitas/relasi.
    has_kg = not query_param.return_chunk_only and (entities_context or relations_context)
    if not has_kg and not chunks_context:
        # Return empty raw data structure when no entities/relations
        empty_raw_data = convert_to_user_format(
            [],
            [],
            [],
            [],
            "mix",
        )
        empty_raw_data["status"] = "failure"
        empty_raw_data["message"] = "Query returned empty dataset."
        return "", empty_raw_data

    # output chunks tracking infomations
    # format: <source><frequency>/<order> (e.g., E5/2 R2/1 C1/1)
    if truncated_chunks and chunk_tracking:
        chunk_tracking_log = []
        for chunk in truncated_chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id and chunk_id in chunk_tracking:
                tracking_info = chunk_tracking[chunk_id]
                source = tracking_info["source"]
                frequency = tracking_info["frequency"]
                order = tracking_info["order"]
                chunk_tracking_log.append(f"{source}{frequency}/{order}")
            else:
                chunk_tracking_log.append("?0/0")

        if chunk_tracking_log:
            logger.info(f"Final chunks S+F/O: {' '.join(chunk_tracking_log)}")

    result = kg_context_template.format(
        entities_str=entities_str,
        relations_str=relations_str,
        text_chunks_str=text_units_str,
        references_block=TEMPLATES["kg_query_references"].format(reference_list_str=reference_list_str)
        if reference_list
        else "",
    )

    # Always return both context and complete data structure (unified approach)
    logger.debug(
        f"[_build_context_str] Converting to user format: {len(entities_context)} entities, {len(relations_context)} relations, {len(truncated_chunks)} chunks"
    )
    final_data = convert_to_user_format(
        entities_context,
        relations_context,
        truncated_chunks,
        reference_list,
        "mix",
        entity_id_to_original,
        relation_id_to_original,
    )
    final_data_payload = final_data.get("data", {})
    logger.debug(
        f"[_build_context_str] Final data after conversion: {len(final_data_payload.get('entities', []))} entities, {len(final_data_payload.get('relationships', []))} relationships, {len(final_data_payload.get('chunks', []))} chunks"
    )
    return result, final_data

async def _find_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
    query: str = None,
    chunks_vdb: BaseVectorStorage = None,
    chunk_tracking: dict = None,
    query_embedding=None,
):
    """
    Find text chunks related to entities using configurable chunk selection method.

    This function supports two chunk selection strategies:
    1. WEIGHT: Linear gradient weighted polling based on chunk occurrence count
    2. VECTOR: Vector similarity-based selection using embedding cosine similarity
    """
    logger.debug(f"Finding text chunks from {len(node_datas)} entities")

    if not node_datas:
        return []

    # Step 1: Collect all text chunks for each entity
    entities_with_chunks = []
    for entity in node_datas:
        if entity.get("source_id"):
            chunks = split_string_by_multi_markers(
                entity["source_id"], [GRAPH_FIELD_SEP]
            )
            if chunks:
                entities_with_chunks.append(
                    {
                        "entity_name": entity["entity_name"],
                        "chunks": chunks,
                        "entity_data": entity,
                    }
                )

    if not entities_with_chunks:
        logger.warning("No entities with text chunks found")
        return []

    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", settings.kg_chunk_pick_method
    )
    max_related_chunks = text_chunks_db.global_config.get(
        "related_chunk_number", settings.related_chunk_number
    )

    # Step 2: Count chunk occurrences and deduplicate (keep chunks from earlier positioned entities)
    chunk_occurrence_count = {}
    for entity_info in entities_with_chunks:
        deduplicated_chunks = []
        for chunk_id in entity_info["chunks"]:
            chunk_occurrence_count[chunk_id] = (
                chunk_occurrence_count.get(chunk_id, 0) + 1
            )

            # If this is the first occurrence (count == 1), keep it; otherwise skip (duplicate from later position)
            if chunk_occurrence_count[chunk_id] == 1:
                deduplicated_chunks.append(chunk_id)
            # count > 1 means this chunk appeared in an earlier entity, so skip it

        # Update entity's chunks to deduplicated chunks
        entity_info["chunks"] = deduplicated_chunks

    # Step 3: Sort chunks for each entity by occurrence count (higher count = higher priority)
    total_entity_chunks = 0
    for entity_info in entities_with_chunks:
        sorted_chunks = sorted(
            entity_info["chunks"],
            key=lambda chunk_id: chunk_occurrence_count.get(chunk_id, 0),
            reverse=True,
        )
        entity_info["sorted_chunks"] = sorted_chunks
        total_entity_chunks += len(sorted_chunks)

    selected_chunk_ids = []  # Initialize to avoid UnboundLocalError

    # Step 4: Apply the selected chunk selection algorithm
    # Pick by vector similarity:
    #     The order of text chunks aligns with the naive retrieval's destination.
    #     When reranking is disabled, the text chunks delivered to the LLM tend to favor naive retrieval.
    if kg_chunk_pick_method == "VECTOR" and query and chunks_vdb:
        num_of_chunks = int(max_related_chunks * len(entities_with_chunks) / 2)

        # Get embedding function from global config
        actual_embedding_func = text_chunks_db.embedding_func
        if not actual_embedding_func:
            logger.warning("No embedding function found, falling back to WEIGHT method")
            kg_chunk_pick_method = "WEIGHT"
        else:
            try:
                selected_chunk_ids = await pick_by_vector_similarity(
                    query=query,
                    text_chunks_storage=text_chunks_db,
                    chunks_vdb=chunks_vdb,
                    num_of_chunks=num_of_chunks,
                    entity_info=entities_with_chunks,
                    embedding_func=actual_embedding_func,
                    query_embedding=query_embedding,
                )

                if selected_chunk_ids == []:
                    kg_chunk_pick_method = "WEIGHT"
                    logger.warning(
                        "No entity-related chunks selected by vector similarity, falling back to WEIGHT method"
                    )
                else:
                    logger.info(
                        f"Selecting {len(selected_chunk_ids)} from {total_entity_chunks} entity-related chunks by vector similarity"
                    )

            except Exception as e:
                logger.error(
                    f"Error in vector similarity sorting: {e}, falling back to WEIGHT method"
                )
                kg_chunk_pick_method = "WEIGHT"

    if kg_chunk_pick_method == "WEIGHT":
        # Pick by entity and chunk weight:
        #     When reranking is disabled, delivered more solely KG related chunks to the LLM
        selected_chunk_ids = pick_by_weighted_polling(
            entities_with_chunks, max_related_chunks, min_related_chunks=1
        )

        logger.info(
            f"Selecting {len(selected_chunk_ids)} from {total_entity_chunks} entity-related chunks by weighted polling"
        )

    if not selected_chunk_ids:
        return []

    # Step 5: Batch retrieve chunk data
    unique_chunk_ids = list(
        dict.fromkeys(selected_chunk_ids)
    )  # Remove duplicates while preserving order
    chunk_data_list = await text_chunks_db.get_by_ids(unique_chunk_ids)

    # Step 6: Build result chunks with valid data and update chunk tracking
    result_chunks = []
    for i, (chunk_id, chunk_data) in enumerate(zip(unique_chunk_ids, chunk_data_list)):
        if chunk_data is not None and "content" in chunk_data:
            chunk_data_copy = chunk_data.copy()
            chunk_data_copy["source_type"] = "entity"
            chunk_data_copy["chunk_id"] = chunk_id  # Add chunk_id for deduplication
            result_chunks.append(chunk_data_copy)

            # Update chunk tracking if provided
            if chunk_tracking is not None:
                chunk_tracking[chunk_id] = {
                    "source": "E",
                    "frequency": chunk_occurrence_count.get(chunk_id, 1),
                    "order": i + 1,  # 1-based order in final entity-related results
                }

    return result_chunks

async def _find_related_text_unit_from_relations(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    entity_chunks: list[dict] = None,
    query: str = None,
    chunks_vdb: BaseVectorStorage = None,
    chunk_tracking: dict = None,
    query_embedding=None,
):
    """
    Find text chunks related to relationships using configurable chunk selection method.

    This function supports two chunk selection strategies:
    1. WEIGHT: Linear gradient weighted polling based on chunk occurrence count
    2. VECTOR: Vector similarity-based selection using embedding cosine similarity
    """
    logger.debug(f"Finding text chunks from {len(edge_datas)} relations")

    if not edge_datas:
        return []

    # Step 1: Collect all text chunks for each relationship
    relations_with_chunks = []
    for relation in edge_datas:
        if relation.get("source_id"):
            chunks = split_string_by_multi_markers(
                relation["source_id"], [GRAPH_FIELD_SEP]
            )
            if chunks:
                # Build relation identifier
                if "src_tgt" in relation:
                    rel_key = tuple(sorted(relation["src_tgt"]))
                else:
                    rel_key = tuple(
                        sorted([relation.get("src_id"), relation.get("tgt_id")])
                    )

                relations_with_chunks.append(
                    {
                        "relation_key": rel_key,
                        "chunks": chunks,
                        "relation_data": relation,
                    }
                )

    if not relations_with_chunks:
        logger.warning("No relation-related chunks found")
        return []

    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", settings.kg_chunk_pick_method
    )
    max_related_chunks = text_chunks_db.global_config.get(
        "related_chunk_number", settings.related_chunk_number
    )

    # Step 2: Count chunk occurrences and deduplicate (keep chunks from earlier positioned relationships)
    # Also remove duplicates with entity_chunks

    # Extract chunk IDs from entity_chunks for deduplication
    entity_chunk_ids = set()
    if entity_chunks:
        for chunk in entity_chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id:
                entity_chunk_ids.add(chunk_id)

    chunk_occurrence_count = {}
    # Track unique chunk_ids that have been removed to avoid double counting
    removed_entity_chunk_ids = set()

    for relation_info in relations_with_chunks:
        deduplicated_chunks = []
        for chunk_id in relation_info["chunks"]:
            # Skip chunks that already exist in entity_chunks
            if chunk_id in entity_chunk_ids:
                # Only count each unique chunk_id once
                removed_entity_chunk_ids.add(chunk_id)
                continue

            chunk_occurrence_count[chunk_id] = (
                chunk_occurrence_count.get(chunk_id, 0) + 1
            )

            # If this is the first occurrence (count == 1), keep it; otherwise skip (duplicate from later position)
            if chunk_occurrence_count[chunk_id] == 1:
                deduplicated_chunks.append(chunk_id)
            # count > 1 means this chunk appeared in an earlier relationship, so skip it

        # Update relationship's chunks to deduplicated chunks
        relation_info["chunks"] = deduplicated_chunks

    # Check if any relations still have chunks after deduplication
    relations_with_chunks = [
        relation_info
        for relation_info in relations_with_chunks
        if relation_info["chunks"]
    ]

    if not relations_with_chunks:
        logger.info(
            f"Find no additional relations-related chunks from {len(edge_datas)} relations"
        )
        return []

    # Step 3: Sort chunks for each relationship by occurrence count (higher count = higher priority)
    total_relation_chunks = 0
    for relation_info in relations_with_chunks:
        sorted_chunks = sorted(
            relation_info["chunks"],
            key=lambda chunk_id: chunk_occurrence_count.get(chunk_id, 0),
            reverse=True,
        )
        relation_info["sorted_chunks"] = sorted_chunks
        total_relation_chunks += len(sorted_chunks)

    logger.info(
        f"Find {total_relation_chunks} additional chunks in {len(relations_with_chunks)} relations (deduplicated {len(removed_entity_chunk_ids)})"
    )

    # Step 4: Apply the selected chunk selection algorithm
    selected_chunk_ids = []  # Initialize to avoid UnboundLocalError

    if kg_chunk_pick_method == "VECTOR" and query and chunks_vdb:
        num_of_chunks = int(max_related_chunks * len(relations_with_chunks) / 2)

        # Get embedding function from global config
        actual_embedding_func = text_chunks_db.embedding_func
        if not actual_embedding_func:
            logger.warning("No embedding function found, falling back to WEIGHT method")
            kg_chunk_pick_method = "WEIGHT"
        else:
            try:
                selected_chunk_ids = await pick_by_vector_similarity(
                    query=query,
                    text_chunks_storage=text_chunks_db,
                    chunks_vdb=chunks_vdb,
                    num_of_chunks=num_of_chunks,
                    entity_info=relations_with_chunks,
                    embedding_func=actual_embedding_func,
                    query_embedding=query_embedding,
                )

                if selected_chunk_ids == []:
                    kg_chunk_pick_method = "WEIGHT"
                    logger.warning(
                        "No relation-related chunks selected by vector similarity, falling back to WEIGHT method"
                    )
                else:
                    logger.info(
                        f"Selecting {len(selected_chunk_ids)} from {total_relation_chunks} relation-related chunks by vector similarity"
                    )

            except Exception as e:
                logger.error(
                    f"Error in vector similarity sorting: {e}, falling back to WEIGHT method"
                )
                kg_chunk_pick_method = "WEIGHT"

    if kg_chunk_pick_method == "WEIGHT":
        # Apply linear gradient weighted polling algorithm
        selected_chunk_ids = pick_by_weighted_polling(
            relations_with_chunks, max_related_chunks, min_related_chunks=1
        )

        logger.info(
            f"Selecting {len(selected_chunk_ids)} from {total_relation_chunks} relation-related chunks by weighted polling"
        )

    logger.debug(
        f"KG related chunks: {len(entity_chunks)} from entitys, {len(selected_chunk_ids)} from relations"
    )

    if not selected_chunk_ids:
        return []

    # Step 5: Batch retrieve chunk data
    unique_chunk_ids = list(
        dict.fromkeys(selected_chunk_ids)
    )  # Remove duplicates while preserving order
    chunk_data_list = await text_chunks_db.get_by_ids(unique_chunk_ids)

    # Step 6: Build result chunks with valid data and update chunk tracking
    result_chunks = []
    for i, (chunk_id, chunk_data) in enumerate(zip(unique_chunk_ids, chunk_data_list)):
        if chunk_data is not None and "content" in chunk_data:
            chunk_data_copy = chunk_data.copy()
            chunk_data_copy["source_type"] = "relationship"
            chunk_data_copy["chunk_id"] = chunk_id  # Add chunk_id for deduplication
            result_chunks.append(chunk_data_copy)

            # Update chunk tracking if provided
            if chunk_tracking is not None:
                chunk_tracking[chunk_id] = {
                    "source": "R",
                    "frequency": chunk_occurrence_count.get(chunk_id, 1),
                    "order": i + 1,  # 1-based order in final relation-related results
                }

    return result_chunks
