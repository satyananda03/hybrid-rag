from hybrid_rag.core.concurrency import _cooperative_yield
import asyncio
import json
from contextlib import asynccontextmanager
import re
from typing import Any
from collections import Counter, defaultdict
from hybrid_rag.core.exceptions import (
    PipelineCancelledException,
)
from hybrid_rag.core.source_ids import apply_source_ids_limit, merge_source_ids
from hybrid_rag.ingestion.utils import (
    INVALID_OUTPUT_ERRORS,
    create_prefixed_exception,
    pack_user_ass_to_openai_messages,
    update_chunk_cache_list,
)
from hybrid_rag.ingestion.text_cleaning import sanitize_and_normalize_extracted_text
from hybrid_rag.core.utils import (
    compute_mdhash_id,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
)
from hybrid_rag.core.tokenizer import Tokenizer
from hybrid_rag.ingestion.text_cleaning import sanitize_text_for_encoding
from hybrid_rag.ingestion.utils import safe_vdb_operation_with_exception
from hybrid_rag.core.source_ids import make_relation_chunk_key
import logging
from hybrid_rag import progress

from hybrid_rag.core.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
)
from hybrid_rag.core.chunk_schema import (
    format_heading_context,
    strip_internal_multimodal_markup_for_extraction,
)
# Every prompt that reaches a model lives in knowledge_graph.yaml and is wired
# into chains by knowledge_graph.chain. What is left in TEMPLATES is the two context templates
# that serialize retrieved entities/relations/chunks — formatting, not
# instructions.
from hybrid_rag.ingestion.knowledge_graph.chain import (
    get_kg_chains,
    summarize,
)
from hybrid_rag.ingestion.knowledge_graph.schema import (
    EntityExtractionOutput,
    ExtractedProperty,
    OntologyExtractionOutput,
)
from hybrid_rag.ingestion.knowledge_graph.ontology_schema import (
    load_ontology_schema,
    prune_chunk,
)
from langchain_core.runnables import Runnable
from pydantic import ValidationError
from hybrid_rag.core.llm_cache import (
    CACHE_CHUNK_ID,
    CACHE_KEYS,
    CACHE_KV,
)
from hybrid_rag.core.constants import (
    GRAPH_FIELD_SEP,
    DEFAULT_MAX_SECTION_CONTEXT_TOKENS,
    SOURCE_IDS_LIMIT_METHOD_KEEP,
    SOURCE_IDS_LIMIT_METHOD_FIFO,
    DEFAULT_FILE_PATH_MORE_PLACEHOLDER,
    DEFAULT_ENTITY_NAME_MAX_LENGTH,
    DEFAULT_ENTITY_NAME_MAX_BYTES,
)
from hybrid_rag.storage.shared import get_storage_keyed_lock
from hybrid_rag.ingestion.knowledge_graph.merge_session import MergeSession
from hybrid_rag.ingestion.knowledge_graph.entity_repairment import (
    accept_repair,
    repair_inputs,
    unlinked_names,
)
from hybrid_rag.ingestion.knowledge_graph.entity_resolution import (
    EntityResolver,
    rewrite as rewrite_entity_names,
)
from hybrid_rag.core.chunk_schema import _truncate_section_context
import time
from hybrid_rag.config import settings

logger = logging.getLogger(__name__)


def _get_relationship_vdb_timeout_seconds(global_config: dict[str, Any]) -> float:
    """Derive a defensive timeout for relation VDB upserts.

    Rationale:
    - `knowledge_graph_inst.upsert_edge()` for the default NetworkX storage is in-memory and fast.
    - `relationships_vdb.upsert()` performs embedding calls and remote I/O, which is the more likely
      point of silent stalls during relation merge.
    """
    configured = global_config.get("default_embedding_timeout")
    try:
        base_timeout = float(configured)
    except (TypeError, ValueError):
        base_timeout = 30.0
    # Keep a fixed lower bound high enough to avoid false positives on slow providers.
    return max(base_timeout * 3, 120.0)


def _format_relation_edge_label(edge_key: tuple[str, str] | list[str]) -> str:
    if isinstance(edge_key, tuple):
        left, right = edge_key
    else:
        left, right = edge_key[0], edge_key[1]
    return f"{left}->{right}"


def _truncate_entity_identifier(
    identifier: str,
    limit: int,
    chunk_key: str,
    identifier_role: str,
    byte_limit: int = DEFAULT_ENTITY_NAME_MAX_BYTES,
) -> str:
    """Truncate entity identifiers that exceed the configured length limit.

    Enforces both a character limit (``limit``) and a UTF-8 byte limit
    (``byte_limit``). Milvus validates VARCHAR ``max_length`` in BYTES, not
    characters, so a CJK identifier within the character limit can still
    overflow the field (e.g. 256 Chinese chars ~= 694 bytes > 512). The byte
    truncation cuts on a character boundary so the result stays valid UTF-8.
    """

    char_len = len(identifier)
    byte_len = len(identifier.encode("utf-8"))
    if char_len <= limit and byte_len <= byte_limit:
        return identifier

    display_value = identifier[:limit]
    encoded = display_value.encode("utf-8")
    if len(encoded) > byte_limit:
        # Drop the partial trailing multi-byte char left by the byte slice.
        display_value = encoded[:byte_limit].decode("utf-8", errors="ignore")
    preview = identifier[:50]  # Show first 50 characters as preview
    logger.warning(
        "%s: %s len %d chars / %d bytes > %d chars / %d bytes (Name: '%s...')",
        chunk_key,
        identifier_role,
        char_len,
        byte_len,
        limit,
        byte_limit,
        preview,
    )
    return display_value


def _truncate_vdb_content(content: str, global_config: dict, content_label: str) -> str:
    """Clamp vector-store payload size to stay under embedding limits."""

    if not content:
        return content

    embedding_token_limit = global_config.get("embedding_token_limit")
    tokenizer: Tokenizer | None = global_config.get("tokenizer")
    if embedding_token_limit is None or tokenizer is None:
        return content

    threshold = int(embedding_token_limit)
    if threshold <= 0:
        return content

    tokens = tokenizer.encode(content)
    if len(tokens) <= threshold:
        return content

    # Leave headroom because tokenizer behavior can differ slightly from the provider.
    effective_limit = max(threshold - min(256, max(32, threshold // 16)), 1)
    truncated_content = tokenizer.decode(tokens[:effective_limit])
    logger.warning(
        "%s VDB content truncated from %d to %d tokens (embedding limit: %d)",
        content_label,
        len(tokens),
        effective_limit,
        threshold,
    )
    return truncated_content

_MM_DISPLAY_NAME_PATTERN = re.compile(
    r"^\[(?:Image|Table|Equation) Name\](.+)$",
    flags=re.MULTILINE,
)


def _parse_mm_display_name(content: str, fallback: str) -> str:
    """Return the friendly name embedded in a multimodal chunk.

    Matches the leading ``[Image Name]…`` / ``[Table Name]…`` /
    ``[Equation Name]…`` segment produced by
    the multimodal chunk producer (``multimodal/analyze.py``). Falls
    back to the sidecar id when the segment is missing or empty so
    callers never end up with a blank label.
    """
    if content:
        match = _MM_DISPLAY_NAME_PATTERN.search(content)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate
    return fallback


async def _handle_entity_relation_summary(
    description_type: str,
    entity_or_relation_name: str,
    description_list: list[str],
    separator: str,
    global_config: dict,
    llm_response_cache: BaseKVStorage | None = None,
) -> tuple[str, bool]:
    """Handle entity relation description summary using map-reduce approach.

    This function summarizes a list of descriptions using a map-reduce strategy:
    1. If total tokens < summary_context_size and len(description_list) < force_llm_summary_on_merge, no need to summarize
    2. If total tokens < summary_max_tokens, summarize with LLM directly
    3. Otherwise, split descriptions into chunks that fit within token limits
    4. Summarize each chunk, then recursively process the summaries
    5. Continue until we get a final summary within token limits or num of descriptions is less than force_llm_summary_on_merge

    Args:
        entity_or_relation_name: Name of the entity or relation being summarized
        description_list: List of description strings to summarize
        global_config: Global configuration containing tokenizer and limits
        llm_response_cache: Optional cache for LLM responses

    Returns:
        Tuple of (final_summarized_description_string, llm_was_used_boolean)
    """
    # Handle empty input
    if not description_list:
        return "", False

    # If only one description, return it directly (no need for LLM call)
    # Still sanitize: descriptions read back from existing graph nodes (or
    # injected by non-extraction producers) may carry XML-illegal control
    # characters that would crash the GraphML flush downstream.
    if len(description_list) == 1:
        return sanitize_text_for_encoding(description_list[0]), False

    # Get configuration
    tokenizer: Tokenizer = global_config["tokenizer"]
    summary_context_size = global_config["summary_context_size"]
    summary_max_tokens = global_config["summary_max_tokens"]
    force_llm_summary_on_merge = global_config["force_llm_summary_on_merge"]

    current_list = description_list[:]  # Copy the list to avoid modifying original
    llm_was_used = False  # Track whether LLM was used during the entire process

    # Iterative map-reduce process
    while True:
        # Calculate total tokens in current list while periodically yielding so
        # a large merge does not monopolize the event loop in single-worker mode.
        total_tokens = 0
        for i, desc in enumerate(current_list, start=1):
            total_tokens += len(tokenizer.encode(desc))
            await _cooperative_yield(i, every=32)

        # If total length is within limits, perform final summarization
        if total_tokens <= summary_context_size or len(current_list) <= 2:
            if (
                len(current_list) < force_llm_summary_on_merge
                and total_tokens < summary_max_tokens
            ):
                # no LLM needed, just join the descriptions
                # Sanitize for the same reason as the single-description
                # early return above: inputs merged from pre-existing graph
                # data are not guaranteed XML-safe.
                final_description = sanitize_text_for_encoding(
                    separator.join(current_list)
                )
                return final_description if final_description else "", llm_was_used
            else:
                if total_tokens > summary_context_size and len(current_list) <= 2:
                    logger.warning(
                        f"Summarizing {entity_or_relation_name}: Oversize description found"
                    )
                # Final summarization of remaining descriptions - LLM will be used
                final_summary = await _summarize_descriptions(
                    description_type,
                    entity_or_relation_name,
                    current_list,
                    global_config,
                    llm_response_cache,
                )
                return final_summary, True  # LLM was used for final summarization

        # Need to split into chunks - Map phase
        # Ensure each chunk has minimum 2 descriptions to guarantee progress
        chunks = []
        current_chunk = []
        current_tokens = 0

        # Currently least 3 descriptions in current_list
        for i, desc in enumerate(current_list, start=1):
            desc_tokens = len(tokenizer.encode(desc))
            await _cooperative_yield(i, every=32)

            # If adding current description would exceed limit, finalize current chunk
            if current_tokens + desc_tokens > summary_context_size and current_chunk:
                # Ensure we have at least 2 descriptions in the chunk (when possible)
                if len(current_chunk) == 1:
                    # Force add one more description to ensure minimum 2 per chunk
                    current_chunk.append(desc)
                    chunks.append(current_chunk)
                    logger.warning(
                        f"Summarizing {entity_or_relation_name}: Oversize description found"
                    )
                    current_chunk = []  # next group is empty
                    current_tokens = 0
                else:  # curren_chunk is ready for summary in reduce phase
                    chunks.append(current_chunk)
                    current_chunk = [desc]  # leave it for next group
                    current_tokens = desc_tokens
            else:
                current_chunk.append(desc)
                current_tokens += desc_tokens

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)

        logger.info(
            f"   Summarizing {entity_or_relation_name}: Map {len(current_list)} descriptions into {len(chunks)} groups"
        )
        progress.activity(f"Merangkum {entity_or_relation_name} ({len(current_list)} deskripsi)")

        # Reduce phase: summarize each group from chunks
        new_summaries = []
        for i, chunk in enumerate(chunks, start=1):
            if len(chunk) == 1:
                # Optimization: single description chunks don't need LLM summarization
                new_summaries.append(chunk[0])
            else:
                # Multiple descriptions need LLM summarization
                summary = await _summarize_descriptions(
                    description_type,
                    entity_or_relation_name,
                    chunk,
                    global_config,
                    llm_response_cache,
                )
                new_summaries.append(summary)
                llm_was_used = True  # Mark that LLM was used in reduce phase

        # Update current list with new summaries for next iteration
        current_list = new_summaries


async def _summarize_descriptions(
    description_type: str,
    description_name: str,
    description_list: list[str],
    global_config: dict,
    llm_response_cache: BaseKVStorage | None = None,
) -> str:
    """Helper function to summarize a list of descriptions using LLM.

    Args:
        entity_or_relation_name: Name of the entity or relation being summarized
        descriptions: List of description strings to summarize
        global_config: Global configuration containing LLM function and settings
        llm_response_cache: Optional cache for LLM responses

    Returns:
        Summarized description string
    """
    summary_length_recommended = global_config["summary_length_recommended"]

    # Convert descriptions to JSONL format and apply token-based truncation
    tokenizer = global_config["tokenizer"]
    summary_context_size = global_config["summary_context_size"]

    # Create list of JSON objects with "Description" field
    json_descriptions = [{"Description": desc} for desc in description_list]

    # Use truncate_list_by_token_size for length truncation
    truncated_json_descriptions = truncate_list_by_token_size(
        json_descriptions,
        key=lambda x: json.dumps(x, ensure_ascii=False),
        max_token_size=summary_context_size,
        tokenizer=tokenizer,
    )

    # Convert to JSONL format (one JSON object per line)
    joined_descriptions = "\n".join(
        json.dumps(desc, ensure_ascii=False) for desc in truncated_json_descriptions
    )

    summary = await summarize(
        {
            "description_type": description_type,
            "description_name": description_name,
            "description_list": joined_descriptions,
            "summary_length": summary_length_recommended,
        },
        config={"configurable": {CACHE_KV: llm_response_cache}},
    )

    # The LLM response is the only description path that bypasses
    # extraction-time sanitization; control chars / surrogates left here
    # would later break GraphML (XML) serialization on write. Strip them
    # at the source, symmetric with how extracted descriptions are cleaned.
    summary = sanitize_text_for_encoding(summary)

    # Check summary token length against embedding limit
    embedding_token_limit = global_config.get("embedding_token_limit")
    if embedding_token_limit is not None and summary:
        tokenizer = global_config["tokenizer"]
        summary_token_count = len(tokenizer.encode(summary))
        threshold = int(embedding_token_limit)

        if summary_token_count > threshold:
            logger.warning(
                f"Summary tokens({summary_token_count}) exceeds embedding_token_limit({embedding_token_limit}) "
                f" for {description_type}: {description_name}"
            )

    return summary


#: Every dash glyph a PDF or a model emits for a range ("2025–2029", "2025‑2029"),
#: folded onto "-": the name is the node key, so a glyph must not split one entity
#: into two nodes.
_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—―−﹘﹣－"})


def _title_keep_acronyms(text: str) -> str:
    """Title-case each word but leave all-caps words of 2+ letters alone (ICCN, UU, UMKM)."""
    return re.sub(r"[^\W\d_]+", lambda m: m[0] if len(m[0]) > 1 and m[0].isupper() else m[0].capitalize(), text)


def _clean_identifier(raw: str) -> str:
    """Normalize an entity name the way the KG stores it: sanitized, dashes folded,
    title case with acronyms kept in ALL CAPS (the extraction prompt asks for that).

    Entity names and both relationship endpoints all pass through here, so a node
    key and the edges pointing at it cannot drift apart. Canonical naming (e.g.
    "Pasal 10") is the extraction prompt's job: see the ontology's class comments.
    """
    name = sanitize_and_normalize_extracted_text(raw, remove_inner_quotes=True).translate(_DASHES)
    return _title_keep_acronyms(name)


def _parse_properties(properties: list[ExtractedProperty]) -> dict[str, str]:
    """An ontology-mode name/value list as the dict the node or edge stores."""
    parsed = {}
    for prop in properties:
        name = sanitize_and_normalize_extracted_text(prop.name, remove_inner_quotes=True)
        # Encoding cleanup only: the name normalizer drops short numeric text,
        # which as a value is often the point ("Pasal 9" -> articleNumber "9").
        value = sanitize_text_for_encoding(prop.value).strip()
        if name and value:
            parsed[name] = value
    return parsed


#: Keys the KG itself writes on nodes and edges. An ontology property with one of
#: these names is stored as ``ontology_<name>`` rather than overwrite it.
_RESERVED_GRAPH_KEYS = frozenset(
    {
        "entity_id", "entity_type", "description", "source_id", "file_path",
        "created_at", "truncate", "workspace", "weight", "keywords",
        "relation_type", "relation_triples", "aliases",
    }
)


def _ontology_properties(records: list[dict], existing: dict | None) -> dict:
    """Ontology properties of the records not yet on the stored node/edge.

    First value wins, the stored one before any new record's — what neo4j-graphrag's
    entity resolution does when it merges nodes (``apoc.refactor.mergeNodes`` with
    ``properties: 'discard'``). Returning only absent keys keeps the stored values,
    since the graph write merges (``SET n += props``).
    """
    existing = existing or {}
    merged: dict = {}
    for record in records:
        for name, value in (record.get("properties") or {}).items():
            key = f"ontology_{name}" if name in _RESERVED_GRAPH_KEYS else name
            if existing.get(key) is None:
                merged.setdefault(key, value)
    return merged


def _relation_typing(edges: list[dict], existing: dict | None) -> dict:
    """``relation_type`` / ``relation_triples`` of one undirected edge, merged.

    The graph keeps one undirected edge per entity pair, so the ontology types
    and directions it carries are accumulated like keywords: every type seen, and
    every ``"Source|type|Target"`` in its validated direction. Empty for edges
    extracted without an ontology, so those keep exactly their old properties.
    """
    existing = existing or {}
    types = {t for t in (existing.get("relation_type") or "").split(",") if t}
    triples = set(existing.get("relation_triples") or [])
    for edge in edges:
        if edge.get("relation_type"):
            types.add(edge["relation_type"])
            triples.add(f"{edge['src_id']}|{edge['relation_type']}|{edge['tgt_id']}")
    if not types:
        return {}
    return {"relation_type": ",".join(sorted(types)), "relation_triples": sorted(triples)}


async def _extract_or_none(
    chain: Runnable,
    inputs: dict[str, Any],
    config: dict[str, Any],
    chunk_key: str,
    pass_name: str,
) -> EntityExtractionOutput | None:
    """Run one extraction pass; ``None`` when its output is still unusable after one retry.

    ``None`` means "no records from this pass", not an entity:
    :func:`_parse_extraction_output` turns it into no nodes and no edges. The
    chunk's text and vector are already stored, so it stays findable by vector
    search; only its graph contribution is missing. A failed reply is never
    cached, so the retry really asks the model again.

    Only invalid *output* is caught — the contract broken as a whole (invalid
    JSON, no ``entities`` array); a single broken record never gets here, the
    schema drops it. Infrastructure errors (timeouts, 5xx, auth)
    still fail the document: skipping those would let a dead provider "succeed"
    with an empty graph.
    """
    for attempt in (1, 2):
        try:
            return await chain.ainvoke(inputs, config=config)
        except INVALID_OUTPUT_ERRORS as e:
            logger.warning(
                f"{chunk_key}: invalid {pass_name} output (attempt {attempt}/2): {str(e)[:500]}"
            )
    progress.activity(
        f"Ekstraksi KG satu chunk dilewati: output model tidak valid ({chunk_key}, {pass_name})",
        level="warning",
    )
    return None


def _parse_extraction_output(
    parsed: EntityExtractionOutput | None,
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
) -> tuple[dict, dict]:
    """Turn one chain result into KG nodes and edges.

    ``parsed`` is what the extraction chain returned, already validated against
    its output schema, so this only applies the KG's own rules: names are
    title-cased and length-capped, types are normalized, and records without a
    usable name, type or description are dropped rather than stored half-formed.

    ``None`` (structured output failed) yields no records for the chunk — the
    same outcome as an empty extraction, so one bad chunk never fails the
    document.

    An ``OntologyExtractionOutput`` keeps the type exactly as the ontology spells
    it (the pruning step matches labels verbatim) and also carries the records'
    ``properties`` and each relationship's ``relation_type``.
    """
    ontology = isinstance(parsed, OntologyExtractionOutput)
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)

    if parsed is None:
        logger.warning(f"{chunk_key}: extraction chain returned no structured output")
        return dict(maybe_nodes), dict(maybe_edges)

    for entity in parsed.entities:
        entity_name = _clean_identifier(entity.name)
        if not entity_name:
            logger.info(f"{chunk_key}: dropping entity with empty name")
            continue

        entity_type = sanitize_and_normalize_extracted_text(
            entity.type, remove_inner_quotes=True
        )
        # A type carrying delimiters would corrupt the GraphML / VDB fields it
        # is written into, so reject rather than sanitize into something else.
        if not entity_type.strip() or any(
            char in entity_type for char in "'()<>|/\\"
        ):
            logger.warning(
                f"{chunk_key}: invalid entity type {entity_type!r} for {entity_name!r}"
            )
            continue

        description = sanitize_and_normalize_extracted_text(entity.description)
        if not description.strip():
            logger.warning(f"{chunk_key}: empty description for entity {entity_name!r}")
            continue

        truncated_name = _truncate_entity_identifier(
            entity_name, DEFAULT_ENTITY_NAME_MAX_LENGTH, chunk_key, "Entity name"
        )
        record = dict(
            entity_name=truncated_name,
            entity_type=entity_type if ontology else entity_type.replace(" ", "").lower(),
            description=description,
            source_id=chunk_key,
            file_path=file_path,
            timestamp=timestamp,
        )
        if ontology:
            record["properties"] = _parse_properties(entity.properties)
        maybe_nodes[truncated_name].append(record)

    for rel in parsed.relationships:
        source = _clean_identifier(rel.source)
        target = _clean_identifier(rel.target)
        if not source or not target:
            logger.info(f"{chunk_key}: dropping relationship with an empty endpoint")
            continue
        if source == target:
            logger.debug(f"{chunk_key}: self-relation on {source!r}")
            continue

        description = sanitize_and_normalize_extracted_text(rel.description)
        if not description.strip():
            logger.warning(
                f"{chunk_key}: empty description for {source!r} ~ {target!r}, skipping"
            )
            continue

        truncated_source = _truncate_entity_identifier(
            source, DEFAULT_ENTITY_NAME_MAX_LENGTH, chunk_key, "Relation entity"
        )
        truncated_target = _truncate_entity_identifier(
            target, DEFAULT_ENTITY_NAME_MAX_LENGTH, chunk_key, "Relation entity"
        )
        record = dict(
            src_id=truncated_source,
            tgt_id=truncated_target,
            weight=1.0,
            description=description,
            # Full-width comma: models writing CJK-style punctuation would
            # otherwise produce one unsplittable keyword blob downstream.
            keywords=sanitize_and_normalize_extracted_text(
                rel.keywords, remove_inner_quotes=True
            ).replace("\uff0c", ","),
            source_id=chunk_key,
            file_path=file_path,
            timestamp=timestamp,
        )
        if ontology:
            record["relation_type"] = sanitize_and_normalize_extracted_text(
                rel.type, remove_inner_quotes=True
            )
            record["properties"] = _parse_properties(rel.properties)
        maybe_edges[(truncated_source, truncated_target)].append(record)

    return dict(maybe_nodes), dict(maybe_edges)


async def _units_ingested_without_resolution(
    unit_ids: list[str],
    text_chunks_storage: BaseKVStorage,
    doc_status_storage: BaseKVStorage | None,
) -> set[str]:
    """Extraction units whose document was ingested with ``entity_resolution=False``."""
    if doc_status_storage is None or not unit_ids:
        return set()
    units = await text_chunks_storage.get_by_ids(unit_ids)
    doc_of = {
        unit_id: row["full_doc_id"]
        for unit_id, row in zip(unit_ids, units)
        if isinstance(row, dict) and row.get("full_doc_id")
    }
    doc_ids = list(dict.fromkeys(doc_of.values()))
    rows = await doc_status_storage.get_by_ids(doc_ids)
    off = {
        doc_id
        for doc_id, row in zip(doc_ids, rows)
        if ((row or {}).get("metadata") or {}).get("entity_resolution", True) is False
    }
    return {unit_id for unit_id, doc_id in doc_of.items() if doc_id in off}


async def rebuild_knowledge_from_chunks(
    entities_to_rebuild: dict[str, list[str]],
    relationships_to_rebuild: dict[tuple[str, str], list[str]],
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_storage: BaseKVStorage,
    llm_response_cache: BaseKVStorage,
    global_config: dict[str, str],
    entity_chunks_storage: BaseKVStorage | None = None,
    relation_chunks_storage: BaseKVStorage | None = None,
    doc_status_storage: BaseKVStorage | None = None,
    entity_resolver: EntityResolver | None = None,
) -> None:
    """Rebuild entity and relationship descriptions from cached extraction results with parallel processing

    This method uses cached LLM extraction results instead of calling LLM again,
    following the same approach as the insert process. Now with parallel processing
    controlled by llm_model_max_async and using get_storage_keyed_lock for data consistency.

    Args:
        entities_to_rebuild: Dict mapping entity_name -> list of remaining chunk_ids
        relationships_to_rebuild: Dict mapping (src, tgt) -> list of remaining chunk_ids
        knowledge_graph_inst: Knowledge graph storage
        entities_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        text_chunks_storage: Text chunks storage
        llm_response_cache: LLM response cache
        global_config: Global configuration containing llm_model_max_async
        pipeline_status: Pipeline status dictionary
        pipeline_status_lock: Lock for pipeline status
        entity_chunks_storage: KV storage maintaining full chunk IDs per entity
        relation_chunks_storage: KV storage maintaining full chunk IDs per relation
        doc_status_storage: Document status storage; a chunk whose document was
            ingested under an ontology (``metadata.kg_ontology``) is re-pruned
            against it, so the rebuild reproduces what extraction kept
        entity_resolver: Maps the re-parsed raw names onto the canonical names
            ingest wrote (registry lookup only), so they match ``entities_to_rebuild``
    """
    if not entities_to_rebuild and not relationships_to_rebuild:
        return

    # Get all referenced chunk IDs
    all_referenced_chunk_ids = set()
    for chunk_ids in entities_to_rebuild.values():
        all_referenced_chunk_ids.update(chunk_ids)
    for chunk_ids in relationships_to_rebuild.values():
        all_referenced_chunk_ids.update(chunk_ids)

    status_message = f"Rebuilding knowledge from {len(all_referenced_chunk_ids)} cached chunk extractions (parallel processing)"
    logger.info(status_message)

    # Extraction ran per unit (a parent or a multimodal chunk), not per child
    # chunk the graph points at, so the cache is looked up by unit.
    extraction_units = await _resolve_extraction_units(
        text_chunks_storage, all_referenced_chunk_ids
    )
    # cached_results: unit_id -> [(extraction_result, create_time, file_path)], see _get_cached_extraction_results for the ordering
    cached_results = await _get_cached_extraction_results(
        llm_response_cache,
        set(extraction_units.values()),
        text_chunks_storage=text_chunks_storage,
        doc_status_storage=doc_status_storage,
    )

    entities_to_rebuild = _keep_rebuildable(
        entities_to_rebuild, extraction_units, cached_results, kind="entity"
    )
    relationships_to_rebuild = _keep_rebuildable(
        relationships_to_rebuild, extraction_units, cached_results, kind="relation"
    )
    if not entities_to_rebuild and not relationships_to_rebuild:
        return

    # Process cached results to get entities and relationships for each chunk
    chunk_entities = {}  # chunk_id -> {entity_name: [entity_data]}
    chunk_relationships = {}  # chunk_id -> {(src, tgt): [relationship_data]}

    for chunk_id, results in cached_results.items():
        try:
            # Handle multiple extraction results per chunk
            chunk_entities[chunk_id] = defaultdict(list)
            chunk_relationships[chunk_id] = defaultdict(list)

            # process multiple LLM extraction results for a single chunk_id
            for extraction_result, timestamp, file_path, _ontology in results:
                entities, relationships = _parse_extraction_output(
                    extraction_result, chunk_id, timestamp, file_path
                )

                # Merge entities and relationships from this extraction result
                # Compare description lengths and keep the better version for the same chunk_id
                for entity_name, entity_list in entities.items():
                    if entity_name not in chunk_entities[chunk_id]:
                        # New entity for this chunk_id
                        chunk_entities[chunk_id][entity_name].extend(entity_list)
                    elif len(chunk_entities[chunk_id][entity_name]) == 0:
                        # Empty list, add the new entities
                        chunk_entities[chunk_id][entity_name].extend(entity_list)
                    else:
                        # Compare description lengths and keep the better one
                        existing_desc_len = len(
                            chunk_entities[chunk_id][entity_name][0].get(
                                "description", ""
                            )
                            or ""
                        )
                        new_desc_len = len(entity_list[0].get("description", "") or "")

                        if new_desc_len > existing_desc_len:
                            # Replace with the new entity that has longer description
                            chunk_entities[chunk_id][entity_name] = list(entity_list)
                        # Otherwise keep existing version

                # Compare description lengths and keep the better version for the same chunk_id
                for rel_key, rel_list in relationships.items():
                    if rel_key not in chunk_relationships[chunk_id]:
                        # New relationship for this chunk_id
                        chunk_relationships[chunk_id][rel_key].extend(rel_list)
                    elif len(chunk_relationships[chunk_id][rel_key]) == 0:
                        # Empty list, add the new relationships
                        chunk_relationships[chunk_id][rel_key].extend(rel_list)
                    else:
                        # Compare description lengths and keep the better one
                        existing_desc_len = len(
                            chunk_relationships[chunk_id][rel_key][0].get(
                                "description", ""
                            )
                            or ""
                        )
                        new_desc_len = len(rel_list[0].get("description", "") or "")

                        if new_desc_len > existing_desc_len:
                            # Replace with the new relationship that has longer description
                            chunk_relationships[chunk_id][rel_key] = list(rel_list)
                        # Otherwise keep existing version

            # Same validation as extraction ran on this unit (see extract_entities)
            ontology = results[0][3]
            if ontology:
                nodes, edges = await prune_chunk(
                    dict(chunk_entities[chunk_id]),
                    dict(chunk_relationships[chunk_id]),
                    load_ontology_schema(ontology),
                    chunk_id,
                )
                chunk_entities[chunk_id] = defaultdict(list, nodes)
                chunk_relationships[chunk_id] = defaultdict(list, edges)

        except Exception as e:
            status_message = (
                f"Failed to parse cached extraction result for chunk {chunk_id}: {e}"
            )
            logger.info(status_message)  # Per requirement, change to info
            continue

    # The cache holds raw names; ingest wrote canonical ones. Without this the
    # lookups below miss and a deletion brings the alias names back. A document
    # ingested with entity_resolution=False wrote its raw names, so it keeps them.
    if entity_resolver is not None:
        unresolved = await _units_ingested_without_resolution(
            list(chunk_entities), text_chunks_storage, doc_status_storage
        )
        resolved = [c for c in chunk_entities if c not in unresolved]
        mapping = await entity_resolver.canonical_names(chunk_entities[c] for c in resolved)
        for chunk_id in resolved:
            chunk_entities[chunk_id], chunk_relationships[chunk_id] = rewrite_entity_names(
                chunk_entities[chunk_id], chunk_relationships[chunk_id], mapping
            )

    # Get max async tasks limit from global_config for semaphore control
    graph_max_async = global_config.get("llm_model_max_async", 4) * 2
    semaphore = asyncio.Semaphore(graph_max_async)

    # Counters for tracking progress
    rebuilt_entities_count = 0
    rebuilt_relationships_count = 0
    failed_entities_count = 0
    failed_relationships_count = 0

    async def _locked_rebuild_entity(entity_name, chunk_ids):
        nonlocal rebuilt_entities_count, failed_entities_count
        async with semaphore:
            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            async with get_storage_keyed_lock(
                [entity_name], namespace=namespace
            ):
                try:
                    await _rebuild_single_entity(
                        knowledge_graph_inst=knowledge_graph_inst,
                        entities_vdb=entities_vdb,
                        entity_name=entity_name,
                        chunk_ids=chunk_ids,
                        chunk_entities=chunk_entities,
                        extraction_units=extraction_units,
                        llm_response_cache=llm_response_cache,
                        global_config=global_config,
                        entity_chunks_storage=entity_chunks_storage,
                    )
                    rebuilt_entities_count += 1
                except Exception as e:
                    failed_entities_count += 1
                    status_message = f"Failed to rebuild `{entity_name}`: {e}"
                    logger.info(status_message)  # Per requirement, change to info

    async def _locked_rebuild_relationship(src, tgt, chunk_ids):
        nonlocal rebuilt_relationships_count, failed_relationships_count
        async with semaphore:
            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            # Sort src and tgt to ensure order-independent lock key generation
            sorted_key_parts = sorted([src, tgt])
            async with get_storage_keyed_lock(
                sorted_key_parts,
                namespace=namespace,
            ):
                try:
                    await _rebuild_single_relationship(
                        knowledge_graph_inst=knowledge_graph_inst,
                        relationships_vdb=relationships_vdb,
                        entities_vdb=entities_vdb,
                        src=src,
                        tgt=tgt,
                        chunk_ids=chunk_ids,
                        chunk_relationships=chunk_relationships,
                        extraction_units=extraction_units,
                        llm_response_cache=llm_response_cache,
                        global_config=global_config,
                        relation_chunks_storage=relation_chunks_storage,
                        entity_chunks_storage=entity_chunks_storage,
                    )
                    rebuilt_relationships_count += 1
                except Exception as e:
                    failed_relationships_count += 1
                    status_message = f"Failed to rebuild `{src}`~`{tgt}`: {e}"
                    logger.info(status_message)  # Per requirement, change to info

    # Create tasks for parallel processing
    tasks = []

    # Add entity rebuilding tasks
    for entity_name, chunk_ids in entities_to_rebuild.items():
        task = asyncio.create_task(_locked_rebuild_entity(entity_name, chunk_ids))
        tasks.append(task)

    # Add relationship rebuilding tasks
    for (src, tgt), chunk_ids in relationships_to_rebuild.items():
        task = asyncio.create_task(_locked_rebuild_relationship(src, tgt, chunk_ids))
        tasks.append(task)

    # Log parallel processing start
    status_message = f"Starting parallel rebuild of {len(entities_to_rebuild)} entities and {len(relationships_to_rebuild)} relationships (async: {graph_max_async})"
    logger.info(status_message)

    # Execute all tasks in parallel with semaphore control and early failure detection
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    # Check if any task raised an exception and ensure all exceptions are retrieved
    first_exception = None

    for task in done:
        try:
            exception = task.exception()
            if exception is not None:
                if first_exception is None:
                    first_exception = exception
            else:
                # Task completed successfully, retrieve result to mark as processed
                task.result()
        except Exception as e:
            if first_exception is None:
                first_exception = e

    # If any task failed, cancel all pending tasks and raise the first exception
    if first_exception is not None:
        # Cancel all pending tasks
        for pending_task in pending:
            pending_task.cancel()

        # Wait for cancellation to complete
        if pending:
            await asyncio.wait(pending)

        # Re-raise the first exception to notify the caller
        raise first_exception

    # Final status report
    status_message = f"KG rebuild completed: {rebuilt_entities_count} entities and {rebuilt_relationships_count} relationships rebuilt successfully."
    if failed_entities_count > 0 or failed_relationships_count > 0:
        status_message += f" Failed: {failed_entities_count} entities, {failed_relationships_count} relationships."

    logger.info(status_message)


async def _resolve_extraction_units(
    text_chunks_storage: BaseKVStorage, chunk_ids: set[str]
) -> dict[str, str]:
    """Map each existing chunk to the chunk its extraction ran on (its unit).

    Text is extracted on parent chunks and the graph is then redirected onto
    their children (``redirect_source_ids``), so a child's extraction — and the
    cache keys recording it — live on its parent (``sidecar.parent_id``). A chunk
    that carries its own cache keys (multimodal, or a parent) is its own unit.
    Chunks no longer in storage are left out: retrieval cannot return them.
    """
    ids = list(chunk_ids)
    units: dict[str, str] = {}
    for chunk_id, chunk_data in zip(ids, await text_chunks_storage.get_by_ids(ids)):
        if not chunk_data:
            continue
        parent_id = (chunk_data.get("sidecar") or {}).get("parent_id")
        if chunk_data.get("llm_cache_list") or not parent_id:
            units[chunk_id] = chunk_id
        else:
            units[chunk_id] = parent_id
    return units


def _units_of(chunk_ids: list[str], extraction_units: dict[str, str]) -> list[str]:
    """Distinct units behind ``chunk_ids``, in order.

    Several children share one parent; reading the parent once per child would
    count its descriptions — and a relation's weight — several times over.
    """
    return list(
        dict.fromkeys(extraction_units[c] for c in chunk_ids if c in extraction_units)
    )


def _keep_rebuildable(
    to_rebuild: dict,
    extraction_units: dict[str, str],
    cached_results: dict,
    *,
    kind: str,
) -> dict:
    """Drop vanished chunks, and every item that cannot be rebuilt faithfully.

    An item is rebuilt only when the extraction of *every* remaining source is
    cached. Rebuilding from a subset would silently drop the other documents'
    contribution, so such items are left untouched and reported instead: their
    documents predate the LLM cache and must be re-ingested.
    """
    kept: dict = {}
    for key, chunk_ids in to_rebuild.items():
        existing = [c for c in chunk_ids if c in extraction_units]
        missing = [u for u in _units_of(existing, extraction_units) if u not in cached_results]
        if not existing:
            logger.warning(
                f"Cannot rebuild {kind} `{key}`: none of its source chunks exist any "
                f"more; run scripts/repair_orphan_graph.py"
            )
        elif missing:
            logger.warning(
                f"Cannot rebuild {kind} `{key}`: no cached extraction for "
                f"{len(missing)} source(s) (e.g. {missing[0]}); re-ingest the "
                f"documents they belong to"
            )
        else:
            kept[key] = existing
    return kept


async def _get_cached_extraction_results(
    llm_response_cache: BaseKVStorage,
    chunk_ids: set[str],
    text_chunks_storage: BaseKVStorage,
    doc_status_storage: BaseKVStorage | None = None,
) -> dict[str, list[tuple[EntityExtractionOutput, int, str, str | None]]]:
    """Get cached extraction results for specific chunk IDs

    This function retrieves cached LLM extraction results for the given chunk IDs and returns
    them sorted by creation time. The results are sorted at two levels:
    1. Individual extraction results within each chunk are sorted by create_time (earliest first)
    2. Chunks themselves are sorted by the create_time of their earliest extraction result

    Args:
        llm_response_cache: LLM response cache storage
        chunk_ids: Set of chunk IDs to get cached results for
        text_chunks_storage: Text chunks storage for retrieving chunk data and LLM cache references

    Returns:
        Dict mapping chunk_id -> list of (EntityExtractionOutput, create_time,
        file_path, ontology), holding only entries that still validate against
        the current schema, where ``ontology`` is the ``kg_ontology`` of the
        chunk's document (its entries are ``OntologyExtractionOutput``), and:
        - Keys (chunk_ids) are ordered by the create_time of their first extraction result
        - Values (extraction results) are ordered by create_time within each chunk
    """
    cached_results = {}

    # Collect all LLM cache IDs from chunks
    all_cache_ids = set()

    # One read gives each chunk's cache references and the file_path its
    # extracted entities and relations are attributed to.
    file_paths: dict[str, str] = {}
    doc_ids: dict[str, str] = {}
    ids = list(chunk_ids)
    chunk_data_list = await text_chunks_storage.get_by_ids(ids)
    for chunk_id, chunk_data in zip(ids, chunk_data_list):
        if chunk_data and isinstance(chunk_data, dict):
            file_paths[chunk_id] = chunk_data.get("file_path", "unknown_source")
            if chunk_data.get("full_doc_id"):
                doc_ids[chunk_id] = chunk_data["full_doc_id"]
            llm_cache_list = chunk_data.get("llm_cache_list", [])
            if llm_cache_list:
                all_cache_ids.update(llm_cache_list)
        else:
            logger.warning(f"Chunk data is invalid or None: {chunk_data}")

    if not all_cache_ids:
        logger.warning(f"No LLM cache IDs found for {len(chunk_ids)} chunk IDs")
        return cached_results

    # The ontology a chunk was extracted under lives on its document's status
    # (doc_status metadata ``kg_ontology``).
    chunk_ontologies: dict[str, str] = {}
    if doc_status_storage is not None and doc_ids:
        unique_doc_ids = list(dict.fromkeys(doc_ids.values()))
        doc_rows = await doc_status_storage.get_by_ids(unique_doc_ids)
        doc_ontology = {
            doc_id: ((row or {}).get("metadata") or {}).get("kg_ontology")
            for doc_id, row in zip(unique_doc_ids, doc_rows)
        }
        chunk_ontologies = {
            chunk_id: doc_ontology[doc_id]
            for chunk_id, doc_id in doc_ids.items()
            if doc_ontology.get(doc_id)
        }

    # Batch get LLM cache entries
    cache_data_list = await llm_response_cache.get_by_ids(list(all_cache_ids))

    # Process cache entries and group by chunk_id
    valid_entries = 0
    for cache_entry in cache_data_list:
        if (
            cache_entry is not None
            and isinstance(cache_entry, dict)
            and cache_entry.get("cache_type") == "extract"
            and cache_entry.get("chunk_id") in chunk_ids
        ):
            chunk_id = cache_entry["chunk_id"]
            ontology = chunk_ontologies.get(chunk_id)
            output_model = OntologyExtractionOutput if ontology else EntityExtractionOutput
            # Stored as the chain's JSON; validated the same way cached_chain
            # does on read, so an entry from an older contract counts as absent.
            try:
                extraction_result = output_model.model_validate_json(
                    cache_entry["return"]
                )
                if ontology:
                    load_ontology_schema(ontology)  # its file must still exist
            except (ValidationError, FileNotFoundError, ValueError) as e:
                logger.warning(f"Skipping unusable extraction cache for {chunk_id}: {e}")
                continue
            create_time = cache_entry.get(
                "create_time", 0
            )  # Get creation time, default to 0
            valid_entries += 1

            # Support multiple LLM caches per chunk
            if chunk_id not in cached_results:
                cached_results[chunk_id] = []
            cached_results[chunk_id].append(
                (
                    extraction_result,
                    create_time,
                    file_paths.get(chunk_id, "unknown_source"),
                    ontology,
                )
            )

    # Sort extraction results by create_time for each chunk and collect earliest times
    chunk_earliest_times = {}
    for chunk_id in cached_results:
        # Sort by create_time (x[1])
        cached_results[chunk_id].sort(key=lambda x: x[1])
        # Store the earliest create_time for this chunk (first item after sorting)
        chunk_earliest_times[chunk_id] = cached_results[chunk_id][0][1]

    # Sort cached_results by the earliest create_time of each chunk
    sorted_chunk_ids = sorted(
        chunk_earliest_times.keys(), key=lambda chunk_id: chunk_earliest_times[chunk_id]
    )

    # Rebuild cached_results in sorted order
    sorted_cached_results = {}
    for chunk_id in sorted_chunk_ids:
        sorted_cached_results[chunk_id] = cached_results[chunk_id]

    logger.info(
        f"Found {valid_entries} valid cache entries, {len(sorted_cached_results)} chunks with results"
    )
    return sorted_cached_results


async def _rebuild_single_entity(
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    entity_name: str,
    chunk_ids: list[str],
    chunk_entities: dict,
    extraction_units: dict[str, str],
    llm_response_cache: BaseKVStorage,
    global_config: dict[str, str],
    entity_chunks_storage: BaseKVStorage | None = None,
) -> None:
    """Rebuild a single entity from cached extraction results.

    ``chunk_entities`` is keyed by extraction unit (see
    :func:`_resolve_extraction_units`); ``chunk_ids`` are the child chunks the
    graph points at and stay the entity's ``source_id``, as after ingestion.
    """

    # Get current entity data
    current_entity = await knowledge_graph_inst.get_node(entity_name)
    if not current_entity:
        return

    # Helper function to update entity in both graph and vector storage
    async def _update_entity_storage(
        final_description: str,
        entity_type: str,
        file_paths: list[str],
        source_chunk_ids: list[str],
        truncation_info: str = "",
    ):
        try:
            # Update entity in graph storage (critical path)
            updated_entity_data = {
                **current_entity,
                "description": final_description,
                "entity_type": entity_type,
                "source_id": GRAPH_FIELD_SEP.join(source_chunk_ids),
                "file_path": GRAPH_FIELD_SEP.join(file_paths)
                if file_paths
                else current_entity.get("file_path", "unknown_source"),
                "created_at": int(time.time()),
                "truncate": truncation_info,
            }
            # Aliases follow the remaining records: a deleted document's alias goes
            # with it. Rebuilt from relationships only (no records): keep them.
            aliases = (
                EntityResolver.merged_aliases(entity_name, None, all_entity_data)
                if all_entity_data
                else current_entity.get("aliases", "")
            )
            if aliases or current_entity.get("aliases"):
                updated_entity_data["aliases"] = aliases
            await knowledge_graph_inst.upsert_node(entity_name, updated_entity_data)

            # Update entity in vector database (equally critical)
            entity_vdb_id = compute_mdhash_id(entity_name, prefix="ent-")
            entity_content = _truncate_vdb_content(
                EntityResolver.vdb_content(entity_name, final_description, aliases),
                global_config,
                f"entity:{entity_name}",
            )

            vdb_data = {
                entity_vdb_id: {
                    "content": entity_content,
                    "entity_name": entity_name,
                    "source_id": updated_entity_data["source_id"],
                    "description": final_description,
                    "entity_type": entity_type,
                    "file_path": updated_entity_data["file_path"],
                }
            }

            # Use safe operation wrapper - VDB failure must throw exception
            await safe_vdb_operation_with_exception(
                operation=lambda: entities_vdb.upsert(vdb_data),
                operation_name="rebuild_entity_upsert",
                entity_name=entity_name,
                max_retries=3,
                retry_delay=0.1,
            )

        except Exception as e:
            error_msg = f"Failed to update entity storage for `{entity_name}`: {e}"
            logger.error(error_msg)
            raise  # Re-raise exception

    # normalized_chunk_ids = merge_source_ids([], chunk_ids)
    normalized_chunk_ids = chunk_ids

    if entity_chunks_storage is not None and normalized_chunk_ids:
        await entity_chunks_storage.upsert(
            {
                entity_name: {
                    "chunk_ids": normalized_chunk_ids,
                    "count": len(normalized_chunk_ids),
                }
            }
        )

    limit_method = (
        global_config.get("source_ids_limit_method") or SOURCE_IDS_LIMIT_METHOD_KEEP
    )

    limited_chunk_ids = apply_source_ids_limit(
        normalized_chunk_ids,
        global_config["max_source_ids_per_entity"],
        limit_method,
        identifier=f"`{entity_name}`",
    )

    # Collect all entity data from the units of the relevant (limited) chunks
    all_entity_data = []
    for unit_id in _units_of(limited_chunk_ids, extraction_units):
        all_entity_data.extend(chunk_entities.get(unit_id, {}).get(entity_name, []))

    if not all_entity_data:
        logger.warning(
            f"No entity data found for `{entity_name}`, trying to rebuild from relationships"
        )

        # Get all edges connected to this entity
        edges = await knowledge_graph_inst.get_node_edges(entity_name)
        if not edges:
            logger.warning(f"No relations attached to entity `{entity_name}`")
            return

        # Collect relationship data to extract entity information
        relationship_descriptions = []
        file_paths = set()

        # Get edge data for all connected relationships
        for src_id, tgt_id in edges:
            edge_data = await knowledge_graph_inst.get_edge(src_id, tgt_id)
            if edge_data:
                if edge_data.get("description"):
                    relationship_descriptions.append(edge_data["description"])

                if edge_data.get("file_path"):
                    edge_file_paths = edge_data["file_path"].split(GRAPH_FIELD_SEP)
                    file_paths.update(edge_file_paths)

        # deduplicate descriptions
        description_list = list(dict.fromkeys(relationship_descriptions))

        # Generate final description from relationships or fallback to current
        if description_list:
            final_description, _ = await _handle_entity_relation_summary(
                "Entity",
                entity_name,
                description_list,
                GRAPH_FIELD_SEP,
                global_config,
                llm_response_cache=llm_response_cache,
            )
        else:
            final_description = current_entity.get("description", "")

        entity_type = current_entity.get("entity_type", "UNKNOWN")
        await _update_entity_storage(
            final_description,
            entity_type,
            file_paths,
            limited_chunk_ids,
        )
        return

    # Process cached entity data
    descriptions = []
    entity_types = []
    file_paths_list = []
    seen_paths = set()

    for entity_data in all_entity_data:
        if entity_data.get("description"):
            descriptions.append(entity_data["description"])
        if entity_data.get("entity_type"):
            entity_types.append(entity_data["entity_type"])
        if entity_data.get("file_path"):
            file_path = entity_data["file_path"]
            if file_path and file_path not in seen_paths:
                file_paths_list.append(file_path)
                seen_paths.add(file_path)

    # Apply MAX_FILE_PATHS limit
    max_file_paths = global_config.get("max_file_paths", settings.max_file_paths)
    file_path_placeholder = DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    limit_method = global_config.get("source_ids_limit_method")

    original_count = len(file_paths_list)
    if original_count > max_file_paths:
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            file_paths_list = file_paths_list[-max_file_paths:]
        else:
            # KEEP: keep head (earliest), discard tail
            file_paths_list = file_paths_list[:max_file_paths]

        file_paths_list.append(
            f"...{file_path_placeholder}...({limit_method} {max_file_paths}/{original_count})"
        )
        logger.info(
            f"Limited `{entity_name}`: file_path {original_count} -> {max_file_paths} ({limit_method})"
        )

    # Remove duplicates while preserving order
    description_list = list(dict.fromkeys(descriptions))
    entity_types = list(dict.fromkeys(entity_types))

    # Get most common entity type
    entity_type = (
        max(set(entity_types), key=entity_types.count)
        if entity_types
        else current_entity.get("entity_type", "UNKNOWN")
    )

    # Generate final description from entities or fallback to current
    if description_list:
        final_description, _ = await _handle_entity_relation_summary(
            "Entity",
            entity_name,
            description_list,
            GRAPH_FIELD_SEP,
            global_config,
            llm_response_cache=llm_response_cache,
        )
    else:
        final_description = current_entity.get("description", "")

    if len(limited_chunk_ids) < len(normalized_chunk_ids):
        truncation_info = (
            f"{limit_method} {len(limited_chunk_ids)}/{len(normalized_chunk_ids)}"
        )
    else:
        truncation_info = ""

    await _update_entity_storage(
        final_description,
        entity_type,
        file_paths_list,
        limited_chunk_ids,
        truncation_info,
    )

    # Log rebuild completion with truncation info
    status_message = f"Rebuild `{entity_name}` from {len(chunk_ids)} chunks"
    if truncation_info:
        status_message += f" ({truncation_info})"
    logger.info(status_message)


async def _rebuild_single_relationship(
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    entities_vdb: BaseVectorStorage,
    src: str,
    tgt: str,
    chunk_ids: list[str],
    chunk_relationships: dict,
    extraction_units: dict[str, str],
    llm_response_cache: BaseKVStorage,
    global_config: dict[str, str],
    relation_chunks_storage: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
) -> None:
    """Rebuild a single relationship from cached extraction results

    ``chunk_relationships`` is keyed by extraction unit, like ``chunk_entities``
    in :func:`_rebuild_single_entity`.

    Note: This function assumes the caller has already acquired the appropriate
    keyed lock for the relationship pair to ensure thread safety.
    """

    # Get current relationship data
    current_relationship = await knowledge_graph_inst.get_edge(src, tgt)
    if not current_relationship:
        return

    # normalized_chunk_ids = merge_source_ids([], chunk_ids)
    normalized_chunk_ids = chunk_ids

    if relation_chunks_storage is not None and normalized_chunk_ids:
        storage_key = make_relation_chunk_key(src, tgt)
        await relation_chunks_storage.upsert(
            {
                storage_key: {
                    "chunk_ids": normalized_chunk_ids,
                    "count": len(normalized_chunk_ids),
                }
            }
        )

    limit_method = (
        global_config.get("source_ids_limit_method") or SOURCE_IDS_LIMIT_METHOD_KEEP
    )
    limited_chunk_ids = apply_source_ids_limit(
        normalized_chunk_ids,
        global_config["max_source_ids_per_relation"],
        limit_method,
        identifier=f"`{src}`~`{tgt}`",
    )

    # Collect all relationship data from the units of the relevant chunks
    all_relationship_data = []
    for unit_id in _units_of(limited_chunk_ids, extraction_units):
        unit_relationships = chunk_relationships.get(unit_id, {})
        # Check both (src, tgt) and (tgt, src) since relationships can be bidirectional
        for edge_key in [(src, tgt), (tgt, src)]:
            all_relationship_data.extend(unit_relationships.get(edge_key, []))

    if not all_relationship_data:
        logger.warning(f"No relation data found for `{src}-{tgt}`")
        return

    # Merge descriptions and keywords
    descriptions = []
    keywords = []
    weights = []
    file_paths_list = []
    seen_paths = set()

    for rel_data in all_relationship_data:
        if rel_data.get("description"):
            descriptions.append(rel_data["description"])
        if rel_data.get("keywords"):
            keywords.append(rel_data["keywords"])
        if rel_data.get("weight"):
            weights.append(rel_data["weight"])
        if rel_data.get("file_path"):
            file_path = rel_data["file_path"]
            if file_path and file_path not in seen_paths:
                file_paths_list.append(file_path)
                seen_paths.add(file_path)

    # Apply count limit
    max_file_paths = global_config.get("max_file_paths", settings.max_file_paths)
    file_path_placeholder = DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    limit_method = global_config.get("source_ids_limit_method")

    original_count = len(file_paths_list)
    if original_count > max_file_paths:
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            file_paths_list = file_paths_list[-max_file_paths:]
        else:
            # KEEP: keep head (earliest), discard tail
            file_paths_list = file_paths_list[:max_file_paths]

        file_paths_list.append(
            f"...{file_path_placeholder}...({limit_method} {max_file_paths}/{original_count})"
        )
        logger.info(
            f"Limited `{src}`~`{tgt}`: file_path {original_count} -> {max_file_paths} ({limit_method})"
        )

    # Remove duplicates while preserving order
    description_list = list(dict.fromkeys(descriptions))
    keywords = list(dict.fromkeys(keywords))

    combined_keywords = (
        ", ".join(set(keywords))
        if keywords
        else current_relationship.get("keywords", "")
    )

    weight = sum(weights) if weights else current_relationship.get("weight", 1.0)

    # Generate final description from relations or fallback to current
    if description_list:
        final_description, _ = await _handle_entity_relation_summary(
            "Relation",
            f"{src}-{tgt}",
            description_list,
            GRAPH_FIELD_SEP,
            global_config,
            llm_response_cache=llm_response_cache,
        )
    else:
        # fallback to keep current(unchanged)
        final_description = current_relationship.get("description", "")

    if len(limited_chunk_ids) < len(normalized_chunk_ids):
        truncation_info = (
            f"{limit_method} {len(limited_chunk_ids)}/{len(normalized_chunk_ids)}"
        )
    else:
        truncation_info = ""

    # Update relationship in graph storage
    updated_relationship_data = {
        **current_relationship,
        "description": final_description
        if final_description
        else current_relationship.get("description", ""),
        "keywords": combined_keywords,
        "weight": weight,
        # recomputed from the remaining records, not merged with the stored value:
        # a type only the deleted document asserted must go (a null value
        # removes the property in Neo4j's ``SET r += props``)
        "relation_type": None,
        "relation_triples": None,
        **_relation_typing(all_relationship_data, None),
        "source_id": GRAPH_FIELD_SEP.join(limited_chunk_ids),
        "file_path": GRAPH_FIELD_SEP.join([fp for fp in file_paths_list if fp])
        if file_paths_list
        else current_relationship.get("file_path", "unknown_source"),
        "truncate": truncation_info,
    }

    # Ensure both endpoint nodes exist before writing the edge back
    # (certain storage backends require pre-existing nodes).
    node_description = (
        updated_relationship_data["description"]
        if updated_relationship_data.get("description")
        else current_relationship.get("description", "")
    )
    node_source_id = updated_relationship_data.get("source_id", "")
    node_file_path = updated_relationship_data.get("file_path", "unknown_source")

    for node_id in {src, tgt}:
        if not (await knowledge_graph_inst.has_node(node_id)):
            node_created_at = int(time.time())
            node_data = {
                "entity_id": node_id,
                "source_id": node_source_id,
                "description": node_description,
                "entity_type": "UNKNOWN",
                "file_path": node_file_path,
                "created_at": node_created_at,
                "truncate": "",
            }
            await knowledge_graph_inst.upsert_node(node_id, node_data=node_data)

            # Update entity_chunks_storage for the newly created entity
            if entity_chunks_storage is not None and limited_chunk_ids:
                await entity_chunks_storage.upsert(
                    {
                        node_id: {
                            "chunk_ids": limited_chunk_ids,
                            "count": len(limited_chunk_ids),
                        }
                    }
                )

            # Update entity_vdb for the newly created entity
            if entities_vdb is not None:
                entity_vdb_id = compute_mdhash_id(node_id, prefix="ent-")
                entity_content = _truncate_vdb_content(
                    f"{node_id}\n{node_description}",
                    global_config,
                    f"entity:{node_id}",
                )
                vdb_data = {
                    entity_vdb_id: {
                        "content": entity_content,
                        "entity_name": node_id,
                        "source_id": node_source_id,
                        "entity_type": "UNKNOWN",
                        "file_path": node_file_path,
                    }
                }
                await safe_vdb_operation_with_exception(
                    operation=lambda payload=vdb_data: entities_vdb.upsert(payload),
                    operation_name="rebuild_added_entity_upsert",
                    entity_name=node_id,
                    max_retries=3,
                    retry_delay=0.1,
                )

    await knowledge_graph_inst.upsert_edge(src, tgt, updated_relationship_data)

    # Update relationship in vector database
    # Sort src and tgt to ensure consistent ordering (smaller string first)
    if src > tgt:
        src, tgt = tgt, src
    try:
        rel_vdb_id = compute_mdhash_id(src + tgt, prefix="rel-")
        rel_vdb_id_reverse = compute_mdhash_id(tgt + src, prefix="rel-")

        # Delete old vector records first (both directions to be safe)
        try:
            await relationships_vdb.delete([rel_vdb_id, rel_vdb_id_reverse])
        except Exception as e:
            logger.debug(
                f"Could not delete old relationship vector records {rel_vdb_id}, {rel_vdb_id_reverse}: {e}"
            )

        # Insert new vector record
        rel_content = f"{combined_keywords}\t{src}\n{tgt}\n{final_description}"
        vdb_data = {
            rel_vdb_id: {
                "src_id": src,
                "tgt_id": tgt,
                "source_id": updated_relationship_data["source_id"],
                "content": rel_content,
                "keywords": combined_keywords,
                "description": final_description,
                "weight": weight,
                "file_path": updated_relationship_data["file_path"],
            }
        }

        # Use safe operation wrapper - VDB failure must throw exception
        await safe_vdb_operation_with_exception(
            operation=lambda: relationships_vdb.upsert(vdb_data),
            operation_name="rebuild_relationship_upsert",
            entity_name=f"{src}-{tgt}",
            max_retries=3,
            retry_delay=0.2,
        )

    except Exception as e:
        error_msg = f"Failed to rebuild relationship storage for `{src}-{tgt}`: {e}"
        logger.error(error_msg)
        raise  # Re-raise exception

    # Log rebuild completion with truncation info
    status_message = f"Rebuild `{src}`~`{tgt}` from {len(chunk_ids)} chunks"
    if truncation_info:
        status_message += f" ({truncation_info})"
    # Add truncation info from apply_source_ids_limit if truncation occurred
    if len(limited_chunk_ids) < len(normalized_chunk_ids):
        truncation_info = (
            f" ({limit_method}:{len(limited_chunk_ids)}/{len(normalized_chunk_ids)})"
        )
        status_message += truncation_info

    logger.info(status_message)


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage | None,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
):
    """Get existing nodes from knowledge graph use name,if exists, merge data, else create, then upsert."""
    already_entity_types = []
    already_source_ids = []
    already_description = []
    already_file_paths = []

    # 1. Get existing node data from knowledge graph
    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node:
        existing_entity_type = already_node.get("entity_type")
        if (
            not isinstance(existing_entity_type, str)
            or not existing_entity_type.strip()
        ):
            existing_entity_type = "UNKNOWN"
        # Sanitize entity_type read back from DB to prevent dirty data from propagating
        if "," in existing_entity_type:
            original = existing_entity_type
            tokens = [t.strip() for t in existing_entity_type.split(",")]
            non_empty = [t for t in tokens if t]
            existing_entity_type = non_empty[0] if non_empty else "UNKNOWN"
            logger.warning(
                f"Entity type read from DB contains comma, taking first non-empty token: '{original}' -> '{existing_entity_type}'"
            )
        already_entity_types.append(existing_entity_type)

        existing_source_id = already_node.get("source_id") or ""
        already_source_ids.extend(existing_source_id.split(GRAPH_FIELD_SEP))

        existing_file_path = already_node.get("file_path") or "unknown_source"
        already_file_paths.extend(existing_file_path.split(GRAPH_FIELD_SEP))

        existing_desc = (already_node.get("description") or "").strip()
        if existing_desc:
            already_description.extend(existing_desc.split(GRAPH_FIELD_SEP))

    new_source_ids = []
    for dp in nodes_data:
        if dp.get("source_id"):
            new_source_ids.extend(dp["source_id"].split(GRAPH_FIELD_SEP))
    existing_full_source_ids = []
    if entity_chunks_storage is not None:
        stored_chunks = await entity_chunks_storage.get_by_id(entity_name)
        if stored_chunks and isinstance(stored_chunks, dict):
            existing_full_source_ids = [
                chunk_id
                for chunk_id in stored_chunks.get("chunk_ids", [])
                if chunk_id
            ]

    if not existing_full_source_ids:
        existing_full_source_ids = [
            chunk_id for chunk_id in already_source_ids if chunk_id
        ]

    # 2. Merging new source ids with existing ones
    full_source_ids = merge_source_ids(existing_full_source_ids, new_source_ids)

    if entity_chunks_storage is not None and full_source_ids:
        await entity_chunks_storage.upsert(
            {
                entity_name: {
                    "chunk_ids": full_source_ids,
                    "count": len(full_source_ids),
                }
            }
        )

    # 3. Finalize source_id by applying source ids limit
    limit_method = global_config.get("source_ids_limit_method")
    max_source_limit = global_config.get("max_source_ids_per_entity")
    source_ids = apply_source_ids_limit(
        full_source_ids,
        max_source_limit,
        limit_method,
        identifier=f"`{entity_name}`",
    )

    # 4. Only keep nodes not filter by apply_source_ids_limit if limit_method is KEEP
    if limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP:
        allowed_source_ids = set(source_ids)
        existing_full_source_ids_set = set(existing_full_source_ids)
        filtered_nodes = []
        for dp in nodes_data:
            raw_source_id = dp.get("source_id") or ""
            # source_id bisa berupa multi-chunk string (GRAPH_FIELD_SEP) akibat
            # parent-child redirect. Pecah agar perbandingan terhadap set individual benar.
            chunk_ids = [
                c for c in raw_source_id.split(GRAPH_FIELD_SEP) if c
            ] if raw_source_id else []
            if chunk_ids:
                # Pertahankan entry jika minimal satu child ada di allowed/existing
                if not (set(chunk_ids) & allowed_source_ids) and \
                not (set(chunk_ids) & existing_full_source_ids_set):
                    continue
            filtered_nodes.append(dp)
        nodes_data = filtered_nodes
    else:  # In FIFO mode, keep all nodes - truncation happens at source_ids level only
        nodes_data = list(nodes_data)

    # 5. Check if we need to skip summary due to source_ids limit
    if (
        limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP
        and len(existing_full_source_ids) >= max_source_limit
        and not nodes_data
    ):
        if already_node:
            logger.info(
                f"Skipped `{entity_name}`: KEEP old chunks {already_source_ids}/{len(full_source_ids)}"
            )
            existing_node_data = dict(already_node)
            return existing_node_data
        else:
            logger.error(
                f"Internal Error: already_node missing for `{entity_name}`"
            )
            raise ValueError(
                f"Internal Error: already_node missing for `{entity_name}`"
            )

    # 6.1 Finalize source_id
    source_id = GRAPH_FIELD_SEP.join(source_ids)

    # 6.2 Finalize entity type by highest count
    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]

    # 7. Deduplicate nodes by description, keeping first occurrence in the same document
    unique_nodes = {}
    for i, dp in enumerate(nodes_data, start=1):
        desc = dp.get("description")
        if not desc:
            continue
        if desc not in unique_nodes:
            unique_nodes[desc] = dp
        await _cooperative_yield(i, every=32)

    sorted_descriptions = [dp["description"] for dp in unique_nodes.values()]

    description_list = list(dict.fromkeys(already_description + sorted_descriptions))
    if not description_list:
        fallback_description = f"Entity {entity_name}"
        logger.warning(
            f"Entity `{entity_name}` has no description; fallback to `{fallback_description}`"
        )
        description_list = [fallback_description]

    # Check for cancellation before LLM summary
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException(
                    "User cancelled during entity summary"
                )

    # 8. Get summary description an LLM usage status
    description, llm_was_used = await _handle_entity_relation_summary(
        "Entity",
        entity_name,
        description_list,
        GRAPH_FIELD_SEP,
        global_config,
        llm_response_cache,
    )

    # 9. Build file_path within MAX_FILE_PATHS
    file_paths_list = []
    seen_paths = set()
    has_placeholder = False  # Indicating file_path has been truncated before

    max_file_paths = global_config.get("max_file_paths", settings.max_file_paths)
    file_path_placeholder = DEFAULT_FILE_PATH_MORE_PLACEHOLDER

    # Collect from already_file_paths, excluding placeholder
    for fp in already_file_paths:
        if fp and fp.startswith(f"...{file_path_placeholder}"):  # Skip placeholders
            has_placeholder = True
            continue
        if fp and fp not in seen_paths:
            file_paths_list.append(fp)
            seen_paths.add(fp)

    # Collect from new data
    for i, dp in enumerate(nodes_data, start=1):
        file_path_item = dp.get("file_path")
        if file_path_item and file_path_item not in seen_paths:
            file_paths_list.append(file_path_item)
            seen_paths.add(file_path_item)
        await _cooperative_yield(i, every=32)

    # Apply count limit
    if len(file_paths_list) > max_file_paths:
        limit_method = global_config.get(
            "source_ids_limit_method", SOURCE_IDS_LIMIT_METHOD_KEEP
        )
        # Add + sign to indicate actual file count is higher
        original_count_str = (
            f"{len(file_paths_list)}+"
            if has_placeholder
            else str(len(file_paths_list))
        )

        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            file_paths_list = file_paths_list[-max_file_paths:]
            file_paths_list.append(f"...{file_path_placeholder}...(FIFO)")
        else:
            # KEEP: keep head (earliest), discard tail
            file_paths_list = file_paths_list[:max_file_paths]
            file_paths_list.append(f"...{file_path_placeholder}...(KEEP Old)")

        logger.info(
            f"Limited `{entity_name}`: file_path {original_count_str} -> {max_file_paths} ({limit_method})"
        )
    # Finalize file_path
    file_path = GRAPH_FIELD_SEP.join(file_paths_list)

    # 10.Log based on actual LLM usage
    num_fragment = len(description_list)
    already_fragment = len(already_description)
    if llm_was_used:
        status_message = f"LLMmrg: `{entity_name}` | {already_fragment}+{num_fragment - already_fragment}"
    else:
        status_message = f"Merged: `{entity_name}` | {already_fragment}+{num_fragment - already_fragment}"

    truncation_info = truncation_info_log = ""
    if len(source_ids) < len(full_source_ids):
        # Add truncation info from apply_source_ids_limit if truncation occurred
        truncation_info_log = (
            f"{limit_method} {len(source_ids)}/{len(full_source_ids)}"
        )
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            truncation_info = truncation_info_log
        else:
            truncation_info = "KEEP Old"

    deduplicated_num = already_fragment + len(nodes_data) - num_fragment
    dd_message = ""
    if deduplicated_num > 0:
        # Duplicated description detected across multiple trucks for the same entity
        dd_message = f"dd {deduplicated_num}"

    if dd_message or truncation_info_log:
        status_message += (
            f" ({', '.join(filter(None, [truncation_info_log, dd_message]))})"
        )

    # Add message to pipeline satus when merge happens
    if already_fragment > 0 or llm_was_used:
        logger.info(status_message)
    else:
        logger.debug(status_message)

    # 11. Update both graph and vector db
    node_data = dict(
        entity_id=entity_name,
        entity_type=entity_type,
        description=description,
        source_id=source_id,
        file_path=file_path,
        created_at=int(time.time()),
        truncate=truncation_info,
        **_ontology_properties(nodes_data, already_node),
    )
    # Names entity resolution merged into this node (ICCN for Indonesia Creative Cities Network)
    aliases = EntityResolver.merged_aliases(
        entity_name, (already_node or {}).get("aliases"), nodes_data
    )
    if aliases:
        node_data["aliases"] = aliases
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    if entity_vdb is not None:
        entity_vdb_id = compute_mdhash_id(str(entity_name), prefix="ent-")
        entity_content = _truncate_vdb_content(
            EntityResolver.vdb_content(entity_name, description, aliases),
            global_config,
            f"entity:{entity_name}",
        )
        data_for_vdb = {
            entity_vdb_id: {
                "entity_name": entity_name,
                "entity_type": entity_type,
                "content": entity_content,
                "source_id": source_id,
                "file_path": file_path,
            }
        }
        await safe_vdb_operation_with_exception(
            operation=lambda payload=data_for_vdb: entity_vdb.upsert(payload),
            operation_name="entity_upsert",
            entity_name=entity_name,
            max_retries=3,
            retry_delay=0.1,
        )
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage | None,
    entity_vdb: BaseVectorStorage | None,
    global_config: dict,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    added_entities: list = None,  # New parameter to track entities added during edge processing
    relation_chunks_storage: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
):
    if src_id == tgt_id:
        return None
    relation_key = f"{src_id}->{tgt_id}"

    already_edge = None
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []
    already_file_paths = []

    # 1. Get existing edge data from graph storage
    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        # Handle the case where get_edge returns None or missing fields
        if already_edge:
            # Get weight with default 1.0 if missing
            already_weights.append(already_edge.get("weight", 1.0))

            # Get source_id with empty string default if missing or None
            if already_edge.get("source_id") is not None:
                already_source_ids.extend(
                    already_edge["source_id"].split(GRAPH_FIELD_SEP)
                )

            # Get file_path with empty string default if missing or None
            if already_edge.get("file_path") is not None:
                already_file_paths.extend(
                    already_edge["file_path"].split(GRAPH_FIELD_SEP)
                )

            # Get description with empty string default if missing or None
            if already_edge.get("description") is not None:
                already_description.extend(
                    already_edge["description"].split(GRAPH_FIELD_SEP)
                )

            # Get keywords with empty string default if missing or None
            if already_edge.get("keywords") is not None:
                already_keywords.extend(
                    split_string_by_multi_markers(
                        already_edge["keywords"], [GRAPH_FIELD_SEP]
                    )
                )

    new_source_ids = []
    for dp in edges_data:
        if dp.get("source_id"):
            new_source_ids.extend(dp["source_id"].split(GRAPH_FIELD_SEP))

    storage_key = make_relation_chunk_key(src_id, tgt_id)
    existing_full_source_ids = []
    if relation_chunks_storage is not None:
        stored_chunks = await relation_chunks_storage.get_by_id(storage_key)
        if stored_chunks and isinstance(stored_chunks, dict):
            existing_full_source_ids = [
                chunk_id
                for chunk_id in stored_chunks.get("chunk_ids", [])
                if chunk_id
            ]

    if not existing_full_source_ids:
        existing_full_source_ids = [
            chunk_id for chunk_id in already_source_ids if chunk_id
        ]

    # 2. Merge new source ids with existing ones
    full_source_ids = merge_source_ids(existing_full_source_ids, new_source_ids)

    if relation_chunks_storage is not None and full_source_ids:
        await relation_chunks_storage.upsert(
            {
                storage_key: {
                    "chunk_ids": full_source_ids,
                    "count": len(full_source_ids),
                }
            }
        )

    # 3. Finalize source_id by applying source ids limit
    limit_method = global_config.get("source_ids_limit_method")
    max_source_limit = global_config.get("max_source_ids_per_relation")
    source_ids = apply_source_ids_limit(
        full_source_ids,
        max_source_limit,
        limit_method,
        identifier=f"`{src_id}`~`{tgt_id}`",
    )
    limit_method = (
        global_config.get("source_ids_limit_method") or SOURCE_IDS_LIMIT_METHOD_KEEP
    )

    # 4. Only keep edges with source_id in the final source_ids list if in KEEP mode
    if limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP:
        allowed_source_ids = set(source_ids)
        existing_full_source_ids_set = set(existing_full_source_ids)
        filtered_edges = []
        for dp in edges_data:
            raw_source_id = dp.get("source_id") or ""
            # source_id bisa berupa multi-chunk string (GRAPH_FIELD_SEP) akibat
            # parent-child redirect. Pecah agar perbandingan terhadap set individual benar.
            chunk_ids = [
                c for c in raw_source_id.split(GRAPH_FIELD_SEP) if c
            ] if raw_source_id else []

            if chunk_ids:
                # Pertahankan entry jika minimal satu child ada di allowed/existing
                # Skip relationship fragments sourced from chunks dropped by keep oldest cap
                if not (set(chunk_ids) & allowed_source_ids) and \
                   not (set(chunk_ids) & existing_full_source_ids_set):
                    continue
            filtered_edges.append(dp)
        edges_data = filtered_edges
    else:  # In FIFO mode, keep all edges - truncation happens at source_ids level only
        edges_data = list(edges_data)

    # 5. Check if we need to skip summary due to source_ids limit
    if (
        limit_method == SOURCE_IDS_LIMIT_METHOD_KEEP
        and len(existing_full_source_ids) >= max_source_limit
        and not edges_data
    ):
        if already_edge:
            logger.info(
                f"Skipped `{src_id}`~`{tgt_id}`: KEEP old chunks  {already_source_ids}/{len(full_source_ids)}"
            )
            existing_edge_data = dict(already_edge)
            return existing_edge_data
        else:
            logger.error(
                f"Internal Error: already_node missing for `{src_id}`~`{tgt_id}`"
            )
            raise ValueError(
                f"Internal Error: already_node missing for `{src_id}`~`{tgt_id}`"
            )

    # 6.1 Finalize source_id
    source_id = GRAPH_FIELD_SEP.join(source_ids)

    # 6.2 Finalize weight by summing new edges and existing weights
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)

    # 6.2 Finalize keywords by merging existing and new keywords
    all_keywords = set()
    # Process already_keywords (which are comma-separated)
    for i, keyword_str in enumerate(already_keywords, start=1):
        if keyword_str:  # Skip empty strings
            all_keywords.update(
                k.strip() for k in keyword_str.split(",") if k.strip()
            )
        await _cooperative_yield(i, every=32)
    # Process new keywords from edges_data
    for i, edge in enumerate(edges_data, start=1):
        if edge.get("keywords"):
            all_keywords.update(
                k.strip() for k in edge["keywords"].split(",") if k.strip()
            )
        await _cooperative_yield(i, every=32)
    # Join all unique keywords with commas
    keywords = ",".join(sorted(all_keywords))

    # 7. Deduplicate by description, keeping first occurrence in the same document
    unique_edges = {}
    for i, dp in enumerate(edges_data, start=1):
        description_value = dp.get("description")
        if not description_value:
            continue
        if description_value not in unique_edges:
            unique_edges[description_value] = dp
        await _cooperative_yield(i, every=32)

    # Document order, not the extraction timestamp — see _merge_nodes_then_upsert.
    sorted_descriptions = [dp["description"] for dp in unique_edges.values()]

    # Combine already_description with the new descriptions, skipping
    # duplicates — see _merge_nodes_then_upsert.
    description_list = list(dict.fromkeys(already_description + sorted_descriptions))
    if not description_list:
        logger.error(f"Relation {src_id}~{tgt_id} has no description")
        raise ValueError(f"Relation {src_id}~{tgt_id} has no description")

    # Check for cancellation before LLM summary
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException(
                    "User cancelled during relation summary"
                )

    # 8. Get summary description an LLM usage status
    description, llm_was_used = await _handle_entity_relation_summary(
        "Relation",
        f"({src_id}, {tgt_id})",
        description_list,
        GRAPH_FIELD_SEP,
        global_config,
        llm_response_cache,
    )

    # 9. Build file_path within MAX_FILE_PATHS limit
    file_paths_list = []
    seen_paths = set()
    has_placeholder = False  # Track if already_file_paths contains placeholder

    max_file_paths = global_config.get("max_file_paths", settings.max_file_paths)
    file_path_placeholder = DEFAULT_FILE_PATH_MORE_PLACEHOLDER

    # Collect from already_file_paths, excluding placeholder
    for fp in already_file_paths:
        # Check if this is a placeholder record
        if fp and fp.startswith(f"...{file_path_placeholder}"):  # Skip placeholders
            has_placeholder = True
            continue
        if fp and fp not in seen_paths:
            file_paths_list.append(fp)
            seen_paths.add(fp)

    # Collect from new data
    for i, dp in enumerate(edges_data, start=1):
        file_path_item = dp.get("file_path")
        if file_path_item and file_path_item not in seen_paths:
            file_paths_list.append(file_path_item)
            seen_paths.add(file_path_item)
        await _cooperative_yield(i, every=32)
    # Apply count limit
    if len(file_paths_list) > max_file_paths:
        limit_method = global_config.get(
            "source_ids_limit_method", SOURCE_IDS_LIMIT_METHOD_KEEP
        )

        # Add + sign to indicate actual file count is higher
        original_count_str = (
            f"{len(file_paths_list)}+"
            if has_placeholder
            else str(len(file_paths_list))
        )

        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            # FIFO: keep tail (newest), discard head
            file_paths_list = file_paths_list[-max_file_paths:]
            file_paths_list.append(f"...{file_path_placeholder}...(FIFO)")
        else:
            # KEEP: keep head (earliest), discard tail
            file_paths_list = file_paths_list[:max_file_paths]
            file_paths_list.append(f"...{file_path_placeholder}...(KEEP Old)")

        logger.info(
            f"Limited `{src_id}`~`{tgt_id}`: file_path {original_count_str} -> {max_file_paths} ({limit_method})"
        )
    # Finalize file_path
    file_path = GRAPH_FIELD_SEP.join(file_paths_list)

    # 10. Log based on actual LLM usage
    num_fragment = len(description_list)
    already_fragment = len(already_description)
    if llm_was_used:
        status_message = f"LLMmrg: `{src_id}`~`{tgt_id}` | {already_fragment}+{num_fragment - already_fragment}"
    else:
        status_message = f"Merged: `{src_id}`~`{tgt_id}` | {already_fragment}+{num_fragment - already_fragment}"

    truncation_info = truncation_info_log = ""
    if len(source_ids) < len(full_source_ids):
        # Add truncation info from apply_source_ids_limit if truncation occurred
        truncation_info_log = (
            f"{limit_method} {len(source_ids)}/{len(full_source_ids)}"
        )
        if limit_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
            truncation_info = truncation_info_log
        else:
            truncation_info = "KEEP Old"

    deduplicated_num = already_fragment + len(edges_data) - num_fragment
    dd_message = ""
    if deduplicated_num > 0:
        # Duplicated description detected across multiple trucks for the same entity
        dd_message = f"dd {deduplicated_num}"

    if dd_message or truncation_info_log:
        status_message += (
            f" ({', '.join(filter(None, [truncation_info_log, dd_message]))})"
        )

    # Add message to pipeline satus when merge happens
    if already_fragment > 0 or llm_was_used:
        logger.info(status_message)
    else:
        logger.debug(status_message)

    # 11. Update both graph and vector db
    for need_insert_id in [src_id, tgt_id]:
        # Optimization: Use get_node instead of has_node + get_node
        existing_node = await knowledge_graph_inst.get_node(need_insert_id)

        if existing_node is None:
            # Node doesn't exist - create new node
            node_created_at = int(time.time())
            node_data = {
                "entity_id": need_insert_id,
                "source_id": source_id,
                "description": description,
                "entity_type": "UNKNOWN",
                "file_path": file_path,
                "created_at": node_created_at,
                "truncate": "",
            }
            await knowledge_graph_inst.upsert_node(
                need_insert_id, node_data=node_data
            )

            # Update entity_chunks_storage for the newly created entity
            if entity_chunks_storage is not None:
                chunk_ids = [chunk_id for chunk_id in full_source_ids if chunk_id]
                if chunk_ids:
                    await entity_chunks_storage.upsert(
                        {
                            need_insert_id: {
                                "chunk_ids": chunk_ids,
                                "count": len(chunk_ids),
                            }
                        }
                    )

            if entity_vdb is not None:
                entity_vdb_id = compute_mdhash_id(need_insert_id, prefix="ent-")
                entity_content = _truncate_vdb_content(
                    f"{need_insert_id}\n{description}",
                    global_config,
                    f"entity:{need_insert_id}",
                )
                vdb_data = {
                    entity_vdb_id: {
                        "content": entity_content,
                        "entity_name": need_insert_id,
                        "source_id": source_id,
                        "entity_type": "UNKNOWN",
                        "file_path": file_path,
                    }
                }
                await safe_vdb_operation_with_exception(
                    operation=lambda payload=vdb_data: entity_vdb.upsert(payload),
                    operation_name="added_entity_upsert",
                    entity_name=f"{need_insert_id} [relation:{relation_key}]",
                    max_retries=3,
                    retry_delay=0.1,
                    timeout_seconds=_get_relationship_vdb_timeout_seconds(
                        global_config
                    ),
                    log_start=False,
                    success_log_threshold_seconds=5.0,
                )

            # Track entities added during edge processing
            if added_entities is not None:
                entity_data = {
                    "entity_name": need_insert_id,
                    "entity_type": "UNKNOWN",
                    "description": description,
                    "source_id": source_id,
                    "file_path": file_path,
                    "created_at": node_created_at,
                }
                added_entities.append(entity_data)
        else:
            # Node exists - update its source_ids by merging with new source_ids
            updated = False  # Track if any update occurred

            # 1. Get existing full source_ids from entity_chunks_storage
            existing_full_source_ids = []
            if entity_chunks_storage is not None:
                stored_chunks = await entity_chunks_storage.get_by_id(
                    need_insert_id
                )
                if stored_chunks and isinstance(stored_chunks, dict):
                    existing_full_source_ids = [
                        chunk_id
                        for chunk_id in stored_chunks.get("chunk_ids", [])
                        if chunk_id
                    ]

            # If not in entity_chunks_storage, get from graph database
            if not existing_full_source_ids:
                if existing_node.get("source_id"):
                    existing_full_source_ids = existing_node["source_id"].split(
                        GRAPH_FIELD_SEP
                    )

            # 2. Merge with new source_ids from this relationship
            new_source_ids_from_relation = [
                chunk_id for chunk_id in source_ids if chunk_id
            ]
            merged_full_source_ids = merge_source_ids(
                existing_full_source_ids, new_source_ids_from_relation
            )

            # 3. Save merged full list to entity_chunks_storage (conditional)
            if (
                entity_chunks_storage is not None
                and merged_full_source_ids != existing_full_source_ids
            ):
                updated = True
                await entity_chunks_storage.upsert(
                    {
                        need_insert_id: {
                            "chunk_ids": merged_full_source_ids,
                            "count": len(merged_full_source_ids),
                        }
                    }
                )

            # 4. Apply source_ids limit for graph and vector db
            limit_method = global_config.get(
                "source_ids_limit_method", SOURCE_IDS_LIMIT_METHOD_KEEP
            )
            max_source_limit = global_config.get("max_source_ids_per_entity")
            limited_source_ids = apply_source_ids_limit(
                merged_full_source_ids,
                max_source_limit,
                limit_method,
                identifier=f"`{need_insert_id}`",
            )

            # 5. Update graph database and vector database with limited source_ids (conditional)
            limited_source_id_str = GRAPH_FIELD_SEP.join(limited_source_ids)

            if limited_source_id_str != existing_node.get("source_id", ""):
                updated = True
                updated_node_data = {
                    **existing_node,
                    "source_id": limited_source_id_str,
                }
                await knowledge_graph_inst.upsert_node(
                    need_insert_id, node_data=updated_node_data
                )

                # Update vector database
                if entity_vdb is not None:
                    entity_vdb_id = compute_mdhash_id(need_insert_id, prefix="ent-")
                    entity_content = EntityResolver.vdb_content(
                        need_insert_id,
                        existing_node.get("description", ""),
                        existing_node.get("aliases"),
                    )
                    vdb_data = {
                        entity_vdb_id: {
                            "content": entity_content,
                            "entity_name": need_insert_id,
                            "source_id": limited_source_id_str,
                            "entity_type": existing_node.get(
                                "entity_type", "UNKNOWN"
                            ),
                            "file_path": existing_node.get(
                                "file_path", "unknown_source"
                            ),
                        }
                    }
                await safe_vdb_operation_with_exception(
                    operation=lambda payload=vdb_data: entity_vdb.upsert(payload),
                    operation_name="existing_entity_update",
                    entity_name=f"{need_insert_id} [relation:{relation_key}]",
                    max_retries=3,
                    retry_delay=0.1,
                    timeout_seconds=_get_relationship_vdb_timeout_seconds(
                        global_config
                    ),
                    log_start=False,
                    success_log_threshold_seconds=5.0,
                )

            # 6. Log once at the end if any update occurred
            if updated:
                status_message = (
                    f"Chunks appended from relation: `{need_insert_id}`"
                )
                logger.info(status_message)

    edge_created_at = int(time.time())
    edge_upsert_started = time.perf_counter()
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
            file_path=file_path,
            created_at=edge_created_at,
            truncate=truncation_info,
            **_relation_typing(edges_data, already_edge),
            **_ontology_properties(edges_data, already_edge),
        ),
    )
    edge_upsert_elapsed = time.perf_counter() - edge_upsert_started
    if edge_upsert_elapsed >= 5.0:
        logger.info(
            "Graph edge upsert slow for `%s` in %.2fs",
            relation_key,
            edge_upsert_elapsed,
        )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
        source_id=source_id,
        file_path=file_path,
        created_at=edge_created_at,
        truncate=truncation_info,
        weight=weight,
    )

    # Sort src_id and tgt_id to ensure consistent ordering (smaller string first)
    if src_id > tgt_id:
        src_id, tgt_id = tgt_id, src_id

    if relationships_vdb is not None:
        rel_vdb_id = compute_mdhash_id(src_id + tgt_id, prefix="rel-")
        rel_vdb_id_reverse = compute_mdhash_id(tgt_id + src_id, prefix="rel-")
        try:
            await relationships_vdb.delete([rel_vdb_id, rel_vdb_id_reverse])
        except Exception as e:
            logger.debug(
                f"Could not delete old relationship vector records {rel_vdb_id}, {rel_vdb_id_reverse}: {e}"
            )
        rel_content = _truncate_vdb_content(
            f"{keywords}\t{src_id}\n{tgt_id}\n{description}",
            global_config,
            f"relationship:{src_id}-{tgt_id}",
        )
        vdb_data = {
            rel_vdb_id: {
                "src_id": src_id,
                "tgt_id": tgt_id,
                "source_id": source_id,
                "content": rel_content,
                "keywords": keywords,
                "description": description,
                "weight": weight,
                "file_path": file_path,
            }
        }
        relation_status_message = f"Upserting relation VDB: `{relation_key}`"
        logger.info(relation_status_message)
        await safe_vdb_operation_with_exception(
            operation=lambda payload=vdb_data: relationships_vdb.upsert(payload),
            operation_name="relationship_upsert",
            entity_name=relation_key,
            max_retries=3,
            retry_delay=0.2,
            timeout_seconds=_get_relationship_vdb_timeout_seconds(global_config),
            log_start=False,
            success_log_threshold_seconds=5.0,
        )

    return edge_data


@asynccontextmanager
async def _merge_storages(
    batched: bool,
    namespace: str,
    node_names,
    edge_pairs,
    knowledge_graph_inst: BaseGraphStorage,
    entity_chunks_storage: BaseKVStorage | None,
    relation_chunks_storage: BaseKVStorage | None,
):
    """Yield ``(graph, entity_chunks, relation_chunks, item_lock)`` for one merge.

    Batched: hold the graph keyed lock for every entity the document touches (relation
    endpoints included) for the whole merge, read them all once, let the per-item
    merge run against in-memory overlays, and write everything once on exit. Holding
    the lock throughout is what keeps the snapshot current: every graph writer takes
    this namespace's lock. Documents sharing an entity therefore merge one after
    another instead of interleaving per entity — same content, only the order differs.

    On an error the completed part is still written, as the per-item path would have.
    """
    if not batched:
        yield (
            knowledge_graph_inst,
            entity_chunks_storage,
            relation_chunks_storage,
            lambda keys: get_storage_keyed_lock(
                keys, namespace=namespace
            ),
        )
        return

    node_names = list(node_names)
    edge_pairs = list(edge_pairs)
    # _KeyedLockContext sorts but is not reentrant: a duplicate key would deadlock.
    lock_keys = list(dict.fromkeys([*node_names, *(n for pair in edge_pairs for n in pair)]))
    async with get_storage_keyed_lock(
        lock_keys, namespace=namespace
    ):
        session = MergeSession(
            knowledge_graph_inst, entity_chunks_storage, relation_chunks_storage
        )
        await session.prefetch(node_names, edge_pairs)
        try:
            yield (
                session.graph,
                session.entity_chunks if entity_chunks_storage is not None else None,
                session.relation_chunks if relation_chunks_storage is not None else None,
                session.lock,
            )
        except Exception:
            try:
                await session.flush()
            except Exception as flush_error:
                logger.error(f"Merge flush after failure also failed: {flush_error}")
            raise
        await session.flush()


async def merge_nodes_and_edges(
    chunk_results: list,
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict[str, str],
    full_entities_storage: BaseKVStorage = None,
    full_relations_storage: BaseKVStorage = None,
    doc_id: str = None,
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    entity_chunks_storage: BaseKVStorage | None = None,
    relation_chunks_storage: BaseKVStorage | None = None,
    current_file_number: int = 0,
    total_files: int = 0,
    file_path: str = "unknown_source",
    _batched: bool = True,
    entity_resolver: EntityResolver | None = None,
) -> None:
    """Two-phase merge: process all entities first, then all relationships

    This approach ensures data consistency by:
    1. Phase 1: Process all entities concurrently
    2. Phase 2: Process all relationships concurrently (may add missing entities)
    3. Phase 3: Update full_entities and full_relations storage with final results

    Args:
        chunk_results: List of tuples (maybe_nodes, maybe_edges) containing extracted entities and relationships
        knowledge_graph_inst: Knowledge graph storage
        entity_vdb: Entity vector database
        relationships_vdb: Relationship vector database
        global_config: Global configuration
        full_entities_storage: Storage for document entity lists
        full_relations_storage: Storage for document relation lists
        doc_id: Document ID for storage indexing
        pipeline_status: Pipeline status dictionary
        pipeline_status_lock: Lock for pipeline status
        llm_response_cache: LLM response cache
        entity_chunks_storage: Storage tracking full chunk lists per entity
        relation_chunks_storage: Storage tracking full chunk lists per relation
        current_file_number: Current file number for logging
        total_files: Total files for logging
        file_path: File path for logging
        _batched: Read the graph once, merge in memory, write once (see
            ``merge_session``). ``False`` runs the per-item path; kept for the
            equivalence test only.
        entity_resolver: Rewrites name variants to their canonical name before
            anything is read or written (see ``entity_resolution``); the
            per-document YAML registry comes from ``global_config["kg_entity_registry"]``
            and the LLM stage (Fase 2) from ``global_config["entity_resolution_llm"]``.
            None keeps the exact-name merge.
    """

    # Check for cancellation at the start of merge
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException("User cancelled during merge phase")

    # Collect all nodes and edges from all chunks
    all_nodes = defaultdict(list)
    all_edges = defaultdict(list)

    for i, (maybe_nodes, maybe_edges) in enumerate(chunk_results, start=1):
        # Collect nodes
        for entity_name, entities in maybe_nodes.items():
            all_nodes[entity_name].extend(entities)

        # Collect edges with sorted keys for undirected graph
        for edge_key, edges in maybe_edges.items():
            sorted_edge_key = tuple(sorted(edge_key))
            all_edges[sorted_edge_key].extend(edges)
        await _cooperative_yield(i, every=32)

    # Before _merge_storages, so its locks and prefetch already use canonical names.
    if entity_resolver is not None:
        all_nodes, all_edges = await entity_resolver.resolve(
            all_nodes,
            all_edges,
            global_config.get("kg_entity_registry"),
            use_llm=global_config.get("entity_resolution_llm", False),
        )

    total_entities_count = len(all_nodes)
    total_relations_count = len(all_edges)

    log_message = f"Merging stage {current_file_number}/{total_files}: {file_path}"
    logger.info(log_message)

    # Get max async tasks limit from global_config for semaphore control
    graph_max_async = global_config.get("llm_model_max_async", 4) * 2
    semaphore = asyncio.Semaphore(graph_max_async)

    workspace = global_config.get("workspace", "")
    graph_namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"

    async with _merge_storages(
        _batched,
        graph_namespace,
        all_nodes.keys(),
        all_edges.keys(),
        knowledge_graph_inst,
        entity_chunks_storage,
        relation_chunks_storage,
    ) as (graph, entity_chunks, relation_chunks, item_lock):
        # ===== Phase 1: Process all entities concurrently =====
        log_message = f"Phase 1: Processing {total_entities_count} entities from {doc_id} (async: {graph_max_async})"
        logger.info(log_message)

        async def _locked_process_entity_name(entity_name, entities):
            async with semaphore:
                # Check for cancellation before processing entity
                if pipeline_status is not None and pipeline_status_lock is not None:
                    async with pipeline_status_lock:
                        if pipeline_status.get("cancellation_requested", False):
                            raise PipelineCancelledException(
                                "User cancelled during entity merge"
                            )

                async with item_lock([entity_name]):
                    try:
                        logger.debug(f"Processing entity {entity_name}")
                        entity_data = await _merge_nodes_then_upsert(
                            entity_name,
                            entities,
                            graph,
                            entity_vdb,
                            global_config,
                            pipeline_status,
                            pipeline_status_lock,
                            llm_response_cache,
                            entity_chunks,
                        )

                        return entity_data

                    except Exception as e:
                        error_msg = f"Error processing entity `{entity_name}`: {e}"
                        logger.error(error_msg)


                        # Re-raise the original exception with a prefix
                        prefixed_exception = create_prefixed_exception(
                            e, f"`{entity_name}`"
                        )
                        raise prefixed_exception from e

        # Create entity processing tasks
        entity_tasks = []
        for i, (entity_name, entities) in enumerate(all_nodes.items(), start=1):
            task = asyncio.create_task(_locked_process_entity_name(entity_name, entities))
            entity_tasks.append(task)
            await _cooperative_yield(i, every=16)

        # Execute entity tasks with error handling
        processed_entities = []
        if entity_tasks:
            done, pending = await asyncio.wait(
                entity_tasks, return_when=asyncio.FIRST_EXCEPTION
            )

            first_exception = None
            processed_entities = []

            for i, task in enumerate(done, start=1):
                try:
                    result = task.result()
                except BaseException as e:
                    if first_exception is None:
                        first_exception = e
                else:
                    processed_entities.append(result)
                await _cooperative_yield(i, every=32)

            if pending:
                for task in pending:
                    task.cancel()
                pending_results = await asyncio.gather(*pending, return_exceptions=True)
                for result in pending_results:
                    if isinstance(result, BaseException):
                        if first_exception is None:
                            first_exception = result
                    else:
                        processed_entities.append(result)

            if first_exception is not None:
                raise first_exception

            await asyncio.sleep(0)

        # ===== Phase 2: Process all relationships concurrently =====
        log_message = f"Phase 2: Processing {total_relations_count} relations from {doc_id} (async: {graph_max_async})"
        logger.info(log_message)

        async def _locked_process_edges(edge_key, edges):
            async with semaphore:
                # Check for cancellation before processing edges
                if pipeline_status is not None and pipeline_status_lock is not None:
                    async with pipeline_status_lock:
                        if pipeline_status.get("cancellation_requested", False):
                            raise PipelineCancelledException(
                                "User cancelled during relation merge"
                            )

                sorted_edge_key = sorted([edge_key[0], edge_key[1]])
                edge_label = _format_relation_edge_label(edge_key)

                async with item_lock(sorted_edge_key):
                    try:
                        added_entities = []  # Track entities added during edge processing

                        edge_data = await _merge_edges_then_upsert(
                            edge_key[0],
                            edge_key[1],
                            edges,
                            graph,
                            relationships_vdb,
                            entity_vdb,
                            global_config,
                            pipeline_status,
                            pipeline_status_lock,
                            llm_response_cache,
                            added_entities,  # Pass list to collect added entities
                            relation_chunks,
                            entity_chunks,
                        )

                        if edge_data is None:
                            return None, []

                        return edge_data, added_entities

                    except Exception as e:
                        error_msg = f"Error processing relation `{edge_label}`: {e}"
                        logger.error(error_msg)


                        # Re-raise the original exception with a prefix
                        prefixed_exception = create_prefixed_exception(e, f"{edge_label}")
                        raise prefixed_exception from e

        # Create relationship processing tasks
        edge_tasks = []
        edge_task_labels: dict[asyncio.Task, str] = {}
        for i, (edge_key, edges) in enumerate(all_edges.items(), start=1):
            task = asyncio.create_task(_locked_process_edges(edge_key, edges))
            edge_tasks.append(task)
            edge_task_labels[task] = _format_relation_edge_label(edge_key)
            await _cooperative_yield(i, every=32)

        # Execute relationship tasks with error handling
        processed_edges = []
        all_added_entities = []

        if edge_tasks:
            done, pending = await asyncio.wait(
                edge_tasks, return_when=asyncio.FIRST_EXCEPTION
            )

            first_exception = None

            for i, task in enumerate(done, start=1):
                try:
                    edge_data, added_entities = task.result()
                except BaseException as e:
                    if first_exception is None:
                        first_exception = e
                else:
                    if edge_data is not None:
                        processed_edges.append(edge_data)
                    all_added_entities.extend(added_entities)
                await _cooperative_yield(i, every=32)

            if pending:
                pending_labels = [
                    edge_task_labels.get(task, "<unknown>") for task in pending
                ]
                preview = ", ".join(pending_labels[:10])
                if len(pending_labels) > 10:
                    preview += f", ... (+{len(pending_labels) - 10} more)"
                logger.warning(
                    "Phase 2 pending relation tasks for %s: %s",
                    doc_id,
                    preview or "<none>",
                )
                for task in pending:
                    task.cancel()
                pending_results = await asyncio.gather(*pending, return_exceptions=True)
                for result in pending_results:
                    if isinstance(result, BaseException):
                        if first_exception is None:
                            first_exception = result
                    else:
                        edge_data, added_entities = result
                        if edge_data is not None:
                            processed_edges.append(edge_data)
                        all_added_entities.extend(added_entities)

                logger.info(
                    "Phase 2 pending relation tasks drained for %s: collected_edges=%d collected_added_entities=%d",
                    doc_id,
                    len(processed_edges),
                    len(all_added_entities),
                )

            if first_exception is not None:
                raise first_exception

            logger.info(
                "Phase 2 relation processing completed for %s: edges=%d added_entities=%d",
                doc_id,
                len(processed_edges),
                len(all_added_entities),
            )
            await asyncio.sleep(0)

    # ===== Phase 3: Update full_entities and full_relations storage =====
    if full_entities_storage and full_relations_storage and doc_id:
        try:
            # Merge all entities: original entities + entities added during edge processing
            final_entity_names = set()

            # Add original processed entities
            for i, entity_data in enumerate(processed_entities, start=1):
                if entity_data and entity_data.get("entity_name"):
                    final_entity_names.add(entity_data["entity_name"])
                await _cooperative_yield(i, every=32)

            # Add entities that were added during relationship processing
            for i, added_entity in enumerate(all_added_entities, start=1):
                if added_entity and added_entity.get("entity_name"):
                    final_entity_names.add(added_entity["entity_name"])
                await _cooperative_yield(i, every=32)

            # Collect all relation pairs
            final_relation_pairs = set()
            for i, edge_data in enumerate(processed_edges, start=1):
                if edge_data:
                    src_id = edge_data.get("src_id")
                    tgt_id = edge_data.get("tgt_id")
                    if src_id and tgt_id:
                        relation_pair = tuple(sorted([src_id, tgt_id]))
                        final_relation_pairs.add(relation_pair)
                await _cooperative_yield(i, every=32)

            log_message = f"Phase 3: Updating final {len(final_entity_names)}({len(processed_entities)}+{len(all_added_entities)}) entities and  {len(final_relation_pairs)} relations from {doc_id}"
            logger.info(log_message)

            # Update storage — MERGE with any existing entry for this doc_id
            # instead of overwriting.  The same doc_id is written by more than
            # one pass: this merge (text and multimodal results together) and
            # the Stage 3.5 modal-main entities write in ``_store_multimodal_
            # entities_to_full_entities``.  Overwriting drops entities /
            # relations tracked by earlier passes, and ``adelete_by_doc_id``
            # reads only these two indices to decide what to remove — so any
            # entity dropped here becomes a graph zombie after deletion.
            # Union-with-existing keeps the tracking index authoritative
            # without touching extraction, graph merge, or chunk-tracking.
            if final_entity_names:
                existing_doc_entities = await full_entities_storage.get_by_id(doc_id)
                if isinstance(existing_doc_entities, dict):
                    existing_names = existing_doc_entities.get("entity_names") or []
                    merged_names = list(
                        dict.fromkeys([*existing_names, *final_entity_names])
                    )
                    entities_payload = {
                        **existing_doc_entities,
                        "entity_names": merged_names,
                        "count": len(merged_names),
                    }
                else:
                    entities_payload = {
                        "entity_names": list(final_entity_names),
                        "count": len(final_entity_names),
                    }
                await full_entities_storage.upsert({doc_id: entities_payload})

            if final_relation_pairs:
                new_pairs_list = [list(pair) for pair in final_relation_pairs]
                existing_doc_relations = await full_relations_storage.get_by_id(doc_id)
                if isinstance(existing_doc_relations, dict):
                    existing_pairs = existing_doc_relations.get("relation_pairs") or []
                    seen_pairs: set[tuple[str, str]] = set()
                    merged_pairs: list[list[str]] = []
                    for pair in (*existing_pairs, *new_pairs_list):
                        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                            continue
                        src, tgt = pair[0], pair[1]
                        if src is None or tgt is None:
                            continue
                        key = tuple(sorted((str(src), str(tgt))))
                        if key in seen_pairs:
                            continue
                        seen_pairs.add(key)
                        merged_pairs.append([src, tgt])
                    relations_payload = {
                        **existing_doc_relations,
                        "relation_pairs": merged_pairs,
                        "count": len(merged_pairs),
                    }
                else:
                    relations_payload = {
                        "relation_pairs": new_pairs_list,
                        "count": len(new_pairs_list),
                    }
                await full_relations_storage.upsert({doc_id: relations_payload})

            logger.debug(
                f"Updated entity-relation index for document {doc_id}: {len(final_entity_names)} entities (original: {len(processed_entities)}, added: {len(all_added_entities)}), {len(final_relation_pairs)} relations"
            )

        except Exception as e:
            logger.error(
                f"Failed to update entity-relation index for document {doc_id}: {e}"
            )
            # Don't raise exception to avoid affecting main flow

    log_message = f"Completed merging: {len(processed_entities)} entities, {len(all_added_entities)} extra entities, {len(processed_edges)} relations"
    logger.info(log_message)


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    global_config: dict[str, str],
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
    text_chunks_storage: BaseKVStorage | None = None,
    semaphore: asyncio.Semaphore | None = None,
) -> list:
    """Extract entities and relationships from chunks via the extraction chain.

    ``llm_response_cache`` is handed to the chain through ``configurable`` (see
    ``core/llm_cache.py``) rather than consulted here, and every key it touches
    is written back onto the chunk in ``text_chunks_storage``.

    ``semaphore`` lets concurrent calls for one document share a single
    ``llm_model_max_async`` cap (text and multimodal extraction run side by
    side); omitted, each call gets its own.
    """
    # Check for cancellation at the start of entity extraction
    if pipeline_status is not None and pipeline_status_lock is not None:
        async with pipeline_status_lock:
            if pipeline_status.get("cancellation_requested", False):
                raise PipelineCancelledException(
                    "User cancelled during entity extraction"
                )

    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    max_extract_input_tokens = settings.max_extract_input_tokens
    extract_tokenizer: Tokenizer | None = global_config.get("tokenizer")

    ordered_chunks = list(chunks.items())
    # One document's chunks all share an ontology, so it is resolved once per
    # call rather than per chunk. ``None`` selects the default contract.
    chains = get_kg_chains(global_config.get("kg_ontology"))
    processed_chunks = 0
    total_chunks = len(ordered_chunks)

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """Process a single chunk
        Args:
            chunk_key_dp (tuple[str, TextChunkSchema]):
                ("chunk-xxxxxx", {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int})
        Returns:
            tuple: (maybe_nodes, maybe_edges) containing extracted entities and relationships
        """
        nonlocal processed_chunks
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = strip_internal_multimodal_markup_for_extraction(chunk_dp["content"])
        # Get file path from chunk data or use default
        file_path = chunk_dp.get("file_path", "unknown_source")
        heading_path = _truncate_section_context(
            format_heading_context(chunk_dp),
            extract_tokenizer,
            DEFAULT_MAX_SECTION_CONTEXT_TOKENS,
        )
        heading_context_block = (
            chains.section_context.format(heading_path=heading_path)
            if heading_path
            else ""
        )

        chain_inputs = {
            "input_text": sanitize_text_for_encoding(content),
            "heading_context_block": heading_context_block,
        }
        entity_extraction_user_prompt = chains.extraction_prompt.format_messages(
            **chain_inputs
        )[-1].content

        # Cache keys land here so they can be recorded on the chunk below;
        # without that the delete path cannot reclaim the entries.
        cache_keys_collector: list[str] = []
        cache_config = {
            "configurable": {
                CACHE_KV: llm_response_cache,
                CACHE_CHUNK_ID: chunk_key,
                CACHE_KEYS: cache_keys_collector,
            }
        }

        first_pass = await _extract_or_none(
            chains.extraction_chain, chain_inputs, cache_config, chunk_key, "extraction"
        )
        timestamp = int(time.time())

        # Replay the first pass to the gleaning chain as the model's own turn.
        # Re-serialized from the validated model rather than kept as the raw
        # body, so the history is exactly what we accepted.
        history = pack_user_ass_to_openai_messages(
            entity_extraction_user_prompt,
            first_pass.model_dump_json() if first_pass else "",
        )

        maybe_nodes, maybe_edges = _parse_extraction_output(
            first_pass, chunk_key, timestamp, file_path
        )

        # Process additional gleaning results only 1 time when entity_extract_max_gleaning is greater than zero.
        run_gleaning = entity_extract_max_gleaning > 0
        if (
            run_gleaning
            and extract_tokenizer is not None
            and max_extract_input_tokens > 0
        ):
            # Gleaning replays the initial extraction's user/assistant pair
            # via ``history`` and appends a "continue" instruction.
            # When the initial response was large (many entities/edges) or
            # the chunk content is itself near the budget, that combined
            # payload can blow past MAX_EXTRACT_INPUT_TOKENS and yield a
            # provider ``context_length_exceeded`` error.  Pre-check here
            # and skip rather than fail.
            gleaning_inputs = {**chain_inputs, "messages": history}
            gleaning_token_count = sum(
                len(extract_tokenizer.encode(msg.text))
                for msg in chains.gleaning_prompt.format_messages(**gleaning_inputs)
            )
            if gleaning_token_count > max_extract_input_tokens:
                logger.warning(
                    f"Gleaning stopped for chunk {chunk_key}: "
                    f"Input tokens ({gleaning_token_count}) exceeded limit "
                    f"({max_extract_input_tokens})."
                )
                progress.activity(
                    "Pemeriksaan ulang (gleaning) satu chunk dilewati: melebihi batas token", level="warning",
                )
                run_gleaning = False

        if run_gleaning:
            glean_pass = await _extract_or_none(
                chains.gleaning_chain,
                {**chain_inputs, "messages": history},
                cache_config,
                chunk_key,
                "gleaning",
            )
            timestamp = int(time.time())

            glean_nodes, glean_edges = _parse_extraction_output(
                glean_pass, chunk_key, timestamp, file_path
            )

            # Merge results - compare description lengths to choose better version
            for i, (entity_name, glean_entities) in enumerate(
                glean_nodes.items(), start=1
            ):
                if entity_name in maybe_nodes:
                    # Compare description lengths and keep the better one
                    original_desc_len = len(
                        maybe_nodes[entity_name][0].get("description", "") or ""
                    )
                    glean_desc_len = len(glean_entities[0].get("description", "") or "")

                    if glean_desc_len > original_desc_len:
                        maybe_nodes[entity_name] = list(glean_entities)
                    # Otherwise keep original version
                else:
                    # New entity from gleaning stage
                    maybe_nodes[entity_name] = list(glean_entities)
                await _cooperative_yield(i, every=8)

            for i, (edge_key, glean_edge_list) in enumerate(
                glean_edges.items(), start=1
            ):
                if edge_key in maybe_edges:
                    # Compare description lengths and keep the better one
                    original_desc_len = len(
                        maybe_edges[edge_key][0].get("description", "") or ""
                    )
                    glean_desc_len = len(
                        glean_edge_list[0].get("description", "") or ""
                    )

                    if glean_desc_len > original_desc_len:
                        maybe_edges[edge_key] = list(glean_edge_list)
                    # Otherwise keep original version
                else:
                    # New edge from gleaning stage
                    maybe_edges[edge_key] = list(glean_edge_list)
                await _cooperative_yield(i, every=8)

        # Entities still unlinked get one relationship-only call. Fires only for
        # chunks that need it, so it costs far less than another gleaning pass.
        if global_config.get("orphan_repair", True):
            orphans = unlinked_names(maybe_nodes, maybe_edges)
            if orphans:
                repair_pass = await _extract_or_none(
                    chains.repair_chain,
                    repair_inputs(chain_inputs, maybe_nodes, orphans),
                    cache_config,
                    chunk_key,
                    "orphan repair",
                )
                _, repair_edges = _parse_extraction_output(
                    repair_pass, chunk_key, int(time.time()), file_path
                )
                new_edges = accept_repair(maybe_nodes, maybe_edges, set(orphans), repair_edges)
                maybe_edges.update(new_edges)
                logger.debug(
                    f"{chunk_key}: orphan repair linked {len(new_edges)} edges "
                    f"for {len(orphans)} unlinked entities"
                )

        # Under an ontology, validate the chunk's combined first pass + gleaning the
        # way neo4j-graphrag's GraphPruning validates its extracted graph. The
        # multimodal entity injected below is not an ontology type and is exempt,
        # like neo4j-graphrag's lexical graph.
        if chains.schema is not None:
            maybe_nodes, maybe_edges = await prune_chunk(
                maybe_nodes, maybe_edges, chains.schema, chunk_key
            )

        # Inject multimodal entity + associations for drawing/table/equation
        # chunks. Placed inside the chunk's
        # concurrency slot (rather than the centralized post-pass that used
        # to live in doc_status_fields.augment_chunk_results_with_mm_entities)
        # so each multimodal chunk benefits from the chunk-level concurrency
        # already enforced by extract_entities.
        sidecar_block = chunk_dp.get("sidecar")
        if isinstance(sidecar_block, dict):
            sidecar_type = sidecar_block.get("type")
            sidecar_id = sidecar_block.get("id")
            if (
                sidecar_type in {"drawing", "table", "equation"}
                and isinstance(sidecar_id, str)
                and sidecar_id
            ):
                mm_entity_name = sidecar_id
                now_ts = int(time.time())
                mm_nodes_list = maybe_nodes.setdefault(mm_entity_name, [])
                mm_nodes_list.append(
                    {
                        "entity_name": mm_entity_name,
                        "entity_type": sidecar_type,
                        # description == the full multimodal chunk content so
                        # the extracted entity carries the same grounding
                        # surface the prompt produced; analyze_multimodal's
                        # description/name field is already inlined there.
                        "description": chunk_dp.get("content", "") or "",
                        "source_id": chunk_key,
                        "file_path": file_path,
                        "timestamp": now_ts,
                    }
                )
                heading_block = chunk_dp.get("heading")
                heading_label = ""
                if isinstance(heading_block, dict):
                    heading_label = str(heading_block.get("heading") or "").strip()
                # Omit the "in section ..." clause entirely when the chunk has no
                # real heading — a literal "unknown" filler would otherwise leak
                # into the relation description, embedding, and retrieval as noise.
                location = (
                    f"in section {heading_label} of document"
                    if heading_label
                    else "of document"
                )
                mm_display_name = _parse_mm_display_name(
                    chunk_dp.get("content", "") or "", sidecar_id
                )
                for tgt in list(maybe_nodes.keys()):
                    if tgt == mm_entity_name:
                        continue
                    edge_key = (mm_entity_name, tgt)
                    edge_list = maybe_edges.setdefault(edge_key, [])
                    edge_list.append(
                        {
                            "src_id": mm_entity_name,
                            "tgt_id": tgt,
                            "weight": 1.0,
                            "description": (
                                f"{tgt} is associated with {sidecar_type} "
                                f"{mm_display_name} {location} "
                                f'"{file_path}"'
                            ),
                            "keywords": "associated with, contained in",
                            "source_id": chunk_key,
                            "file_path": file_path,
                            "timestamp": now_ts,
                        }
                    )

        # Record every cache key the two chain calls touched, so deleting the
        # document also reclaims its cached responses.
        if cache_keys_collector and text_chunks_storage:
            await update_chunk_cache_list(
                chunk_key,
                text_chunks_storage,
                cache_keys_collector,
                "entity_extraction",
            )

        processed_chunks += 1
        entities_count = len(maybe_nodes)
        relations_count = len(maybe_edges)
        log_message = f"Chunk {processed_chunks} of {total_chunks} extracted {entities_count} Ent + {relations_count} Rel {chunk_key}"
        logger.info(log_message)

        # Return the extracted nodes and edges for centralized processing
        return maybe_nodes, maybe_edges

    # Get max async tasks limit from global_config
    chunk_max_async = global_config.get("llm_model_max_async", 4)
    semaphore = semaphore or asyncio.Semaphore(chunk_max_async)

    async def _process_with_semaphore(chunk):
        async with semaphore:
            # Check for cancellation before processing chunk
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    if pipeline_status.get("cancellation_requested", False):
                        raise PipelineCancelledException(
                            "User cancelled during chunk processing"
                        )

            try:
                result = await _process_single_content(chunk)
                # Yield once between chunk completions so API coroutines can resume
                # even when many chunk tasks are hitting cache and finishing quickly.
                await asyncio.sleep(0)
                return result
            except Exception as e:
                chunk_id = chunk[0]  # Extract chunk_id from chunk[0]
                prefixed_exception = create_prefixed_exception(e, chunk_id)
                raise prefixed_exception from e

    tasks = []
    for c in ordered_chunks:
        task = asyncio.create_task(_process_with_semaphore(c))
        tasks.append(task)

    # Wait for tasks to complete or for the first exception to occur
    # This allows us to cancel remaining tasks if any task fails
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    # Check if any task raised an exception and ensure all exceptions are retrieved
    first_exception = None
    chunk_results = []

    # Chunk order, not ``done`` (a set): merge concatenates an entity's
    # descriptions in this order, so a random order changes the summary prompt
    # and the summary itself on every run of the same document.
    for task in tasks:
        if task not in done:
            continue
        try:
            exception = task.exception()
            if exception is not None:
                if first_exception is None:
                    first_exception = exception
            else:
                chunk_results.append(task.result())
        except Exception as e:
            if first_exception is None:
                first_exception = e

    # If any task failed, cancel all pending tasks and raise the first exception
    if first_exception is not None:
        # Cancel all pending tasks
        for pending_task in pending:
            pending_task.cancel()

        # Wait for cancellation to complete
        if pending:
            await asyncio.wait(pending)

        # Add progress prefix to the exception message
        progress_prefix = f"C[{processed_chunks + 1}/{total_chunks}]"

        # Re-raise the original exception with a prefix
        prefixed_exception = create_prefixed_exception(first_exception, progress_prefix)
        raise prefixed_exception from first_exception

    # If all tasks completed successfully, chunk_results already contains the results
    # Return the chunk_results for later processing in merge_nodes_and_edges
    return chunk_results
