
"""Multimodal analysis — the image / table / equation half of ingestion.

Each image / table / equation is described by a model and turned into exactly
**one** chunk: no splitting, ever. The chunk body is an enrichment — the
original caption / table body / LaTeX wrapped around the generated description
(see ``_apply_chunk_template``) — plus a modal entity and its ``belongs_to``
relations.

Mixed into :class:`IngestionRAG` and driven by ``process_single_document``: it
returns its extraction results instead of merging them, so the document's
entities and relations reach the graph in a single merge pass together with
the text side.

The processors themselves live next door in this package and are not touched
by the pipeline — this module only orchestrates them.
"""


import asyncio
import contextlib
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from hybrid_rag.core.utils import compute_mdhash_id, raise_first_failure
from hybrid_rag.storage.shared import get_storage_keyed_lock
from hybrid_rag.config import settings
import logging

from hybrid_rag.ingestion.multimodal.content import (
    format_table_body,
    get_equation_text_and_format,
    get_processor_for_type,
    get_table_body,
    normalize_caption_list,
)

logger = logging.getLogger(__name__)


class MultimodalAnalysisError(RuntimeError):
    """An image / table / equation could not be described by the model."""

    def __init__(self, item_info: Dict[str, Any], cause: object):
        super().__init__(
            f"{item_info['type']} item {item_info['index']} "
            f"(page {item_info['page_idx'] + 1}) could not be analyzed: {cause}"
        )


class ImageUnavailableError(RuntimeError):
    """An image item has no readable source — neither a local file nor a URL.

    Raised rather than skipped: a document that silently loses its figures
    looks fully ingested while its most information-dense blocks are missing.
    """


def resolve_image_sources(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return ``items`` with every image's ``img_path`` readable by the VLM.

    ``multimodal_items`` is persisted and captioned later, so a local
    ``img_path`` written by an earlier run points into a temp directory
    ``parse_file`` has long deleted. Those are swapped for the durable S3 URL,
    which the VLM fetches itself — no download here. Items are copied, never
    mutated in place. Tables and equations never read image bytes, so they are
    left alone.

    Raises:
        ImageUnavailableError: an image has neither a local file nor a URL.
            Fatal by design — the caller marks the document FAILED rather than
            indexing it without its figures.
    """
    def is_url(value: Any) -> bool:
        return str(value or "").strip().startswith(("http://", "https://"))

    resolved = []
    for index, item in enumerate(items):
        item = dict(item)
        resolved.append(item)
        img_path = str(item.get("img_path") or "").strip()
        if item.get("type") != "image" or is_url(img_path):
            continue
        if img_path and Path(img_path).is_file():
            continue
        if not is_url(item.get("_original_url")):
            raise ImageUnavailableError(
                f"Image item {index} (page {item.get('page_idx')}) has no local file "
                f"and no downloadable URL; img_path={item.get('img_path')!r}. "
                f"Re-parsing the source document is the only way to recover it."
            )
        item["img_path"] = str(item["_original_url"]).strip()
    return resolved


class _AnalyzeMixin:
    """Caption the multimodal half of a document and extract its graph."""

    async def _analyze_resolved_items(
        self,
        multimodal_items: List[Dict[str, Any]],
        file_path: str,
        doc_id: str,
        extract_semaphore: asyncio.Semaphore | None = None,
    ) -> Tuple[Dict[str, Any], List[Tuple]]:
        """Caption every item and extract its graph, without merging.

        Every image must already have a readable source: the caller runs
        :func:`resolve_image_sources` first, so ``ImageUnavailableError`` fails
        the document before any model call.

        ``extract_semaphore`` is the caller's per-document extraction cap, shared
        with the text side so running both at once never raises the load on the
        extraction model above what a single call would.

        Returns:
            ``(chunks, chunk_results)`` — the stored multimodal chunks keyed by
            id, and the extraction results ready to be merged. The caller folds
            these into the text side's results so the whole document reaches the
            graph in a single merge pass.

        Raises:
            MultimodalAnalysisError: an item could not be described after the
                client's retries. Fatal by design, like ``ImageUnavailableError``.
        """

        kg_domain: str | None = None
        try:
            existing_doc_status = await self.doc_status.get_by_id(doc_id)
            # Figures and tables belong to the same document as its text, so they
            # must be extracted under the same contract.
            kg_domain = ((existing_doc_status or {}).get("metadata") or {}).get("kg_domain")
        except Exception:
            pass

        # Items of this document analyzed at once. ``vlm_service`` has no limiter
        # of its own, so this is the only brake on the VLM endpoint: concurrent
        # calls = VLM_MAX_ASYNC × MAX_PARALLEL_INGESTION. Raise it only as far as
        # the endpoint can take — a call that still fails after the client's
        # retries fails the document.
        semaphore = asyncio.Semaphore(max(1, settings.vlm_max_async))

        total_items = len(multimodal_items)
        completed_count = 0
        self.logger.info(f"Starting to process {total_items} multimodal content items")

        # Stage 1: Concurrent generation of descriptions using correct processors for each type
        async def describe(item: Dict[str, Any], index: int) -> Dict[str, Any]:
            nonlocal completed_count
            content_type = item.get("type", "unknown")
            item_info = {
                "page_idx": item.get("page_idx", 0),
                "index": item.get("_content_list_index", index),
                "type": content_type,
            }
            processor = get_processor_for_type(self.modal_processors, content_type)
            if not processor:
                raise MultimodalAnalysisError(item_info, "no processor for this type")
            # Keys the analysis chain read or wrote; carried onto the chunk
            # below so deleting the document reclaims them.
            cache_keys: List[str] = []
            async with semaphore:
                try:
                    description, entity_info = await processor.generate_description_only(
                        modal_content=item,
                        content_type=content_type,
                        item_info=item_info,
                        entity_name=None,  # Let LLM auto-generate
                        cache_keys=cache_keys,
                    )
                except Exception as e:
                    raise MultimodalAnalysisError(item_info, e) from e

            completed_count += 1
            if completed_count % max(1, total_items // 10) == 0 or completed_count == total_items:
                self.logger.info(
                    f"Multimodal chunk generation progress: {completed_count}/{total_items} "
                    f"({completed_count / total_items * 100:.1f}%)"
                )
            return {
                "index": index,
                "content_type": content_type,
                "description": description,
                "cache_keys": cache_keys,
                "entity_info": entity_info,
                "original_item": item,
                "item_info": item_info,
                # Selalu -1: nomor >= 0 milik chunk teks. Sitasi auditor
                # ``file › chunk N`` dibuka lewat nomor ini, jadi chunk
                # multimodal bernomor 0, 1, … menutupi chunk teks bernomor
                # sama. Multimodal disitasi lewat hash id-nya
                # (retrievers._chunk_token).
                "chunk_order_index": -1,
                "processor": processor,  # Keep reference to the processor used
                "file_path": file_path,
            }

        # Every item or none: a document that silently loses a figure or table
        # looks fully ingested while its most information-dense block is
        # missing. The first failure cancels the rest and fails the document.
        try:
            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(describe(item, i))
                    for i, item in enumerate(multimodal_items)
                ]
        except* MultimodalAnalysisError as failures:
            raise_first_failure(failures)
        multimodal_data_list = [task.result() for task in tasks]

        # Stage 2: Convert to chunk records
        modal_chunks = self._to_modal_chunks(
            multimodal_data_list, file_path, doc_id
        )

        # Stage 3: Store chunks. Before extraction starts:
        # extraction records its LLM cache keys onto these rows.
        await self._store_modal_chunks(modal_chunks)

        # Stage 4 (extraction) runs beside stage 3.5: it reads only the chunk
        # text, never the graph / entities_vdb / full_entities that 3.5 writes,
        # and both are done before the merge reads the graph. 3.5 waits on the
        # serial embedding queue for its entities_vdb flush, so extraction no
        # longer waits behind it.
        extraction = asyncio.create_task(
            self._extract_modal_entities(
                modal_chunks, domain=kg_domain, semaphore=extract_semaphore
            )
        )
        try:
            # Stage 3.5: Store multimodal main entities to entities_vdb and
            # full_entities. Never cancelled for an extraction failure: stopped
            # midway it would leave graph nodes that full_entities (and so the
            # document's deletion) does not know about.
            await self._store_multimodal_main_entities(
                multimodal_data_list, modal_chunks, file_path, doc_id
            )
        except BaseException:
            extraction.cancel()
            with contextlib.suppress(BaseException):
                await extraction
            raise
        chunk_results = await extraction

        # Stage 5: Add belongs_to relations (multimodal-specific)
        enhanced_chunk_results = await self._batch_add_belongs_to_relations_type_aware(
            chunk_results, multimodal_data_list
        )

        # No merge here: the caller merges these together with the text side's
        # results, so an entity that appears in both a paragraph and a figure is
        # summarised once instead of twice. Registering the chunk ids in
        # doc_status is likewise the caller's job — it writes chunks_list once,
        # and deletion no longer depends on that list anyway (it resolves chunks
        # through full_doc_id).
        return modal_chunks, enhanced_chunk_results

    def _to_modal_chunks(
        self, multimodal_data_list: List[Dict[str, Any]], file_path: str, doc_id: str
    ) -> Dict[str, Any]:
        """Convert multimodal descriptions into chunk records (the text pipeline's chunk shape)."""

        chunks = {}

        for data in multimodal_data_list:
            description = data["description"]
            entity_info = data["entity_info"]
            chunk_order_index = data["chunk_order_index"]
            content_type = data["content_type"]
            original_item = data["original_item"]

            # Apply the appropriate chunk template based on content type
            formatted_chunk_content = self._apply_chunk_template(
                content_type, original_item, description
            )

            # Generate chunk_id
            chunk_id = compute_mdhash_id(formatted_chunk_content, prefix="chunk-")

            # Calculate tokens
            tokens = len(self.tokenizer.encode(formatted_chunk_content))

            # # DEBUG: Print the exact chunk that will be embedded and entity-extracted
            # self.logger.info("\n" + "="*70)
            # self.logger.info(f"📦 DEBUG: CHUNK TO BE EMBEDDED & ENTITY-EXTRACTED")
            # self.logger.info(f"   type       : {content_type}")
            # self.logger.info(f"   chunk_id   : {chunk_id}")
            # self.logger.info(f"   tokens     : {tokens}")
            # self.logger.info(f"   entity_name: {entity_info['entity_name']}")
            # self.logger.info(f"{'─'*70}")
            # self.logger.info(f"[CHUNK CONTENT]:\n{formatted_chunk_content}")
            # self.logger.info("="*70 + "\n")

            # Use full path or basename based on config
            file_ref = os.path.basename(file_path)

            # Same shape as a text chunk
            chunks[chunk_id] = {
                "content": formatted_chunk_content,  # Now uses the templated content
                "tokens": tokens,
                "full_doc_id": doc_id,
                "chunk_order_index": chunk_order_index,
                "file_path": file_ref,
                "llm_cache_list": list(data.get("cache_keys") or []),
                # Multimodal-specific metadata
                "is_multimodal": True,
                "modal_entity_name": entity_info["entity_name"],
                "original_type": data["content_type"],
                "page_idx": data["item_info"].get("page_idx", 0),
                # Satu kontrak dengan chunk teks (parent_child.py): kolom JSONB
                # ``sidecar`` yang sudah ada; ``page_idx`` di atas tidak pernah disimpan.
                "sidecar": {
                    "page_start": data["item_info"].get("page_idx", 0),
                    "page_end": data["item_info"].get("page_idx", 0),
                },
            }

        self.logger.debug(
            f"Converted {len(chunks)} multimodal items to multimodal chunks format"
        )
        return chunks

    def _apply_chunk_template(
        self, content_type: str, original_item: Dict[str, Any], description: str
    ) -> str:
        """
        Apply the appropriate chunk template based on content type

        Args:
            content_type: Type of content (image, table, equation, generic)
            original_item: Original multimodal item data
            description: Enhanced description generated by the processor

        Returns:
            Formatted chunk content using the appropriate template
        """
        from hybrid_rag.ingestion.multimodal.multimodal_template import TEMPLATES

        try:
            if content_type == "image":
                # Prefer _original_url (persistent S3 URL) over img_path (tmp local path)
                image_path = original_item.get("_original_url") or original_item.get("img_path", "")
                captions = normalize_caption_list(
                    original_item.get(
                        "image_caption", original_item.get("img_caption", [])
                    )
                )
                footnotes = normalize_caption_list(
                    original_item.get(
                        "image_footnote", original_item.get("img_footnote", [])
                    )
                )
                section_path = original_item.get("_section_path", "")
                neighbor_text = " ".join(
                    t
                    for t in (
                        original_item.get("_leading_text", ""),
                        original_item.get("_trailing_text", ""),
                    )
                    if t
                )

                return TEMPLATES["image_chunk"].format(
                    section_path=section_path if section_path else "None",
                    neighbor_text=neighbor_text if neighbor_text else "None",
                    image_path=image_path,
                    captions=", ".join(captions) if captions else "None",
                    footnotes=", ".join(footnotes) if footnotes else "None",
                    enhanced_caption=description,
                )

            elif content_type == "table":
                table_img_path = original_item.get("img_path", "")
                table_caption = normalize_caption_list(
                    original_item.get("table_caption", [])
                )
                table_body = format_table_body(get_table_body(original_item))
                table_footnote = normalize_caption_list(
                    original_item.get("table_footnote", [])
                )

                return TEMPLATES["table_chunk"].format(
                    table_img_path=table_img_path,
                    table_caption=", ".join(table_caption) if table_caption else "None",
                    table_body=table_body,
                    table_footnote=", ".join(table_footnote)
                    if table_footnote
                    else "None",
                    enhanced_caption=description,
                )

            elif content_type == "equation":
                equation_text, equation_format = get_equation_text_and_format(
                    original_item
                )

                return TEMPLATES["equation_chunk"].format(
                    equation_text=equation_text,
                    equation_format=equation_format,
                    enhanced_caption=description,
                )

            else:  # generic or unknown types
                content = str(original_item.get("content", original_item))

                return TEMPLATES["generic_chunk"].format(
                    content_type=content_type.title(),
                    content=content,
                    enhanced_caption=description,
                )

        except Exception as e:
            self.logger.warning(
                f"Error applying chunk template for {content_type}: {e}"
            )
            # Fallback to just the description if template fails
            return description

    async def _store_modal_chunks(
        self, chunks: Dict[str, Any]
    ):
        """Store chunks to storage"""
        try:
            # Store in text_chunks storage (required for extract_entities)
            await self.text_chunks.upsert(chunks)

            # Store in chunks vector database for retrieval
            await self.chunks_vdb.upsert(chunks)

            self.logger.debug(f"Stored {len(chunks)} multimodal chunks to storage")

        except Exception as e:
            self.logger.error(f"Error storing chunks to storage: {e}")
            raise

    async def _store_multimodal_main_entities(
        self,
        multimodal_data_list: List[Dict[str, Any]],
        modal_chunks: Dict[str, Any],
        file_path: str,
        doc_id: str = None,
    ):
        """
        Store multimodal main entities to entities_vdb and full_entities.
        This ensures that entities like "TableName (table)" are properly indexed.

        Args:
            multimodal_data_list: List of processed multimodal data with entity info
            modal_chunks: Modal chunks, already formatted with their templates
            file_path: File path for the entities
            doc_id: Document ID for full_entities storage
        """
        if not multimodal_data_list:
            return

        # Create entities_vdb entries for all multimodal main entities
        entities_to_store = {}

        # Use full path or basename based on config
        file_ref = os.path.basename(file_path)

        for data in multimodal_data_list:
            entity_info = data["entity_info"]
            entity_name = entity_info["entity_name"]
            description = data["description"]
            content_type = data["content_type"]
            original_item = data["original_item"]

            # Apply the same chunk template to get the formatted content
            formatted_chunk_content = self._apply_chunk_template(
                content_type, original_item, description
            )

            # Generate chunk_id using the formatted content (same as in _to_modal_chunks)
            chunk_id = compute_mdhash_id(formatted_chunk_content, prefix="chunk-")

            # Generate entity_id with the same hashing as text entities
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")

            # Create entity data in the graph's node format
            entity_data = {
                "entity_name": entity_name,
                "entity_type": entity_info.get("entity_type", content_type),
                "content": entity_info.get("summary", description),
                "source_id": chunk_id,
                "file_path": file_ref,
            }

            entities_to_store[entity_id] = entity_data

        if entities_to_store:
            try:
                # Store entities in knowledge graph. Under the graph keyed lock: a
                # batched merge of another document holds these names for its whole
                # merge, and a write slipped in meanwhile would be overwritten by its
                # flush.
                names = list(dict.fromkeys(d["entity_name"] for d in entities_to_store.values()))
                namespace = f"{self.workspace}:GraphDB" if self.workspace else "GraphDB"
                async with get_storage_keyed_lock(names, namespace=namespace):
                    for entity_id, entity_data in entities_to_store.items():
                        entity_name = entity_data["entity_name"]

                        # Create node data for knowledge graph
                        node_data = {
                            "entity_id": entity_name,
                            "entity_type": entity_data["entity_type"],
                            "description": entity_data["content"],
                            "source_id": entity_data["source_id"],
                            "file_path": entity_data["file_path"],
                            "created_at": int(time.time()),
                        }

                        # Store in knowledge graph
                        await self.chunk_entity_relation_graph.upsert_node(
                            entity_name, node_data
                        )

                # Store in entities_vdb
                await self.entities_vdb.upsert(entities_to_store)
                await self.entities_vdb.index_done_callback()

                # NEW: Store multimodal main entities in full_entities storage
                if doc_id and self.full_entities:
                    await self._store_multimodal_entities_to_full_entities(
                        entities_to_store, doc_id
                    )

                self.logger.debug(
                    f"Stored {len(entities_to_store)} multimodal main entities to knowledge graph, entities_vdb, and full_entities"
                )

            except Exception as e:
                self.logger.error(f"Error storing multimodal main entities: {e}")
                raise

    async def _store_multimodal_entities_to_full_entities(
        self, entities_to_store: Dict[str, Any], doc_id: str
    ):
        """
        Store multimodal main entities to full_entities storage.

        Args:
            entities_to_store: Dictionary of entities to store
            doc_id: Document ID for grouping entities
        """
        try:
            # Get current full_entities data for this document
            current_doc_entities = await self.full_entities.get_by_id(doc_id)

            if current_doc_entities is None:
                # Create new document entry
                entity_names = [
                    entity_data["entity_name"]
                    for entity_data in entities_to_store.values()
                ]
                doc_entities_data = {
                    "entity_names": entity_names,
                    "count": len(entity_names),
                    "update_time": int(time.time()),
                }
            else:
                # Update existing document entry while preserving any existing
                # metadata fields stored by the text pipeline.
                existing_entity_names = list(
                    current_doc_entities.get("entity_names", [])
                )
                seen_entity_names = set(existing_entity_names)

                for entity_data in entities_to_store.values():
                    entity_name = entity_data["entity_name"]
                    if entity_name not in seen_entity_names:
                        existing_entity_names.append(entity_name)
                        seen_entity_names.add(entity_name)

                doc_entities_data = {
                    **current_doc_entities,
                    "entity_names": existing_entity_names,
                    "count": len(existing_entity_names),
                    "update_time": int(time.time()),
                }

            # Store updated data
            await self.full_entities.upsert({doc_id: doc_entities_data})
            await self.full_entities.index_done_callback()

            self.logger.debug(
                f"Added {len(entities_to_store)} multimodal main entities to full_entities for doc {doc_id}"
            )

        except Exception as e:
            self.logger.error(
                f"Error storing multimodal entities to full_entities: {e}"
            )
            raise

    async def _extract_modal_entities(
        self,
        modal_chunks: Dict[str, Any],
        domain: str | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> List[Tuple]:
        """Extract entities and relations from modal chunks with the text pipeline's extractor."""
        from hybrid_rag.storage.shared import (
            get_namespace_data,
            get_namespace_lock,
        )
        from hybrid_rag.ingestion.knowledge_graph.extraction import extract_entities

        # This workspace's pipeline status, as on the text side (worker.py):
        # without the workspace these resolve to the *default* workspace's, so a
        # cancel in this workspace would not reach its multimodal extraction.
        pipeline_status = await get_namespace_data("pipeline_status", workspace=self.workspace)
        pipeline_status_lock = get_namespace_lock("pipeline_status", workspace=self.workspace)

        # Same extractor as the text chunks
        chunk_results = await extract_entities(
            chunks=modal_chunks,
            global_config={**self._build_global_config(), "kg_domain": domain},
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            llm_response_cache=self.llm_response_cache,
            text_chunks_storage=self.text_chunks,
            semaphore=semaphore,
        )

        self.logger.info(
            f"Extracted entities from {len(modal_chunks)} multimodal chunks"
        )
        return chunk_results

    async def _batch_add_belongs_to_relations_type_aware(
        self, chunk_results: List[Tuple], multimodal_data_list: List[Dict[str, Any]]
    ) -> List[Tuple]:
        """Add belongs_to relations for multimodal entities"""
        # Create mapping from chunk_id to modal_entity_name
        chunk_to_modal_entity = {}
        chunk_to_file_path = {}

        for data in multimodal_data_list:
            description = data["description"]
            content_type = data["content_type"]
            original_item = data["original_item"]

            # Use the same template formatting as in _to_modal_chunks
            formatted_chunk_content = self._apply_chunk_template(
                content_type, original_item, description
            )
            chunk_id = compute_mdhash_id(formatted_chunk_content, prefix="chunk-")

            chunk_to_modal_entity[chunk_id] = data["entity_info"]["entity_name"]
            chunk_to_file_path[chunk_id] = data.get("file_path", "multimodal_content")

        enhanced_chunk_results = []
        belongs_to_count = 0

        for maybe_nodes, maybe_edges in chunk_results:
            # Find corresponding modal_entity_name for this chunk
            chunk_id = None
            for nodes_dict in maybe_nodes.values():
                if nodes_dict:
                    chunk_id = nodes_dict[0].get("source_id")
                    break

            if chunk_id and chunk_id in chunk_to_modal_entity:
                modal_entity_name = chunk_to_modal_entity[chunk_id]
                file_path = chunk_to_file_path.get(chunk_id, "multimodal_content")

                # Add belongs_to relations for all extracted entities
                for entity_name in maybe_nodes.keys():
                    if entity_name != modal_entity_name:  # Avoid self-relation
                        belongs_to_relation = {
                            "src_id": entity_name,
                            "tgt_id": modal_entity_name,
                            "description": f"Entity {entity_name} belongs to {modal_entity_name}",
                            "keywords": "belongs_to,part_of,contained_in",
                            "source_id": chunk_id,
                            "weight": 1.0,
                            "file_path": file_path,
                        }

                        # Add to maybe_edges
                        edge_key = (entity_name, modal_entity_name)
                        if edge_key not in maybe_edges:
                            maybe_edges[edge_key] = []
                        maybe_edges[edge_key].append(belongs_to_relation)
                        belongs_to_count += 1

            enhanced_chunk_results.append((maybe_nodes, maybe_edges))

        self.logger.info(
            f"Added {belongs_to_count} belongs_to relations for multimodal entities"
        )
        return enhanced_chunk_results
