"""Multimodal ingestion.

``insert_content_list`` is the entry point: it splits a parsed content list
into text and multimodal blocks, sends the text through the document pipeline,
and captions each image / table / equation before writing it into the same
storages as a chunk plus its entities and relations.

Mixed into :class:`~hybrid_rag.ingestion.rag.IngestionRAG`, so ``self`` here is
the RAG itself — storages, pipeline and processors are all reachable directly.
"""

import os
from typing import Dict, List, Any
from hybrid_rag.ingestion.multimodal.content import (
    separate_content,
)

class _IngestionMixin:
    """Insert a parsed content list into the RAG — the ingestion orchestrator."""
    def _generate_content_based_doc_id(self, content_list: List[Dict[str, Any]]) -> str:
        """
        Generate doc_id based on document content

        Args:
            content_list: Parsed content list

        Returns:
            str: Content-based document ID with doc- prefix
        """
        from hybrid_rag.core.utils import compute_mdhash_id

        # Extract key content for ID generation
        content_hash_data = []

        for item in content_list:
            if isinstance(item, dict):
                # For text content, use the text
                if item.get("type") == "text" and item.get("text"):
                    content_hash_data.append(item["text"].strip())
                # For other content types, use key identifiers
                elif item.get("type") == "image" and item.get("img_path"):
                    content_hash_data.append(f"image:{item['img_path']}")
                elif item.get("type") == "table" and item.get("table_body"):
                    content_hash_data.append(f"table:{item['table_body']}")
                elif item.get("type") == "equation" and item.get("text"):
                    content_hash_data.append(f"equation:{item['text']}")
                else:
                    # For other types, use string representation
                    content_hash_data.append(str(item))

        # Create a content signature
        content_signature = "\n".join(content_hash_data)

        # Generate doc_id from content
        doc_id = compute_mdhash_id(content_signature, prefix="doc-")

        return doc_id

    async def _insert_text_content(
        self,
        text: str | list[str],
        *,
        file_paths: str | list[str] | None = None,
        ids: str | list[str] | None = None,
        multimodal_items: List[Dict[str, Any]] | None = None,
        ontology: str | None = None,
        entity_registry: str | None = None,
        entity_resolution: bool = True,
        entity_resolution_llm: bool = True,
        orphan_repair: bool = True,
        prune_isolated: bool = True,
        page_offsets: List[Dict[str, int]] | None = None,
        save_parent: bool = True,
    ) -> None:
        """Run the plain-text half of a document through the ingest pipeline.

        Content reaches us already parsed, so the pipeline's parse stage is a
        passthrough and chunking is always contextualized parent-child.
        """
        self.logger.info("Starting text content insertion...")
        await self.apipeline_enqueue_documents(
            input=text,
            ids=ids,
            file_paths=file_paths,
            multimodal_items=[list(multimodal_items or [])],
            ontology=ontology,
            entity_registry=entity_registry,
            entity_resolution=entity_resolution,
            entity_resolution_llm=entity_resolution_llm,
            orphan_repair=orphan_repair,
            prune_isolated=prune_isolated,
            page_offsets=[list(page_offsets or [])],
            save_parent=save_parent,
        )
        await self.apipeline_process_enqueue_documents()
        self.logger.info("Text content insertion complete")

    async def insert_content_list(
        self,
        content_list: List[Dict[str, Any]],
        file_path: str = "unknown_document",
        doc_id: str | None = None,
        ontology: str | None = None,
        entity_registry: str | None = None,
        entity_resolution: bool = True,
        entity_resolution_llm: bool = True,
        orphan_repair: bool = True,
        prune_isolated: bool = True,
        save_parent: bool = True,
    ):
        """
        Insert content list directly without document parsing

        Args:
            content_list: Pre-parsed content list containing text and multimodal items.
                         Each item should be a dictionary with the following structure:
                         - Text: {"type": "text", "text": "content", "page_idx": 0}
                         - Image: {"type": "image", "img_path": "/absolute/path/to/image.jpg",
                                  "image_caption": ["caption"], "image_footnote": ["note"], "page_idx": 1}
                         - Table: {"type": "table", "table_body": "markdown table",
                                  "table_caption": ["caption"], "table_footnote": ["note"], "page_idx": 2}
                         - Equation: {"type": "equation", "latex": "LaTeX formula",
                                     "text": "description", "page_idx": 3}
                         - Generic: {"type": "custom_type", "content": "any content", "page_idx": 4}
            file_path: Reference file path/name for citation (defaults to "unknown_document")
            doc_id: Optional document ID, if not provided will be generated from content
            ontology: File name (with or without ``.ttl``/``.owl``) of an ontology in
                    ``knowledge_graph/ontology/``. Extraction is then guided by its
                    schema and validated against it, as neo4j-graphrag does (see
                    knowledge_graph.ontology_schema). None uses the default
                    contract with no validation. Several ontologies may share a
                    workspace, so this is a property of the document; an unknown
                    or unparsable file raises here.
            entity_registry: File name (with or without ``.yaml``) of an entity
                    registry in ``knowledge_graph/entity_registry/``: curated
                    aliases and never_merge pairs applied when this document's
                    entities are merged (see knowledge_graph.entity_resolution).
                    Its one-word aliases are also the acronym glossary given to
                    contextualize, extraction, summaries and table/image analysis
                    (pass it to ``parse_file`` too for slide decks). None applies
                    only the deterministic rules; an unknown file raises here.
            entity_resolution: Merge this document's entity name variants into
                    canonical names before writing the graph (see
                    knowledge_graph.entity_resolution). False keeps the plain
                    exact-name merge for this document; entity_registry is then
                    ignored for merging, while its acronym glossary still guides
                    every LLM that writes text.
            entity_resolution_llm: Also let an LLM judge the name pairs the rules
                    cannot decide (qualifiers, typos, acronyms), comparing with
                    this workspace's existing entities. Only applies when
                    entity_resolution is True; a failing LLM call keeps the
                    rule-only result.
            orphan_repair: For each chunk whose extraction left entities with no
                    relationship, ask the LLM once more for the relationships the
                    text states about them (see knowledge_graph.entity_repairment).
                    Costs one extra call for those chunks only.
            prune_isolated: Once the batch is merged, delete this document's
                    entities that still have no relationship (multimodal anchors
                    excepted); their text stays searchable from the chunks.
            save_parent: Also store the parent chunks next to the embedded
                    children, so retrieval can return the full passage
                    (``return_parent``). Parents are never embedded; turn this
                    off only for documents that will never serve as evidence.

        Note:
            - img_path must be an absolute path to the image file
            - page_idx is where the content appears, used for citations: an int is a
              0-based page (PDF/DOCX), a str is an Excel sheet name; omit it for
              sources without pages
            - Items are processed in the order they appear in the list
        """
        # Ensure the multimodal processors are ready
        init_result = await self._ensure_multimodal_ready()
        if not init_result or not init_result.get("success"):
            raise RuntimeError(
                f"Multimodal initialization failed: {(init_result or {}).get('error', 'unknown error')}"
            )

        self.logger.info(
            f"Starting direct content list insertion for: {file_path} ({len(content_list)} items)"
        )

        # Generate doc_id based on content if not provided
        if doc_id is None:
            doc_id = self._generate_content_based_doc_id(content_list)

        file_ref = os.path.basename(file_path)

        # Hitungan blok per tipe — jejak murah untuk menelusuri dokumen.
        self.logger.info("\nContent Information:")
        self.logger.info(f"* Total blocks in content_list: {len(content_list)}")

        # Count elements by type
        block_types: Dict[str, int] = {}
        for block in content_list:
            if isinstance(block, dict):
                block_type = block.get("type", "unknown")
                if isinstance(block_type, str):
                    block_types[block_type] = block_types.get(block_type, 0) + 1

        self.logger.info("* Content block types:")
        for block_type, count in block_types.items():
            self.logger.info(f"  - {block_type}: {count}")

        # Step 1: Separate text and multimodal content
        text_content, multimodal_items, page_offsets = separate_content(content_list)

        # Step 2: Hand the whole document to the pipeline — the text body plus
        # the multimodal items riding along in full_docs. This runs even when the
        # text half is empty: an image-only document still needs its figures
        # captioned, and that now happens inside the pipeline run.
        await self._insert_text_content(
            text_content,
            file_paths=file_ref,
            ids=doc_id,
            multimodal_items=multimodal_items,
            ontology=ontology,
            entity_registry=entity_registry,
            entity_resolution=entity_resolution,
            entity_resolution_llm=entity_resolution_llm,
            page_offsets=page_offsets,
            save_parent=save_parent,
        )

        # No doc_status write here. The pipeline's state machine owns the status
        # from PENDING through PROCESSED; a second writer landing after it is how
        # a fully processed document ended up reported as "preprocessed".

        # Multimodal content is no longer processed here: it rides along to
        # ``full_docs.multimodal_items`` at enqueue and is captioned by the
        # pipeline's own run, inside ``process_single_document`` — one execution
        # path, one merge, one chunks_list write. See ingestion/multimodal/analyze.py.

        self.logger.info(f"Content list insertion complete for: {file_path}")
