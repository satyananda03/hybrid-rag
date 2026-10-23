"""Human review of the stored graph: edit an entity or a relation, rename or merge an entity, undo an edit.

A host application (the dashboard's knowledge-graph panel) calls this. Every
change reaches every store the ingest wrote (graph, entity and relation
vectors, chunk tracking, per-document indexes, alias table) in the ingest's own
formats, so retrieval reads a reviewed entity exactly as it reads an ingested one.

What a reviewer fixed stays fixed: the node or edge records its
``reviewed_fields`` and the ingest merge and rebuild keep their stored value
(``extraction._reviewed``), while new chunks and files still attach to it. A
renamed entity leaves a ``user`` row in the alias table, so a later document
naming it the old way lands on the new name.

Every review is recorded in ``hybrid_rag_graph_edit_history``. A field edit can
be undone, newest first: the old value comes back exactly, and so does its lock
state. When the value came from the ingest, the field is unlocked again; if the
item's source chunks changed meanwhile (documents ingested or deleted while it
was locked), it is rebuilt from the cached extractions, so it ends as if it had
never been locked. Renames and merges are not undone: rename again instead, and
a merge cannot be split.

Multimodal entities (image, table, equation, drawing) are named and written by
the multimodal pipeline, which entity resolution deliberately skips: only their
name and description can be edited, they are never merged or rebuilt here,
their relations (generated, not extracted) are not editable, and their vector
text keeps the format it had. Re-ingesting their source document runs the VLM
again and replaces them.

Like consolidation, a change refuses to run while the workspace is ingesting:
graph locks are per process and ingestion runs in its own worker.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Iterable, get_args

from hybrid_rag.core.constants import GRAPH_FIELD_SEP
from hybrid_rag.core.source_ids import make_relation_chunk_key
from hybrid_rag.core.utils import compute_mdhash_id
from hybrid_rag.ingestion.knowledge_graph.entity_consolidation import EntityConsolidator
from hybrid_rag.ingestion.knowledge_graph.entity_resolution import (
    _MULTIMODAL_TYPES,
    AliasDecision,
    EntityResolver,
    _alias_id,
    match_key,
    type_key,
)
from hybrid_rag.ingestion.knowledge_graph.extraction import (
    _reviewed,
    _truncate_vdb_content,
    rebuild_knowledge_from_chunks,
)
from hybrid_rag.ingestion.knowledge_graph.schema import EntityType
from hybrid_rag.storage.postgres.sql import SQL_TEMPLATES
from hybrid_rag.storage.shared import get_storage_keyed_lock

logger = logging.getLogger(__name__)

ENTITY_FIELDS = frozenset({"description", "entity_type"})
MULTIMODAL_FIELDS = frozenset({"description"})
RELATION_FIELDS = frozenset({"description", "keywords"})
# Stored as the ingest stores them without an ontology: lowercase, no spaces.
ENTITY_TYPES = frozenset(type_key(t) for t in get_args(EntityType))
_REVIEW_PROPERTIES = ("reviewed_fields", "reviewed_by", "reviewed_at")

# {"fields": {field: {"before", "after", "was_reviewed"}}, "sources": <hash of the
# source chunks>}, plus "merged" (rename) or "rebuilt" / "undoes" (undo).
Changes = dict[str, Any]


class ReviewNotFound(LookupError):
    """The entity, relation or edit to review does not exist."""


class ReviewConflict(RuntimeError):
    """The change cannot run now: the name is taken, the value changed, or the workspace is ingesting."""


class GraphEditHistory:
    """``hybrid_rag_graph_edit_history``: every review of one workspace's graph, for audit and undo.

    Writes are best effort: the graph has already changed when they run, so a
    failed write is logged rather than reported as a failed review.
    """

    def __init__(self, workspace: str, db):
        self.workspace = workspace
        self.db = db

    async def record(self, action: str, target: str, changes: Changes, editor: str) -> None:
        try:
            await self.db.execute(SQL_TEMPLATES["insert_graph_edit"], {
                "workspace": self.workspace, "action": action, "target": target,
                "changes": json.dumps(changes), "edited_by": editor,
            })
        except Exception:
            logger.exception(f"[{self.workspace}] could not record graph review {action} {target!r}")

    async def edits(self, action: str, target: str) -> list[dict]:
        """Standing (not undone) edits of ``target``, newest first."""
        rows = await self.db.query(
            SQL_TEMPLATES["get_graph_edits"], [self.workspace, action, target], multirows=True
        )
        return [self._decoded(row) for row in rows or []]

    async def get(self, edit_id: int) -> dict | None:
        row = await self.db.query(SQL_TEMPLATES["get_graph_edit"], [self.workspace, edit_id])
        return self._decoded(row) if row else None

    async def mark_undone(self, edit_id: int) -> None:
        try:
            await self.db.execute(
                SQL_TEMPLATES["mark_graph_edit_undone"], {"workspace": self.workspace, "id": edit_id}
            )
        except Exception:
            logger.exception(f"[{self.workspace}] could not mark graph edit {edit_id} undone")

    @staticmethod
    def _decoded(row: dict) -> dict:
        changes = row["changes"]  # asyncpg hands JSONB back as a string
        return {**row, "changes": json.loads(changes) if isinstance(changes, str) else changes}


def _is_multimodal(node: dict) -> bool:
    return type_key(node.get("entity_type", "")) in _MULTIMODAL_TYPES


def _editable(node: dict) -> frozenset[str]:
    return MULTIMODAL_FIELDS if _is_multimodal(node) else ENTITY_FIELDS


def _clean(field: str, value: object) -> str:
    """A reviewer's value in the form the ingest stores it."""
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} cannot be empty")
    if GRAPH_FIELD_SEP in text:
        raise ValueError(f"{field} cannot contain {GRAPH_FIELD_SEP}")
    if field == "entity_type":
        text = type_key(text)
        if text not in ENTITY_TYPES:
            raise ValueError(f"entity_type must be one of: {', '.join(sorted(ENTITY_TYPES))}")
    if field == "keywords":  # the merge's own format
        text = ",".join(sorted({k.strip() for k in text.split(",") if k.strip()}))
    return text


def _require_editable(fields: Iterable[str], allowed: frozenset[str]) -> None:
    refused = set(fields) - allowed
    if refused:
        raise ValueError(f"Not editable here: {', '.join(sorted(refused))}")


def _field_changes(record: dict, fields: dict[str, object], allowed: frozenset[str]) -> dict[str, dict]:
    if not fields:
        raise ValueError("Nothing to update")
    _require_editable(fields, allowed)
    locked = _reviewed(record)
    changes = {}
    for field, value in fields.items():
        after = _clean(field, value)
        if after != record.get(field):
            changes[field] = {"before": record.get(field), "after": after, "was_reviewed": field in locked}
    return changes


def _require_unchanged(record: dict, fields: dict[str, dict]) -> None:
    """Rule of undo: the stored value is still what the edit wrote."""
    changed = sorted(f for f, c in fields.items() if record.get(f) != c["after"])
    if changed:
        raise ReviewConflict(f"{', '.join(changed)} changed since this edit; nothing was undone")
    if any(c["before"] is None for c in fields.values()):
        raise ValueError("This edit has no earlier value to restore")


def _review_properties(record: dict) -> dict:
    return {k: record[k] for k in _REVIEW_PROPERTIES if k in record}


def _review_props(fields: Iterable[str], reviewer: str) -> dict:
    """Review properties for the locked ``fields``; all ``None`` (Neo4j removes them) when none is left."""
    locked = sorted(fields)
    if not locked:
        return dict.fromkeys(_REVIEW_PROPERTIES)
    return {"reviewed_fields": locked, "reviewed_by": reviewer, "reviewed_at": int(time.time())}


def _fingerprint(chunk_ids: list[str]) -> str:
    return compute_mdhash_id(GRAPH_FIELD_SEP.join(sorted(chunk_ids)))


def _undoable(rows: list[dict]) -> dict[str, dict]:
    """Per field, the edit an undo reverts. ``rows`` are standing edits, newest first;
    an edit qualifies only when no newer one touched any of its fields."""
    out: dict[str, dict] = {}
    seen: set[str] = set()
    for row in rows:
        fields = row["changes"]["fields"]
        if seen.isdisjoint(fields):
            for field, change in fields.items():
                out[field] = {
                    "id": row["id"], "before": change["before"], "after": change["after"],
                    "edited_by": row["edited_by"], "create_time": row["create_time"],
                }
        seen |= fields.keys()
    return out


class GraphReviewer:
    """Review operations on the workspace of ``rag`` (an ``IngestionRAG``)."""

    def __init__(self, rag):
        self.rag = rag
        self.graph = rag.chunk_entity_relation_graph
        self.consolidator = EntityConsolidator(rag)
        self.history = GraphEditHistory(rag.workspace, rag.entity_aliases.db)
        self.namespace = f"{rag.workspace}:GraphDB" if rag.workspace else "GraphDB"

    # --- edits ------------------------------------------------------------------------

    async def edit_entity(self, name: str, fields: dict[str, object], reviewer: str) -> Changes:
        """Set ``description`` / ``entity_type`` (multimodal: ``description`` only)."""
        await self._refuse_while_ingesting()
        async with self._lock([name]):
            node = await self._node(name)
            changed = _field_changes(node, fields, _editable(node))
            if not changed:
                return {"fields": {}}
            values = {field: c["after"] for field, c in changed.items()}
            await self._write_entity(name, node, values, _reviewed(node) | changed.keys(), reviewer)
        changes = {"fields": changed, "sources": _fingerprint(await self._entity_chunks(name, node))}
        await self.rag._insert_done()
        await self.history.record("edit_entity", name, changes, reviewer)
        return changes

    async def rename_entity(self, old: str, new: str, allow_merge: bool, reviewer: str) -> Changes:
        """Rename ``old`` to ``new``; into an existing ``new`` only with ``allow_merge``.

        The node, its relations and every index move with it
        (``EntityConsolidator.merge``). ``changes["merged"]`` tells which it was.
        """
        new = _clean("entity_id", new)
        if new == old:
            raise ValueError("The new name is the current name")
        await self._refuse_while_ingesting()
        nodes = await self.graph.get_nodes_batch([old, new])
        if old not in nodes:
            raise ReviewNotFound(f"Entity {old!r} not found")
        node, target = nodes[old], nodes.get(new)
        multimodal = _is_multimodal(node)
        if target is not None:
            if multimodal or _is_multimodal(target):
                raise ValueError("Multimodal entities cannot be merged")
            if not allow_merge:
                raise ReviewConflict(f"Entity {new!r} already exists")

        previous = await self.rag.entities_vdb.get_by_id(compute_mdhash_id(old, prefix="ent-"))
        # A rename keeps what a reviewer locked on the node and its relations; a
        # merge does not (the merged text is new, and the target's own locks hold).
        node_stamp = _review_properties(node) if target is None else {}
        edge_stamps = await self._reviewed_relations(old) if target is None else {}
        await self.consolidator.merge(new, old, create_survivor=target is None)
        async with self._lock([new, *edge_stamps]):
            if node_stamp:
                await self.graph.upsert_node(new, {"entity_id": new, **node_stamp})
            for neighbour, stamp in edge_stamps.items():
                await self.graph.upsert_edge(new, neighbour, stamp)
            if multimodal:
                await self._write_entity_vector(new, await self._node(new), previous, old)
            elif match_key(old):
                etype = type_key(node.get("entity_type", ""))
                await self.rag.entity_aliases.add([AliasDecision(
                    _alias_id(etype, match_key(old)), old, etype, new, "user", f"graph review by {reviewer}", 1.0,
                )])
        await self.rag._insert_done()
        merged = target is not None
        changes = {"fields": {"entity_id": {"before": old, "after": new, "was_reviewed": False}}, "merged": merged}
        await self.history.record("merge_entity" if merged else "rename_entity", old, changes, reviewer)
        return changes

    async def edit_relation(self, src: str, tgt: str, fields: dict[str, object], reviewer: str) -> Changes:
        """Set ``description`` / ``keywords`` of the relation between ``src`` and ``tgt``."""
        await self._refuse_while_ingesting()
        async with self._lock([src, tgt]):
            edge = await self._edge(src, tgt)
            changed = _field_changes(edge, fields, RELATION_FIELDS)
            if not changed:
                return {"fields": {}}
            values = {field: c["after"] for field, c in changed.items()}
            await self._write_relation(src, tgt, edge, values, _reviewed(edge) | changed.keys(), reviewer)
        changes = {"fields": changed, "sources": _fingerprint(await self._relation_chunks(src, tgt, edge))}
        await self.rag._insert_done()
        await self.history.record("edit_relation", make_relation_chunk_key(src, tgt), changes, reviewer)
        return changes

    # --- undo -------------------------------------------------------------------------

    async def undoable_entity(self, name: str) -> dict[str, dict]:
        """Per field of ``name``, the edit an undo would revert."""
        return _undoable(await self.history.edits("edit_entity", name))

    async def undoable_relation(self, src: str, tgt: str) -> dict[str, dict]:
        return _undoable(await self.history.edits("edit_relation", make_relation_chunk_key(src, tgt)))

    async def undo(self, edit_id: int, reviewer: str) -> Changes:
        """Revert edit ``edit_id`` (the newest standing edit of each of its fields)."""
        row = await self.history.get(edit_id)
        if row is None or row["action"] not in {"edit_entity", "edit_relation"}:
            raise ReviewNotFound(f"No field edit {edit_id} in this workspace")
        if row["undone_time"] is not None:
            raise ReviewConflict(f"Edit {edit_id} is already undone")
        fields = row["changes"]["fields"]
        standing = _undoable(await self.history.edits(row["action"], row["target"]))
        if any(standing.get(field, {}).get("id") != edit_id for field in fields):
            raise ReviewConflict("A newer edit of this field exists; undo that one first")

        if row["action"] == "edit_entity":
            changes = await self._revert_entity(row["target"], row["changes"], reviewer)
        else:
            src, tgt = row["target"].split(GRAPH_FIELD_SEP, 1)  # make_relation_chunk_key
            changes = await self._revert_relation(src, tgt, row["changes"], reviewer)
        await self.history.mark_undone(edit_id)
        await self.history.record("undo", row["target"], {**changes, "undoes": edit_id}, reviewer)
        return changes

    async def _revert_entity(self, name: str, change: Changes, reviewer: str) -> Changes:
        fields = change["fields"]
        await self._refuse_while_ingesting()
        async with self._lock([name]):
            node = await self._node(name)
            _require_editable(fields, _editable(node))
            _require_unchanged(node, fields)
            values = {field: c["before"] for field, c in fields.items()}
            await self._write_entity(name, node, values, self._relocked(node, fields), reviewer)
        chunk_ids = await self._entity_chunks(name, node)
        rebuilt = self._needs_rebuild(fields, change, chunk_ids) and not _is_multimodal(node)
        if rebuilt:
            await self._rebuild(entities={name: chunk_ids})
        await self.rag._insert_done()
        return self._reverted(fields, await self._node(name), chunk_ids, rebuilt)

    async def _revert_relation(self, src: str, tgt: str, change: Changes, reviewer: str) -> Changes:
        fields = change["fields"]
        await self._refuse_while_ingesting()
        async with self._lock([src, tgt]):
            edge = await self._edge(src, tgt)
            _require_editable(fields, RELATION_FIELDS)
            _require_unchanged(edge, fields)
            values = {field: c["before"] for field, c in fields.items()}
            await self._write_relation(src, tgt, edge, values, self._relocked(edge, fields), reviewer)
        chunk_ids = await self._relation_chunks(src, tgt, edge)
        rebuilt = self._needs_rebuild(fields, change, chunk_ids)
        if rebuilt:
            await self._rebuild(relations={(src, tgt): chunk_ids})
        await self.rag._insert_done()
        return self._reverted(fields, await self._edge(src, tgt), chunk_ids, rebuilt)

    @staticmethod
    def _relocked(record: dict, fields: dict[str, dict]) -> set[str]:
        """Locks after the undo: a field goes back to the lock state it had before the edit."""
        return (_reviewed(record) - fields.keys()) | {f for f, c in fields.items() if c["was_reviewed"]}

    @staticmethod
    def _needs_rebuild(fields: dict[str, dict], change: Changes, chunk_ids: list[str]) -> bool:
        """An unlocked field whose sources changed while it was locked lacks their
        contribution: rebuild it, as the ingest would have merged it."""
        unlocked = any(not c["was_reviewed"] for c in fields.values())
        return unlocked and bool(chunk_ids) and _fingerprint(chunk_ids) != change.get("sources")

    @staticmethod
    def _reverted(fields: dict[str, dict], record: dict, chunk_ids: list[str], rebuilt: bool) -> Changes:
        return {
            "fields": {f: {"before": c["after"], "after": record.get(f), "was_reviewed": True} for f, c in fields.items()},
            "sources": _fingerprint(chunk_ids),
            "rebuilt": rebuilt,
        }

    async def _rebuild(self, entities: dict | None = None, relations: dict | None = None) -> None:
        """The document deletion's rebuild (``purge.py``), except that an entity missing
        from its cached extractions keeps its restored value."""
        rag = self.rag
        await rebuild_knowledge_from_chunks(
            entities_to_rebuild=entities or {},
            relationships_to_rebuild=relations or {},
            knowledge_graph_inst=self.graph,
            entities_vdb=rag.entities_vdb,
            relationships_vdb=rag.relationships_vdb,
            text_chunks_storage=rag.text_chunks,
            llm_response_cache=rag.llm_response_cache,
            global_config=rag._build_global_config(),
            entity_chunks_storage=rag.entity_chunks,
            relation_chunks_storage=rag.relation_chunks,
            doc_status_storage=rag.doc_status,
            entity_resolver=rag.entity_resolver,
            entities_from_relationships=False,
        )

    # --- helpers ----------------------------------------------------------------------

    def _lock(self, keys: Iterable[str]):
        return get_storage_keyed_lock(sorted(set(keys)), namespace=self.namespace)

    async def _refuse_while_ingesting(self) -> None:
        try:
            await self.consolidator._refuse_while_ingesting()
        except RuntimeError as e:
            raise ReviewConflict(str(e)) from e

    async def _node(self, name: str) -> dict:
        node = await self.graph.get_node(name)
        if node is None:
            raise ReviewNotFound(f"Entity {name!r} not found")
        return node

    async def _edge(self, src: str, tgt: str) -> dict:
        edge = await self.graph.get_edge(src, tgt)
        if edge is None:
            raise ReviewNotFound(f"Relation {src!r} - {tgt!r} not found")
        endpoints = await self.graph.get_nodes_batch([src, tgt])
        if any(_is_multimodal(n) for n in endpoints.values()):
            raise ValueError("Relations of multimodal entities cannot be edited")
        return edge

    async def _entity_chunks(self, name: str, node: dict) -> list[str]:
        return await EntityConsolidator._chunk_ids(self.rag.entity_chunks, name, node.get("source_id"))

    async def _relation_chunks(self, src: str, tgt: str, edge: dict) -> list[str]:
        key = make_relation_chunk_key(src, tgt)
        return await EntityConsolidator._chunk_ids(self.rag.relation_chunks, key, edge.get("source_id"))

    async def _reviewed_relations(self, name: str) -> dict[str, dict]:
        """Neighbour -> review properties, for each reviewed relation of ``name``."""
        pairs = [{"src": a, "tgt": b} for a, b in await self.graph.get_node_edges(name) or []]
        edges = await self.graph.get_edges_batch(pairs) if pairs else {}
        return {
            (tgt if src == name else src): _review_properties(edge)
            for (src, tgt), edge in edges.items() if edge.get("reviewed_fields")
        }

    async def _write_entity(self, name: str, node: dict, values: dict, locked: set[str], reviewer: str) -> None:
        await self.graph.upsert_node(name, {"entity_id": name, **values, **_review_props(locked, reviewer)})
        if "description" in values:
            previous = await self.rag.entities_vdb.get_by_id(compute_mdhash_id(name, prefix="ent-"))
            await self._write_entity_vector(name, {**node, **values}, previous, name)

    async def _write_relation(
        self, src: str, tgt: str, edge: dict, values: dict, locked: set[str], reviewer: str
    ) -> None:
        await self.graph.upsert_edge(src, tgt, {**values, **_review_props(locked, reviewer)})
        await self._write_relation_vector(src, tgt, {**edge, **values})

    async def _write_entity_vector(self, name: str, node: dict, previous: dict | None, previous_name: str) -> None:
        """Entity vector in the format its row had: ``name\\n[Alias: …\\n]description``
        (ingest merge) or the description alone (multimodal anchors, ``multimodal/analyze.py``)."""
        description = node.get("description") or ""
        content = (previous or {}).get("content")
        if content is not None and not content.startswith(f"{previous_name}\n"):
            text = description
        else:
            text = EntityResolver.vdb_content(name, description, node.get("aliases"))
        await self.rag.entities_vdb.upsert({compute_mdhash_id(name, prefix="ent-"): {
            "entity_name": name,
            "entity_type": node.get("entity_type"),
            "content": _truncate_vdb_content(text, self.rag._build_global_config(), f"entity:{name}"),
            "source_id": node.get("source_id"),
            "file_path": node.get("file_path"),
        }})

    async def _write_relation_vector(self, src: str, tgt: str, edge: dict) -> None:
        """Same id, order and text as the ingest merge (``_merge_edges_then_upsert``)."""
        src, tgt = sorted((src, tgt))
        vdb = self.rag.relationships_vdb
        rel_id = compute_mdhash_id(src + tgt, prefix="rel-")
        await vdb.delete([rel_id, compute_mdhash_id(tgt + src, prefix="rel-")])
        content = f"{edge['keywords']}\t{src}\n{tgt}\n{edge['description']}"
        await vdb.upsert({rel_id: {
            "src_id": src,
            "tgt_id": tgt,
            "source_id": edge.get("source_id"),
            "content": _truncate_vdb_content(content, self.rag._build_global_config(), f"relationship:{src}-{tgt}"),
            "keywords": edge["keywords"],
            "description": edge["description"],
            "weight": edge.get("weight"),
            "file_path": edge.get("file_path"),
        }})
