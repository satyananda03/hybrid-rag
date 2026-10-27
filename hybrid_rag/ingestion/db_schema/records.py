"""Render a :class:`DatabaseSpec` into the chunks and KG records hybrid_rag stores. Pure.

The shapes are exactly what the document pipeline hands its storages, so they are
written unchanged:

- chunks: ``{doc_id}-chunk-NNN`` ids with ``tokens``, ``chunk_order_index``,
  ``full_doc_id``, ``content``, ``file_path`` — as
  ``build_chunks_dict_from_chunking_result`` produces them;
- nodes / edges: the records ``merge_nodes_and_edges`` takes. Each entity and
  each table pair gets exactly ONE record, so the merge keeps its description
  as written and never asks the LLM to summarise.

Every node and edge cites the chunk its text came from (``source_id``): that is
how retrieval pulls the table card in for a matched column, and how deletion finds
what this document contributed.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

from hybrid_rag.core.constants import (
    DB_ENTITY_COLUMN,
    DB_ENTITY_CONCEPT,
    DB_ENTITY_DATABASE,
    DB_ENTITY_DOMAIN,
    DB_ENTITY_SCHEMA,
    DB_ENTITY_TABLE,
    DB_ENTITY_VALUE,
    GRAPH_FIELD_SEP,
)
from hybrid_rag.core.tokenizer import Tokenizer
from hybrid_rag.ingestion.db_schema.spec import (
    ColumnSpec,
    ConceptSpec,
    DatabaseSpec,
    DomainSpec,
    ForeignKeySpec,
    SchemaSpec,
    TableSpec,
)

logger = logging.getLogger(__name__)

_SUMMARY_CHARS = 200  # one-line table summary in a schema overview
_PART_LABEL_TOKENS = 16  # room for " (bagian 2/3)" when a text is split
_FK_WEIGHT = 2.0  # joins outrank containment when retrieval sorts a node's edges
_MAX_VALUE_NODES = 50  # a longer list is near free text: kept on the card, no node per value


# Entity names. Containers are prefixed: the database and its schema often share
# a name (iccn_dwh / iccn_dwh) and would otherwise collapse into one node. Tables
# and columns keep the qualified identifier SQL uses.
def database_entity(database: str) -> str:
    return f"database:{database}"


def schema_entity(schema: str) -> str:
    return f"schema:{schema}"


def table_entity(schema: str, table: str) -> str:
    return f"{schema}.{table}"


def column_entity(schema: str, table: str, column: str) -> str:
    return f"{schema}.{table}.{column}"


def concept_entity(concept: str) -> str:
    return f"concept:{concept}"


def domain_entity(domain: str) -> str:
    return f"domain:{domain}"


def value_entity(column: str, value: str) -> str:
    """One valid value of ``column`` (a :func:`column_entity` name). Prefixed and
    qualified, so "Paten" never merges with a document's entity or another column's."""
    return f"value:{column}={value}"


@dataclass(frozen=True)
class SchemaRecords:
    chunks: dict[str, dict]
    nodes: dict[str, list[dict]]
    edges: dict[tuple[str, str], list[dict]]
    stats: dict[str, int]

    @property
    def chunk_results(self) -> list[tuple[dict, dict]]:
        """The ``chunk_results`` argument of ``merge_nodes_and_edges``."""
        return [(self.nodes, self.edges)]


def build_records(
    spec: DatabaseSpec,
    *,
    doc_id: str,
    file_path: str,
    tokenizer: Tokenizer,
    max_tokens: int,
) -> SchemaRecords:
    """Chunks, nodes and edges for one database.

    ``max_tokens`` caps each chunk (the embedding limit): an overview or a table
    card that would exceed it is split, every part repeating its heading.
    """
    return _Builder(spec, doc_id, file_path, tokenizer, max_tokens).build()


@dataclass(frozen=True)
class _ForeignKey:
    """One foreign key, resolved to the ``(schema, table)`` pair it joins."""

    source: tuple[str, str]
    target: tuple[str, str]
    spec: ForeignKeySpec

    @property
    def join_on(self) -> str:
        src, tgt = table_entity(*self.source), table_entity(*self.target)
        return " AND ".join(
            f"{src}.{a} = {tgt}.{b}"
            for a, b in zip(self.spec.constrained_columns, self.spec.referred_columns)
        )


@dataclass
class _Builder:
    spec: DatabaseSpec
    doc_id: str
    file_path: str
    tokenizer: Tokenizer
    max_tokens: int
    chunks: dict[str, dict] = field(init=False, default_factory=dict)
    nodes: dict[str, list[dict]] = field(init=False, default_factory=dict)
    edges: dict[tuple[str, str], list[dict]] = field(init=False, default_factory=dict)
    timestamp: int = field(init=False, default_factory=lambda: int(time.time()))
    table_keys: set[tuple[str, str]] = field(init=False, default_factory=set)
    outgoing: defaultdict[tuple[str, str], list[_ForeignKey]] = field(
        init=False, default_factory=lambda: defaultdict(list)
    )
    incoming: defaultdict[tuple[str, str], list[_ForeignKey]] = field(
        init=False, default_factory=lambda: defaultdict(list)
    )
    table_cards: dict[tuple[str, str], list[str]] = field(init=False, default_factory=dict)
    overview_ids: list[str] = field(init=False, default_factory=list)
    domain_tables: defaultdict[str, list[tuple[str, str]]] = field(
        init=False, default_factory=lambda: defaultdict(list)
    )

    def __post_init__(self) -> None:
        self.table_keys = {(s.name, t.name) for s, t in self.spec.iter_tables()}
        for schema, table in self.spec.iter_tables():
            if table.domain:
                self.domain_tables[table.domain].append((schema.name, table.name))
            for fk_spec in table.foreign_keys:
                fk = _ForeignKey((schema.name, table.name), fk_spec.target(schema.name), fk_spec)
                self.outgoing[fk.source].append(fk)
                if fk.target in self.table_keys:
                    self.incoming[fk.target].append(fk)
                else:
                    logger.warning(
                        f"[db_schema] {table_entity(*fk.source)}: foreign key to "
                        f"{table_entity(*fk.target)}, which this schema file does not "
                        f"define — kept on the table card, no graph relation"
                    )

    def build(self) -> SchemaRecords:
        self._add_database()
        for schema in self.spec.schemas.values():
            self._add_schema(schema)
            for table in schema.tables.values():
                self._add_table(schema, table)
        self._add_foreign_keys()
        self._add_domains()  # after the tables: each domain edge cites its table's card
        for concept in self.spec.business_concepts:
            self._add_concept(concept)  # after the columns: its card reuses their text
        return SchemaRecords(
            chunks=self.chunks,
            nodes=self.nodes,
            edges=self.edges,
            stats={
                "schemas": len(self.spec.schemas),
                "tables": len(self.table_keys),
                "columns": sum(len(t.columns) for _, t in self.spec.iter_tables()),
                "foreign_keys": sum(len(fks) for fks in self.outgoing.values()),
                "business_concepts": len(self.spec.business_concepts),
                "domains": len(self.spec.domains),
                "values": sum(r[0]["entity_type"] == DB_ENTITY_VALUE for r in self.nodes.values()),
            },
        )

    # Entities, outermost first

    def _add_database(self) -> None:
        db = self.spec.name
        lines = [f"- {s.name}: {len(s.tables)} tabel" for s in self.spec.schemas.values()]
        if self.spec.domains:
            lines += ["", "## Domain bisnis"] + [
                f"- {d.name}: {_join(_summary(d.description), self._domain_tables_sentence(d))}"
                for d in self.spec.domains
            ]
        if self.spec.business_concepts:
            lines += ["", "## Konsep bisnis"] + [
                f"- {c.name}: {_summary(c.description)}" for c in self.spec.business_concepts
            ]
        self.overview_ids = [
            self._add_chunk(text)
            for text, _ in self._paged(f"# Database {db}", f"{self.spec.description}\n\n## Schema", lines)
        ]
        self._add_node(
            database_entity(db), DB_ENTITY_DATABASE,
            _join(f"Database {db}.", self.spec.description),
            self.overview_ids, {"database_name": db}, display_name=db,
        )

    def _add_schema(self, schema: SchemaSpec) -> None:
        db = self.spec.name
        lines = [
            f"- {table_entity(schema.name, t.name)}: {_summary(t.description)}"
            for t in schema.tables.values()
        ]
        chunk_ids = [
            self._add_chunk(text)
            for text, _ in self._paged(f"# Schema {schema.name} (database {db})", "## Tabel", lines)
        ]
        tables = ", ".join(t.name for t in schema.tables.values())
        self._add_node(
            schema_entity(schema.name), DB_ENTITY_SCHEMA,
            f"Schema {schema.name} di database {db}, berisi {len(schema.tables)} tabel: {tables}.",
            chunk_ids, {"database_name": db, "schema_name": schema.name}, display_name=schema.name,
        )
        self._add_edge(
            database_entity(db), schema_entity(schema.name), "contains_schema", "database,schema",
            f"Database {db} memuat schema {schema.name}.", chunk_ids[:1],
        )

    def _add_table(self, schema: SchemaSpec, table: TableSpec) -> None:
        key = (schema.name, table.name)
        name = table_entity(*key)
        fk_targets = {
            col: column_entity(*fk.target, ref)
            for fk in self.outgoing[key]
            for col, ref in zip(fk.spec.constrained_columns, fk.spec.referred_columns)
        }
        lines = [_column_line(c, fk_targets.get(c.name)) for c in table.columns]
        card_ids: list[str] = []
        column_card: dict[str, str] = {}
        title = f"# Tabel {_labelled(name, table.label)}"
        for text, indexes in self._paged(title, self._card_intro(schema, table), lines):
            card_ids.append(self._add_chunk(text))
            column_card.update({table.columns[i].name: card_ids[-1] for i in indexes})
        self.table_cards[key] = card_ids

        self._add_node(
            name, DB_ENTITY_TABLE,
            _join(f"Tabel {_labelled(name, table.label)} di schema {schema.name}, database {self.spec.name}.",
                  table.description, _primary_key_sentence(table)),
            card_ids,
            {"database_name": self.spec.name, "schema_name": schema.name,
             "table_name": table.name, "primary_keys": list(table.primary_keys)},
            display_name=table.label or table.name,
        )
        self._add_edge(
            schema_entity(schema.name), name, "contains_table", "schema,tabel",
            f"Schema {schema.name} memuat tabel {name}.", card_ids[:1],
        )
        for column in table.columns:
            self._add_column(schema, table, column, column_card[column.name], fk_targets.get(column.name))

    def _add_column(
        self, schema: SchemaSpec, table: TableSpec, column: ColumnSpec,
        card_id: str, fk_target: str | None,
    ) -> None:
        table_name = table_entity(schema.name, table.name)
        name = column_entity(schema.name, table.name, column.name)
        properties = {
            "database_name": self.spec.name, "schema_name": schema.name,
            "table_name": table.name, "column_name": column.name, "data_type": column.type,
            "is_primary_key": column.is_primary_key, "is_nullable": column.is_nullable,
        }
        if column.enum_values:
            properties["enum_values"] = list(column.enum_values)
        if column.vector_source:
            properties["vector_source"] = column.vector_source
        self._add_node(
            name, DB_ENTITY_COLUMN,
            _join(f"Kolom {column.name} ({_column_tags(column, fk_target)}) pada tabel {table_name}.",
                  _column_notes(column)),
            [card_id], properties, display_name=column.label or column.name,
        )
        self._add_edge(
            table_name, name, "has_column", "kolom,tabel",
            f"Tabel {table_name} memiliki kolom {column.name} ({column.type}).", [card_id],
        )
        if len(column.enum_values) <= _MAX_VALUE_NODES:
            for value in column.enum_values:
                self._add_value(name, value, card_id)

    def _add_value(self, column: str, value: str, card_id: str) -> None:
        # Cites the column's table card, so a question naming the value ("paten")
        # brings the table, the column and the filter to write.
        name = value_entity(column, value)
        self._add_node(
            name, DB_ENTITY_VALUE, f"Nilai '{value}' pada kolom {column}.",
            [card_id], {"database_name": self.spec.name, "column": column, "value": value},
            display_name=value,
        )
        self._add_edge(
            column, name, "has_value", "nilai valid,enum,kolom",
            f"Kolom {column} dapat bernilai '{value}'.", [card_id],
        )

    def _add_foreign_keys(self) -> None:
        # The graph keeps one edge per table pair, whichever way it points, so every
        # foreign key between two tables shares one record.
        by_pair: dict[tuple[str, str], list[_ForeignKey]] = defaultdict(list)
        for fk in (fk for fks in self.incoming.values() for fk in fks):
            if fk.source == fk.target:
                continue  # self-reference: on the card only, the graph drops self-loops
            pair = tuple(sorted((table_entity(*fk.source), table_entity(*fk.target))))
            by_pair[pair].append(fk)
        for (a, b), fks in by_pair.items():
            sources = list(dict.fromkeys(self.table_cards[fk.source][0] for fk in fks))
            self._add_edge(
                a, b, "references", "foreign key,join,relasi",
                " ".join(_foreign_key_sentence(fk) for fk in fks), sources,
                weight=_FK_WEIGHT, properties={"join_on": [fk.join_on for fk in fks]},
            )

    def _add_domains(self) -> None:
        db = self.spec.name
        for domain in self.spec.domains:
            name = domain_entity(domain.name)
            self._add_node(
                name, DB_ENTITY_DOMAIN,
                _join(f"Domain bisnis {domain.name} di database {db}.", domain.description,
                      self._domain_tables_sentence(domain)),
                self.overview_ids, {"database_name": db, "domain_name": domain.name},
                display_name=domain.name,
            )
            self._add_edge(
                database_entity(db), name, "contains_domain", "database,domain bisnis",
                f"Database {db} mencakup domain bisnis {domain.name}.", self.overview_ids[:1],
            )
            for key in self.domain_tables[domain.name]:
                self._add_edge(
                    name, table_entity(*key), "covers_table", "domain bisnis,tabel",
                    f"Domain bisnis {domain.name} mencakup tabel {table_entity(*key)}.",
                    self.table_cards[key][:1],
                )

    def _add_concept(self, concept: ConceptSpec) -> None:
        name = concept_entity(concept.name)
        columns = concept.referenced_columns()
        column_nodes = {c: self.nodes[c][0] for c in columns}
        lines = [f"- {node['description']}" for node in column_nodes.values()]
        card_ids = [
            self._add_chunk(text)
            for text, _ in self._paged(f"# Konsep bisnis {concept.name}", self._concept_intro(concept), lines)
        ]
        properties = {
            "database_name": self.spec.name, "sql": concept.sql, "from": _from_clause(concept),
            "filters": list(concept.filters), "columns": columns,
        }
        if concept.time_column:
            properties["time_column"] = concept.time_column
        self._add_node(
            name, DB_ENTITY_CONCEPT,
            _join(f"Konsep bisnis {concept.name} di database {self.spec.name}.", concept.description),
            card_ids, properties, display_name=concept.name,
            raw_name=GRAPH_FIELD_SEP.join(concept.synonyms),  # merged into the node's aliases
        )
        # Each edge also cites the card of the column's table, so matching the concept
        # brings the column types and valid values along.
        for column, node in column_nodes.items():
            role = "sebagai kolom waktu" if column == concept.time_column else "dalam rumusnya"
            self._add_edge(
                name, column, "uses_column", "konsep bisnis,metrik,kolom",
                f"Konsep bisnis {concept.name} memakai kolom {column} {role}.",
                [*card_ids[:1], *node["source_id"].split(GRAPH_FIELD_SEP)],
            )

    # Cards

    def _card_intro(self, schema: SchemaSpec, table: TableSpec) -> str:
        key = (schema.name, table.name)
        location = f"Database: {self.spec.name} · Schema: {schema.name}"
        if table.domain:
            location += f" · Domain: {table.domain}"
        parts = [location, table.description.strip()]
        if table.primary_keys:
            parts.append(f"Primary key: {', '.join(table.primary_keys)}")
        if self.outgoing[key]:
            parts.append("Foreign key keluar:\n" + "\n".join(
                f"- {', '.join(fk.spec.constrained_columns)} → {table_entity(*fk.target)}"
                f"({', '.join(fk.spec.referred_columns)}) — JOIN ON {fk.join_on}"
                for fk in self.outgoing[key]
            ))
        if self.incoming[key]:
            parts.append("Direferensikan oleh:\n" + "\n".join(
                f"- {table_entity(*fk.source)}({', '.join(fk.spec.constrained_columns)}) — JOIN ON {fk.join_on}"
                for fk in self.incoming[key]
            ))
        parts.append("## Kolom")
        return "\n\n".join(p for p in parts if p)

    def _domain_tables_sentence(self, domain: DomainSpec) -> str:
        tables = [table_entity(*key) for key in self.domain_tables[domain.name]]
        return f"Tabel: {', '.join(tables)}." if tables else ""

    def _concept_intro(self, concept: ConceptSpec) -> str:
        sections = {
            "Rumus SQL": concept.sql.strip(),
            "FROM": _from_clause(concept),
            "Filter wajib (WHERE)": "\n".join(f"- {f}" for f in concept.filters),
            "Kolom waktu": concept.time_column or "",
        }
        parts = [
            f"Database: {self.spec.name}",
            f"Sinonim: {', '.join(concept.synonyms)}" if concept.synonyms else "",
            concept.description.strip(),
            *(f"## {title}\n{body}" for title, body in sections.items() if body),
            "## Kolom",
        ]
        return "\n\n".join(p for p in parts if p)

    # Writers

    def _paged(self, title: str, intro: str, lines: list[str]) -> list[tuple[str, list[int]]]:
        """``title + intro + lines`` as texts within ``max_tokens``, each repeating
        the heading; returns each text with the indexes of the lines it holds.

        A single line larger than the budget still gets a part of its own; the
        embedding client truncates it.
        """
        budget = self.max_tokens - self._tokens(f"{title}\n{intro}") - _PART_LABEL_TOKENS
        groups: list[list[int]] = [[]]
        used = 0
        for i, line in enumerate(lines):
            cost = self._tokens(line) + 1  # + the newline joining it
            if groups[-1] and used + cost > budget:
                groups.append([])
                used = 0
            groups[-1].append(i)
            used += cost
        total = len(groups)
        return [
            (
                f"{title}{f' (bagian {n}/{total})' if total > 1 else ''}\n{intro}\n"
                + "\n".join(lines[i] for i in indexes),
                indexes,
            )
            for n, indexes in enumerate(groups, start=1)
        ]

    def _add_chunk(self, content: str) -> str:
        order = len(self.chunks)
        chunk_id = f"{self.doc_id}-chunk-{order:03d}"
        self.chunks[chunk_id] = {
            "tokens": self._tokens(content),
            "chunk_order_index": order,
            "full_doc_id": self.doc_id,
            "content": content,
            "file_path": self.file_path,
        }
        return chunk_id

    def _add_node(
        self, name: str, entity_type: str, description: str,
        chunk_ids: list[str], properties: dict, *, display_name: str, raw_name: str = "",
    ) -> None:
        # ``display_name`` is what the graph viewer shows; ``name`` stays the SQL identifier.
        self.nodes[name] = [{
            "entity_name": name,
            "entity_type": entity_type,
            "description": description,
            "source_id": GRAPH_FIELD_SEP.join(chunk_ids),
            "file_path": self.file_path,
            "timestamp": self.timestamp,
            "properties": {**properties, "display_name": display_name},
            **({"raw_name": raw_name} if raw_name else {}),
        }]

    def _add_edge(
        self, src: str, tgt: str, relation: str, keywords: str, description: str,
        chunk_ids: list[str], *, weight: float = 1.0, properties: dict | None = None,
    ) -> None:
        self.edges[(src, tgt)] = [{
            "src_id": src,
            "tgt_id": tgt,
            "weight": weight,
            "description": description,
            "keywords": keywords,
            "relation_type": relation,
            "source_id": GRAPH_FIELD_SEP.join(chunk_ids),
            "file_path": self.file_path,
            "timestamp": self.timestamp,
            "properties": properties or {},
        }]

    def _tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))


# Text pieces shared by the cards and the entity descriptions


def _join(*parts: str) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())


def _summary(description: str) -> str:
    first_line = description.strip().split("\n", 1)[0]
    return first_line if len(first_line) <= _SUMMARY_CHARS else first_line[:_SUMMARY_CHARS] + "..."


def _labelled(name: str, label: str) -> str:
    return f"{name} ({label})" if label else name


def _primary_key_sentence(table: TableSpec) -> str:
    return f"Primary key: {', '.join(table.primary_keys)}." if table.primary_keys else ""


def _column_tags(column: ColumnSpec, fk_target: str | None) -> str:
    tags = [column.type]
    if column.is_primary_key:
        tags.append("PK")
    tags.append("NULL" if column.is_nullable else "NOT NULL")
    if fk_target:
        tags.append(f"FK → {fk_target}")
    return ", ".join(tags)


def _column_notes(column: ColumnSpec) -> str:
    notes = [f"Label: {column.label}." if column.label else "", column.description]
    if column.enum_values:
        notes.append(f"Nilai valid: {', '.join(column.enum_values)}.")
    if column.vector_source:
        notes.append(
            f"Embedding pgvector dari kolom {column.vector_source}, untuk pencarian semantik."
        )
    return _join(*notes)


def _column_line(column: ColumnSpec, fk_target: str | None) -> str:
    notes = _column_notes(column)
    return f"- {column.name} ({_column_tags(column, fk_target)})" + (f": {notes}" if notes else "")


def _from_clause(concept: ConceptSpec) -> str:
    return concept.from_.strip() or concept.tables()[0]  # one table needs no join


def _foreign_key_sentence(fk: _ForeignKey) -> str:
    name = f" ({fk.spec.name})" if fk.spec.name else ""
    return (
        f"Foreign key{name}: {table_entity(*fk.source)} merujuk {table_entity(*fk.target)}; "
        f"JOIN ON {fk.join_on}."
    )
