"""The schema YAML format, validated once at the boundary.

Everything downstream reads these models and never the raw mapping, so a missing
key or a foreign key naming a column the table does not have fails here, with the
offending path, instead of surfacing as a half-written graph.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, model_validator

from hybrid_rag.ingestion.knowledge_graph.entity_resolution import MAX_ALIASES

# ``description:`` left empty in YAML loads as None; enum values may be numbers.
Text = Annotated[str, BeforeValidator(lambda v: "" if v is None else v)]
Value = Annotated[str, BeforeValidator(str)]


class _Spec(BaseModel):
    # Unknown keys (e.g. the per-column copy of ``foreign_keys``) are ignored.
    model_config = ConfigDict(extra="ignore", frozen=True)


class ForeignKeySpec(_Spec):
    name: Text = ""
    constrained_columns: list[str] = Field(min_length=1)
    referred_schema: str | None = None  # None: the constrained table's own schema
    referred_table: str
    referred_columns: list[str] = Field(min_length=1)

    def target(self, schema: str) -> tuple[str, str]:
        """``(schema, table)`` referred to, from a table living in ``schema``."""
        return self.referred_schema or schema, self.referred_table

    @model_validator(mode="after")
    def _columns_pair_up(self) -> ForeignKeySpec:
        if len(self.constrained_columns) != len(self.referred_columns):
            raise ValueError(
                f"foreign key {self.name or self.referred_table!r}: "
                f"{len(self.constrained_columns)} column(s) refer to {len(self.referred_columns)}"
            )
        return self


class ColumnSpec(_Spec):
    name: str
    label: Text = ""  # human-readable name for the graph viewer; empty shows ``name``
    type: str
    description: Text = ""
    is_primary_key: bool = False
    is_nullable: bool = True
    enum_values: list[Value] = []
    vector_source: str | None = None

    @model_validator(mode="after")
    def _enum_values_unique(self) -> ColumnSpec:
        # Each value becomes a graph node named after it; a repeat would fold into one.
        if len(set(self.enum_values)) != len(self.enum_values):
            raise ValueError(f"column {self.name!r}: duplicate enum_values")
        return self


class TableSpec(_Spec):
    name: str
    label: Text = ""  # human-readable name for the graph viewer; empty shows ``name``
    description: Text = ""
    domain: str | None = None  # a name from ``DatabaseSpec.domains``
    primary_keys: list[str] = []
    foreign_keys: list[ForeignKeySpec] = []
    columns: list[ColumnSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _keys_name_real_columns(self) -> TableSpec:
        names = [c.name for c in self.columns]
        if len(set(names)) != len(names):
            raise ValueError(f"table {self.name!r}: duplicate column names")
        referenced = set(self.primary_keys).union(
            *(fk.constrained_columns for fk in self.foreign_keys)
        )
        unknown = sorted(referenced - set(names))
        if unknown:
            raise ValueError(f"table {self.name!r}: keys name unknown column(s) {unknown}")
        return self


class SchemaSpec(_Spec):
    name: str
    tables: dict[str, TableSpec] = Field(min_length=1)


class DomainSpec(_Spec):
    """A business area grouping tables (e.g. "Berita & Event"), curated by the data owner."""

    name: str
    description: Text = ""


class ConceptSpec(_Spec):
    """A business term whose meaning is a rule over several columns: a ratio, a
    distinct count, a hidden filter. One column's meaning belongs in its description.

    ``sql`` is the aggregate expression and ``filters`` the WHERE conditions scoping
    every row; a condition on one part only (a ratio's numerator) goes inside ``sql``
    as ``FILTER (WHERE ...)``. Columns are ``schema.table.column``.
    """

    name: str
    synonyms: list[str] = Field([], max_length=MAX_ALIASES)  # more would be cut silently
    description: Text = ""
    sql: Text = ""
    filters: list[str] = []
    from_: Text = Field("", alias="from")  # FROM/JOIN clause; the join type is part of the rule
    columns: list[str] = Field(min_length=1)
    time_column: str | None = None

    def referenced_columns(self) -> list[str]:
        return list(dict.fromkeys([*self.columns, *filter(None, [self.time_column])]))

    def tables(self) -> list[str]:
        """``schema.table`` of every referenced column, in order of appearance."""
        return list(dict.fromkeys(c.rpartition(".")[0] for c in self.referenced_columns()))

    @model_validator(mode="after")
    def _defines_a_rule(self) -> ConceptSpec:
        if not (self.sql.strip() or self.filters):
            raise ValueError(f"concept {self.name!r}: needs sql or filters, else it is a column description")
        if len(self.tables()) > 1 and not self.from_.strip():
            raise ValueError(f"concept {self.name!r}: spans {self.tables()}, so `from` must say how they join")
        return self


class DatabaseSpec(_Spec):
    name: str
    description: Text = ""
    schemas: dict[str, SchemaSpec] = Field(min_length=1)
    domains: list[DomainSpec] = []
    business_concepts: list[ConceptSpec] = []

    def iter_tables(self) -> Iterator[tuple[SchemaSpec, TableSpec]]:
        for schema in self.schemas.values():
            for table in schema.tables.values():
                yield schema, table

    @model_validator(mode="after")
    def _names_unique_and_keys_resolve(self) -> DatabaseSpec:
        # Entity names are built from these names, so a duplicate would silently
        # fold two tables into one node.
        schema_names = [s.name for s in self.schemas.values()]
        if len(set(schema_names)) != len(schema_names):
            raise ValueError("duplicate schema names")
        by_key = {(s.name, t.name): t for s, t in self.iter_tables()}
        if len(by_key) != sum(len(s.tables) for s in self.schemas.values()):
            raise ValueError("duplicate table names within a schema")
        # A foreign key may point outside this file (kept, unresolved); one that
        # points inside it must name columns that table really has.
        for schema, table in self.iter_tables():
            for fk in table.foreign_keys:
                target = by_key.get(fk.target(schema.name))
                if target is None:
                    continue
                unknown = sorted(set(fk.referred_columns) - {c.name for c in target.columns})
                if unknown:
                    raise ValueError(
                        f"table {schema.name}.{table.name!r}: foreign key refers to "
                        f"unknown column(s) {unknown} of {'.'.join(fk.target(schema.name))}"
                    )
        self._domains_resolve()
        self._concepts_resolve()
        return self

    def _domains_resolve(self) -> None:
        names = [d.name for d in self.domains]
        if len(set(names)) != len(names):
            raise ValueError("duplicate domain names")
        for schema, table in self.iter_tables():
            if table.domain is not None and table.domain not in names:
                raise ValueError(f"table {schema.name}.{table.name!r}: unknown domain {table.domain!r}")

    def _concepts_resolve(self) -> None:
        # A concept naming a renamed column would hand the agent wrong SQL silently.
        names = [c.name for c in self.business_concepts]
        if len(set(names)) != len(names):
            raise ValueError("duplicate business concept names")
        # One synonym on two concepts makes the term ambiguous at retrieval.
        owner: dict[str, str] = {}
        for concept in self.business_concepts:
            for synonym in concept.synonyms:
                other = owner.setdefault(synonym.casefold(), concept.name)
                if other != concept.name:
                    raise ValueError(f"synonym {synonym!r} used by concepts {other!r} and {concept.name!r}")
        known = {f"{s.name}.{t.name}.{c.name}" for s, t in self.iter_tables() for c in t.columns}
        for concept in self.business_concepts:
            unknown = sorted(set(concept.referenced_columns()) - known)
            if unknown:
                raise ValueError(f"concept {concept.name!r}: unknown column(s) {unknown}")


def load_spec(source: DatabaseSpec | Mapping[str, Any] | str | Path) -> DatabaseSpec:
    """Validate a schema given as a parsed mapping or as a path to its YAML file.

    Raises:
        ValueError: the file is not valid YAML, or the content is not the schema
            format; the message names the offending field.
    """
    if isinstance(source, DatabaseSpec):
        return source
    try:
        if isinstance(source, (str, Path)):
            source = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
        return DatabaseSpec.model_validate(source)
    except (ValidationError, yaml.YAMLError) as e:
        raise ValueError(f"invalid database schema: {e}") from None
