"""Ontology-driven extraction: an OWL/Turtle file -> neo4j-graphrag ``GraphSchema``.

Replicates how neo4j-graphrag's ``SimpleKGPipeline`` uses a user-supplied schema:

1. the ontology becomes a ``GraphSchema`` (``get_schema_from_onto``, a 1:1 port of
   ``ontology-driven-kg/ontology_to_schema.py``);
2. the schema is shown to the LLM as ``schema.model_dump(exclude_none=True)``, the
   same text ``ERExtractionTemplate`` puts under its ``{schema}`` placeholder
   (``render_schema_for_prompt``);
3. each chunk's extraction is validated by neo4j-graphrag's own ``GraphPruning``
   (``prune_chunk``) — node types, relationship types, (source, rel, target)
   patterns with direction repair, and properties.

The port keeps the original conversion's limits on purpose: ``rdfs:subClassOf`` is
ignored, an object property without domain/range yields no pattern (so its
relations are pruned), ``owl:unionOf`` crashes ``get_local_part``, and every
property is ``STRING``. See ontology/regulasi_pemdi.ttl for an ontology written
within those limits.

neo4j-graphrag and rdflib are imported inside the functions: the default
(ontology-less) extraction never loads them.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neo4j_graphrag.components.schema import GraphSchema

logger = logging.getLogger(__name__)

ONTOLOGY_DIR = Path(__file__).resolve().parent / "ontology"
_EXTENSIONS = (".ttl", ".owl")


# --- 1. OWL -> GraphSchema (port of ontology_to_schema.py) --------------------


def get_local_part(uri):
    pos = uri.rfind("#")
    if pos < 0:
        pos = uri.rfind("/")
    if pos < 0:
        pos = uri.rindex(":")
    return uri[pos + 1 :]


def get_properties_for_class(g, cat):
    from neo4j_graphrag.components.schema import PropertyType
    from rdflib.namespace import OWL, RDF, RDFS

    props = []
    for dtp in g.subjects(RDFS.domain, cat):
        if (dtp, RDF.type, OWL.DatatypeProperty) in g:
            prop_name = get_local_part(dtp)
            prop_desc = next(g.objects(dtp, RDFS.comment), "")
            props.append(PropertyType(name=prop_name, type="STRING", description=str(prop_desc)))
    return props


def _node_type(g, cat):
    from neo4j_graphrag.components.schema import NodeType
    from rdflib.namespace import RDFS

    node = {"label": get_local_part(cat), "description": str(next(g.objects(cat, RDFS.comment), ""))}
    props = get_properties_for_class(g, cat)
    # neo4j-graphrag >= 1.13 rejects ``properties=[]``; without the key a class
    # with no DatatypeProperty gets the default ``name: STRING`` property.
    if props:
        node["properties"] = props
    return NodeType.model_validate(node)


def get_schema_from_onto(g) -> GraphSchema:
    from neo4j_graphrag.components.schema import RelationshipType, SchemaBuilder
    from rdflib.namespace import OWL, RDF, RDFS

    classes = {}
    entities = []
    rels = []
    triples = []

    for cat in g.subjects(RDF.type, OWL.Class):
        classes[cat] = None
        entities.append(_node_type(g, cat))

    for cat in g.objects(None, RDFS.domain):
        if cat not in classes.keys():
            classes[cat] = None
            entities.append(_node_type(g, cat))

    for cat in g.objects(None, RDFS.range):
        if not (cat.startswith("http://www.w3.org/2001/XMLSchema#") or cat in classes.keys()):
            classes[cat] = None
            entities.append(_node_type(g, cat))

    for op in g.subjects(RDF.type, OWL.ObjectProperty):
        relname = get_local_part(op)
        rels.append(RelationshipType(label=relname, description=str(next(g.objects(op, RDFS.comment), ""))))

    for op in g.subjects(RDF.type, OWL.ObjectProperty):
        relname = get_local_part(op)
        doms = [get_local_part(dom) for dom in g.objects(op, RDFS.domain) if dom in classes.keys()]
        rans = [get_local_part(ran) for ran in g.objects(op, RDFS.range) if ran in classes.keys()]
        for d in doms:
            for r in rans:
                triples.append((d, relname, r))

    return SchemaBuilder.create_schema_model(node_types=entities, relationship_types=rels, patterns=triples)


# --- 2. Locating and loading an ontology file ----------------------------------


def resolve_ontology_path(name: str) -> Path:
    """``"regulasi_pemdi"`` or ``"regulasi_pemdi.ttl"`` -> a file in ``ontology/``.

    The name arrives from the ingest caller, so it may only name a file directly
    in ONTOLOGY_DIR. An unknown name fails loudly, like an unknown domain does: a
    document extracted under the wrong contract can only be fixed by reingesting it.
    """
    if not name or Path(name).name != name:
        raise ValueError(f"invalid ontology name {name!r}: expected a file name in {ONTOLOGY_DIR}")
    candidates = [name] if Path(name).suffix in _EXTENSIONS else [f"{name}{ext}" for ext in _EXTENSIONS]
    for candidate in candidates:
        path = ONTOLOGY_DIR / candidate
        if path.is_file():
            return path
    available = sorted(p.name for p in ONTOLOGY_DIR.glob("*") if p.suffix in _EXTENSIONS)
    raise FileNotFoundError(
        f"unknown ontology {name!r}: not found in {ONTOLOGY_DIR}. available: {available or '(none)'}"
    )


def load_ontology_schema(name: str) -> GraphSchema:
    """Parse and convert one ontology, once per file."""
    return _load_schema(resolve_ontology_path(name))


@lru_cache(maxsize=16)
def _load_schema(path: Path) -> GraphSchema:
    from rdflib import Graph

    try:
        schema = get_schema_from_onto(Graph().parse(path))
    except Exception as e:
        # e.g. a Turtle syntax error, or "substring not found" from get_local_part
        # on an owl:unionOf blank node: name the file so the caller can fix it
        raise ValueError(f"cannot load ontology {path.name}: {type(e).__name__}: {e}") from e
    logger.info(
        f"ontology {path.name}: {len(schema.node_types)} node types, "
        f"{len(schema.relationship_types)} relationship types, {len(schema.patterns)} patterns"
    )
    return schema


# --- 3. Guidance: the schema as the extraction prompt shows it -----------------


def render_schema_for_prompt(schema: GraphSchema) -> str:
    """What neo4j-graphrag's ``ERExtractionTemplate`` substitutes for ``{schema}``,
    with braces doubled because it is spliced into a ChatPromptTemplate."""
    return str(schema.model_dump(exclude_none=True)).replace("{", "{{").replace("}", "}}")


# --- 4. Validation: neo4j-graphrag's GraphPruning over one chunk ---------------

# GraphPruning passes nodes labelled with the lexical-graph labels ("Document",
# "Chunk") through unvalidated. An ontology may well define a Document class, so
# point the lexical config at labels an LLM cannot emit.
_NO_LEXICAL = dict(
    document_node_label="__HybridRagNoDocument__",
    chunk_node_label="__HybridRagNoChunk__",
    chunk_to_document_relationship_type="__HYBRID_RAG_NONE_1__",
    next_chunk_relationship_type="__HYBRID_RAG_NONE_2__",
    node_to_chunk_relationship_type="__HYBRID_RAG_NONE_3__",
)


async def prune_chunk(
    maybe_nodes: dict[str, list[dict]],
    maybe_edges: dict[tuple[str, str], list[dict]],
    schema: GraphSchema,
    chunk_key: str,
) -> tuple[dict[str, list[dict]], dict[tuple[str, str], list[dict]]]:
    """Run GraphPruning over one chunk's parsed extraction, in the KG's own shape.

    Each entity record becomes a ``Neo4jNode`` labelled with its type and carrying
    ``name`` plus its properties, as neo4j-graphrag's prompt has the model write
    them; each relationship record becomes a ``Neo4jRelationship`` whose endpoints
    are the first record of each name. A relationship whose endpoint was not
    extracted in this chunk is dropped (``INVALID_START_OR_END_NODE``), as in
    neo4j-graphrag, rather than created as an UNKNOWN node by the merge.
    """
    from neo4j_graphrag.components.graph_pruning import GraphPruning
    from neo4j_graphrag.components.types import (
        LexicalGraphConfig,
        Neo4jGraph,
        Neo4jNode,
        Neo4jRelationship,
    )

    node_records: dict[str, dict] = {}
    first_id: dict[str, str] = {}
    nodes = []
    for name, records in maybe_nodes.items():
        for i, rec in enumerate(records):
            node_id = f"{i}:{name}"
            first_id.setdefault(name, node_id)
            node_records[node_id] = rec
            nodes.append(
                Neo4jNode(
                    id=node_id,
                    label=rec["entity_type"],
                    properties={"name": name, **rec.get("properties", {})},
                )
            )

    rel_records: list[dict] = []
    rels = []
    for (src, tgt), records in maybe_edges.items():
        for rec in records:
            rel_records.append(rec)
            rels.append(
                Neo4jRelationship(
                    # an endpoint never extracted gets an id no node has
                    start_node_id=first_id.get(src, f"missing:{src}"),
                    end_node_id=first_id.get(tgt, f"missing:{tgt}"),
                    type=rec.get("relation_type", ""),
                    properties=rec.get("properties", {}),
                )
            )

    result = await GraphPruning().run(
        graph=Neo4jGraph(nodes=nodes, relationships=rels),
        schema=schema,
        lexical_graph_config=LexicalGraphConfig(**_NO_LEXICAL),
    )

    kept_nodes: dict[str, list[dict]] = defaultdict(list)
    for node in result.graph.nodes:
        rec = node_records[node.id]
        props = {k: v for k, v in node.properties.items() if k != "name"}
        kept_nodes[rec["entity_name"]].append({**rec, "properties": props})

    # GraphPruning keeps relationships in input order and carries no id, so pair
    # its output back with the input records in order. Two inputs with the same
    # type and endpoints are validated identically, so this pairing is exact.
    kept_edges: dict[tuple[str, str], list[dict]] = defaultdict(list)
    pruned = result.graph.relationships
    j = 0
    for rel, rec in zip(rels, rel_records):
        if j == len(pruned):
            break
        out = pruned[j]
        if out.type != rel.type or {out.start_node_id, out.end_node_id} != {rel.start_node_id, rel.end_node_id}:
            continue
        j += 1
        src = node_records[out.start_node_id]["entity_name"]
        tgt = node_records[out.end_node_id]["entity_name"]
        kept_edges[(src, tgt)].append({**rec, "src_id": src, "tgt_id": tgt, "properties": dict(out.properties)})
    assert j == len(pruned), f"{chunk_key}: could not pair pruned relationships with their records"

    stats = result.pruning_stats
    if stats.number_of_pruned_nodes or stats.number_of_pruned_relationships or stats.number_of_pruned_properties:
        reasons = Counter(
            f"{kind}:{item.pruned_reason.value}"
            for kind, items in (
                ("node", stats.pruned_nodes),
                ("relationship", stats.pruned_relationships),
                ("property", stats.pruned_properties),
            )
            for item in items
        )
        logger.info(f"{chunk_key}: ontology pruning {dict(reasons)}")
        for item in stats.pruned_relationships:
            logger.debug(f"{chunk_key}: pruned {item.label} {item.pruned_reason.value}: {item.item}")

    return dict(kept_nodes), dict(kept_edges)
