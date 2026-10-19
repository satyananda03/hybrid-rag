"""Entity resolution at ingest (Fase 1 + 2): stop a duplicate before it is written.

Entity deduplication has three phases in three files. This one is the first line
of defence, and it runs **inside the ingest of one document**:

==========  =============================  =====================================
Phase       File                           When / what it looks at
==========  =============================  =====================================
Fase 1      ``entity_resolution`` (here)   Ingest. Rules only, no LLM: names of
                                           the current document vs the alias
                                           registries.
Fase 2      ``entity_adjudication``        Ingest, optional (``use_llm``). An LLM
                                           judges the pairs rules cannot decide.
Fase 3      ``entity_consolidation``       Later, on demand. Cleans duplicates
                                           that are *already stored*.
==========  =============================  =====================================

Rule of thumb: resolution **prevents** (the name is rewritten before any storage
is touched), consolidation **repairs** (two nodes already exist and get folded
into one). Resolution only ever sees one document at a time, so it cannot catch
two documents ingested at the same moment; consolidation exists for that gap.

What Fase 1 does: the names one document extracted are rewritten to a canonical
name before ``merge_nodes_and_edges`` touches any storage, so ``ICCN`` /
``Indonesia Creative Cities Networks (ICCN)`` and ``Kota Malang`` /
``Malang Kota`` / ``KOTA MALANG`` land on one node. Design and the evidence
behind every rule: docs/entity-deduplication/00-ringkasan-dan-rekomendasi.md §6.

Two registries:

- **YAML per domain** (``entity_registry/<name>.yaml``): curated aliases and
  ``never_merge`` pairs, written by people and chosen per document like an
  ontology (``insert_content_list(..., entity_registry="iccn")``).
- **Postgres per workspace** (``hybrid_rag_entity_alias``): every decision that
  was applied, keyed ``"<type>:<match_key>"``. Later documents resolve against
  it, and the rebuild after a deletion reads only it, so a name is decided once.

Merges are precision-first: only an equal ``match_key`` (same tokens in any
order, digits in their original order; institution names keep their order) or a
verified ``Long (ABBR)`` pattern merges, always within one entity type.

Fase 2 (``entity_adjudication.EntityAdjudicator``, optional per call): clusters
the registry has never seen are compared with the workspace's known entities and
with each other; an LLM decides the pairs rules cannot. ``EntityResolver`` stays
the single entry point for both phases.
"""

from __future__ import annotations
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable
import yaml
from hybrid_rag.core.constants import GRAPH_FIELD_SEP
from hybrid_rag.storage.postgres.connection import ClientManager
from hybrid_rag.storage.postgres.sql import SQL_TEMPLATES

if TYPE_CHECKING:
    from hybrid_rag.ingestion.knowledge_graph.entity_adjudication import EntityAdjudicator, Merge

logger = logging.getLogger(__name__)

REGISTRY_DIR = Path(__file__).parent / "entity_registry"
MAX_ALIASES = 10  # names kept on a node / in its VDB text; more only dilutes the embedding

# Sidecar/multimodal entities are named by the multimodal pipeline, which also writes their nodes and belongs_to edges itself: renaming them would orphan those.
_MULTIMODAL_TYPES = frozenset({"drawing", "table", "equation", "image"})
_STOPWORDS = frozenset(
    "dan di ke dari untuk yang atas dalam pada serta atau of and the for in on to".split()
)

# Formatting words that never change identity: "Badan Riset Dan Inovasi Nasional" is "Badan Riset Inovasi Nasional", "Perpres No 12 Tahun 2025" is "Perpres Nomor 12 Tahun 2025".
_CONNECTORS = frozenset("dan and the of no nomor tahun".split())

# An Indonesian institution's name is its identity, prefix and word order included: "Dinas Pariwisata, Ekonomi Kreatif, Kepemudaan Dan Olahraga" and "Dinas Kepemudaan, Olahraga, Pariwisata, Dan Ekonomi Kreatif" are offices of two different regions.
_INSTITUTION_PREFIXES = frozenset(
    "dinas suku badan kementerian kementrian direktorat lembaga balai biro kantor sekretariat "
    "inspektorat kepolisian komisi komite dewan majelis pemerintah pusat unit bidang bagian "
    "deputi satuan".split()
)
_POSSESSIVE = re.compile(r"['’]s\b")
_NON_WORD = re.compile(r"[\W_]+")
_LONG_ABBR = re.compile(r"^(?P<long>.+?)\s*\((?P<abbr>[^()\s]+)\)$")

# --- L0: keys ------------------------------------------------------------------
def entity_key(name: str) -> str:
    """Lookup key, never displayed: ``"Lampiran 1.2"`` -> ``"lampiran 1 2"``.

    Punctuation becomes a space instead of vanishing, so ``1.2`` and ``12`` stay
    apart; ``\\w`` keeps letters of every script, so ``東京`` is not ``""``.
    """
    text = unicodedata.normalize("NFKD", name)
    text = "".join(c for c in text if not unicodedata.combining(c)).casefold()
    return " ".join(_NON_WORD.sub(" ", _POSSESSIVE.sub("", text)).split())

def match_key(name: str) -> str:
    """Token bag of ``entity_key`` without connectors: words sorted, digits in order.

    ``Kota Malang`` == ``Malang Kota``, but ``Pasal 9 Ayat 1`` != ``Pasal 1 Ayat 9``.
    A name starting with an institution word (Dinas, Badan, ...) keeps its order:
    ``"=dinas pariwisata ..."``. ``""`` (nothing left) is never used to merge.
    """
    tokens = [t for t in entity_key(name).split() if t not in _CONNECTORS]
    if not tokens:
        return ""
    if tokens[0] in _INSTITUTION_PREFIXES:
        return "=" + " ".join(tokens)
    numbered = [t for t in tokens if any(c.isdigit() for c in t)]
    words = sorted(t for t in tokens if not any(c.isdigit() for c in t))
    return f"{' '.join(words)}|{' '.join(numbered)}"

def type_key(entity_type: str) -> str:
    return (entity_type or "").replace(" ", "").casefold()

def split_long_abbr(name: str) -> tuple[str, str] | None:
    """``"Badan Pusat Statistik (BPS)"`` -> ``("Badan Pusat Statistik", "BPS")``.

    The abbreviation must be the long form's initials (stopwords optional), so a
    qualifier such as ``Kota Malang (JATIM)`` or ``Foo (table)`` is not an alias.
    """
    m = _LONG_ABBR.match(name.strip())
    if not m:
        return None
    long, abbr = m["long"].strip(), m["abbr"]
    words = entity_key(long).split()
    letters = entity_key(abbr).replace(" ", "")
    if len(letters) < 2 or not any(c.isalpha() for c in letters):
        return None
    initials = {
        "".join(w[0] for w in words),
        "".join(w[0] for w in words if w not in _STOPWORDS),
    }
    return (long, abbr) if letters in initials else None

# --- YAML registry per domain --------------------------------------------------
@dataclass(frozen=True)
class DomainRegistry:
    """One ``entity_registry/<name>.yaml``, keyed for lookup."""
    name: str # (type_key or "*", match_key) -> canonical display name
    aliases: dict[tuple[str, str], str] # pairs of entity_key that must never share a node
    never_merge: frozenset[frozenset[str]]

def resolve_registry_path(name: str) -> Path:
    """``"iccn"`` or ``"iccn.yaml"`` -> a file in ``entity_registry/``; unknown names fail loudly."""
    if not name or Path(name).name != name:
        raise ValueError(f"invalid entity registry name {name!r}: expected a file name in {REGISTRY_DIR}")
    path = REGISTRY_DIR / (name if name.endswith((".yaml", ".yml")) else f"{name}.yaml")
    if path.is_file():
        return path
    available = sorted(p.stem for p in REGISTRY_DIR.glob("*.yaml"))
    raise FileNotFoundError(
        f"unknown entity registry {name!r}: not found in {REGISTRY_DIR}. available: {available or '(none)'}"
    )

def load_entity_registry(name: str) -> DomainRegistry:
    return _load_registry(resolve_registry_path(name))

@lru_cache(maxsize=16)
def _load_registry(path: Path) -> DomainRegistry:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    aliases: dict[tuple[str, str], str] = {}
    def add(scope: str, entries: list[dict[str, Any]] | None) -> None:
        for entry in entries or []:
            canonical = str(entry["canonical"]).strip()
            for alias in [canonical, *(entry.get("aliases") or [])]:
                key = match_key(str(alias))
                if not key:
                    continue
                prior = aliases.setdefault((scope, key), canonical)
                if prior != canonical:
                    raise ValueError(f"{path.name}: {alias!r} is an alias of both {prior!r} and {canonical!r}")

    add("*", raw.get("global"))
    for entity_type, entries in (raw.get("by_type") or {}).items():
        add(type_key(entity_type), entries)
    never_merge = frozenset(
        frozenset(entity_key(str(n)) for n in pair) for pair in raw.get("never_merge") or []
    )
    logger.info(f"entity registry {path.name}: {len(aliases)} alias keys, {len(never_merge)} never_merge pairs")
    return DomainRegistry(path.stem, aliases, never_merge)

_NO_DOMAIN = DomainRegistry("", {}, frozenset())

# --- Postgres registry per workspace -------------------------------------------
@dataclass(frozen=True)
class AliasDecision:
    alias_id: str  # "<type_key>:<match_key>"
    alias_name: str
    entity_type: str
    canonical: str
    source: str  # "user" (YAML) | "rule" | "llm"
    evidence: str
    confidence: float | None = None

class AliasRegistry:
    """``hybrid_rag_entity_alias``: applied decisions for one workspace.

    A ``rule`` row is never overwritten (first decision wins, so two documents
    racing on the same key both read back the winner); a ``user`` row is.
    """

    def __init__(self, workspace: str, role: str):
        self.workspace = workspace
        self.role = role
        self.db = None

    async def initialize(self) -> None:
        if self.db is None:
            self.db = await ClientManager.get_client(self.role)

    async def finalize(self) -> None:
        await ClientManager.release_client(self.db)
        self.db = None

    async def repoint(self, absorbed: str, survivor: str) -> None:
        """Consolidation merged ``absorbed`` into ``survivor``: its aliases follow."""
        await self.db.execute(
            SQL_TEMPLATES["repoint_entity_alias"],
            {"workspace": self.workspace, "absorbed": absorbed, "survivor": survivor},
        )

    async def known_entities(self) -> list[tuple[str, str]]:
        """(canonical name, type) of every entity this workspace has decided on."""
        # ponytail: whole-workspace read per document; fine for thousands of entities, page or cache it if a workspace grows past ~100k
        rows = await self.db.query(
            SQL_TEMPLATES["known_entity_aliases"], [self.workspace], multirows=True
        )
        return [(row["canonical_entity_id"], row["entity_type"]) for row in rows or []]

    async def get(self, alias_ids: Iterable[str]) -> dict[str, str]:
        ids = sorted(set(alias_ids))
        if not ids:
            return {}
        rows = await self.db.query(
            SQL_TEMPLATES["get_entity_aliases"], [self.workspace, ids], multirows=True
        )
        return {row["id"]: row["canonical_entity_id"] for row in rows or []}

    async def add(self, decisions: Iterable[AliasDecision]) -> None:
        values = [
            (self.workspace, d.alias_id, d.alias_id.split(":", 1)[1], d.alias_name,
             d.entity_type, d.canonical, d.source, d.evidence, d.confidence)
            for d in decisions
        ]
        if not values:
            return

        async def _upsert(connection) -> None:
            await connection.executemany(SQL_TEMPLATES["upsert_entity_alias"], values)

        await self.db._run_with_retry(_upsert)


# --- Resolver --------------------------------------------------------------------


def rewrite(
    nodes: dict[str, list[dict]],
    edges: dict[tuple[str, str], list[dict]],
    mapping: dict[str, str],
) -> tuple[dict[str, list[dict]], dict[tuple[str, str], list[dict]]]:
    """Move records onto canonical names; edges become sorted pairs, new self-loops drop."""
    if not mapping:
        return nodes, edges
    new_nodes: dict[str, list[dict]] = defaultdict(list)
    for name, records in nodes.items():
        canonical = mapping.get(name, name)
        renamed = {"raw_name": name} if canonical != name else {}
        new_nodes[canonical].extend({**r, "entity_name": canonical, **renamed} for r in records)
    new_edges: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for (src, tgt), records in edges.items():
        src, tgt = mapping.get(src, src), mapping.get(tgt, tgt)
        if src == tgt:
            logger.debug(f"entity resolution: dropping self-relation on {src!r} after merge")
            continue
        new_edges[tuple(sorted((src, tgt)))].extend(
            {**r, "src_id": mapping.get(r["src_id"], r["src_id"]),
             "tgt_id": mapping.get(r["tgt_id"], r["tgt_id"])}
            for r in records
        )
    return dict(new_nodes), dict(new_edges)


def _resolvable_names(node_lists: Iterable[dict[str, list[dict]]]) -> dict[str, tuple[str, str]]:
    """name -> (type_key, match_key) for every name the resolver may rename.

    The type is the most common one across the name's records (ties: alphabetical),
    the same choice the graph merge makes for the node.
    """
    types: dict[str, Counter] = defaultdict(Counter)
    for nodes in node_lists:
        for name, records in nodes.items():
            types[name].update(type_key(r.get("entity_type", "")) for r in records)
    out = {}
    for name, counts in types.items():
        etype = max(sorted(counts), key=counts.__getitem__)
        key = match_key(name)
        if key and etype and etype not in _MULTIMODAL_TYPES:
            out[name] = (etype, key)
    return out


def _alias_id(etype: str, key: str) -> str:
    return f"{etype}:{key}"


@dataclass
class _Cluster:
    """Fase 1 plan for one group of names: who it is and whether the registry knew it."""

    center: tuple[str, str]
    members: list[tuple[str, str]]  # group keys, center included
    names: dict[tuple[str, str], list[str]]  # group key -> its names (extracted or implied)
    extracted: frozenset[tuple[str, str]]  # keys some extracted name has; the rest are implied
    canonical: str
    is_new: bool  # no registry decision yet: Fase 2 may merge it


class EntityResolver:
    """Rewrites one document's extracted names to canonical names (see module doc).

    The only object callers use: ``resolve`` at ingest, ``canonical_names`` at
    rebuild. ``adjudicator`` (Fase 2) is optional and only runs with ``use_llm``.
    """

    def __init__(self, registry: AliasRegistry, adjudicator: EntityAdjudicator | None = None):
        self.registry = registry
        self.adjudicator = adjudicator

    async def resolve(
        self,
        nodes: dict[str, list[dict]],
        edges: dict[tuple[str, str], list[dict]],
        registry_name: str | None = None,
        use_llm: bool = False,
    ) -> tuple[dict[str, list[dict]], dict[tuple[str, str], list[dict]]]:
        """Ingest path: YAML aliases, rules, optional LLM; record every decision, then apply."""
        domain = load_entity_registry(registry_name) if registry_name else _NO_DOMAIN
        names = _resolvable_names([nodes])
        if not names:
            return nodes, edges

        user_decisions, clusters = await self._plan_clusters(names, domain)
        merges: dict[tuple[str, str], Merge] = {}
        if use_llm and self.adjudicator is not None:
            merges = await self._adjudicate(clusters, user_decisions, nodes, domain)

        # Postgres decides: rule/llm rows never overwrite, so a concurrent document
        # that claimed a key first wins and both documents read back the same name.
        # ponytail: Fase 2 widens that window to the LLM call; a cluster can split
        # across two canonicals then — Fase 3 merges existing duplicates.
        await self.registry.add(self._emit_decisions(user_decisions, clusters, merges, domain))
        return await self._apply(nodes, edges, names, domain)

    async def _plan_clusters(
        self, names: dict[str, tuple[str, str]], domain: DomainRegistry
    ) -> tuple[list[AliasDecision], list[_Cluster]]:
        """Fase 1: YAML aliases, then groups by (type, match_key) joined by Long (ABBR)."""
        # 1. YAML aliases win over everything, including earlier rule decisions.
        user_decisions: list[AliasDecision] = []
        rest: dict[str, tuple[str, str]] = {}
        for name, (etype, key) in names.items():
            canonical = domain.aliases.get((etype, key)) or domain.aliases.get(("*", key))
            if canonical and not self._vetoed(domain, name, canonical):
                user_decisions.append(AliasDecision(
                    _alias_id(etype, key), name, etype, canonical, "user", f"registry:{domain.name}"
                ))
            else:
                rest[name] = (etype, key)

        # 2. Rules: one group per (type, match_key), joined into clusters by Long (ABBR).
        groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for name, gk in sorted(rest.items()):
            groups[gk].append(name)
        # Keys a "X (A)" implies but no extracted name has: "X" and "A" of a later
        # document must find this cluster too.
        centers, implied = self._link_long_abbr(groups)

        members_of: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
        for gk in sorted({*groups, *implied}):
            members_of[centers.get(gk, gk)].append(gk)

        stored = await self.registry.get(_alias_id(*gk) for gk in [*groups, *implied])
        clusters = []
        for center, members in members_of.items():
            hits = [stored[_alias_id(*gk)] for gk in members if _alias_id(*gk) in stored]
            if hits:
                canonical = stored.get(_alias_id(*center)) or sorted(hits)[0]
            else:
                # Longest wins; at equal length "Kota Malang" beats "KOTA MALANG".
                candidates = sorted(groups.get(center) or implied[center])
                canonical = max(candidates, key=lambda n: (len(n), not n.isupper()))
            cluster_names = {gk: groups.get(gk) or implied[gk] for gk in members}
            extracted = frozenset(gk for gk in members if gk in groups)
            clusters.append(_Cluster(center, members, cluster_names, extracted, canonical, not hits))
        return user_decisions, clusters

    async def _adjudicate(
        self,
        clusters: list[_Cluster],
        user_decisions: list[AliasDecision],
        nodes: dict[str, list[dict]],
        domain: DomainRegistry,
    ) -> dict[tuple[str, str], Merge]:
        """Fase 2 for the clusters the registry has never seen."""
        from hybrid_rag.ingestion.knowledge_graph.entity_adjudication import DESCRIPTION_CHARS, Profile

        def description(cluster_names: Iterable[str]) -> str:
            for name in cluster_names:
                for record in nodes.get(name, ()):
                    if record.get("description"):
                        return record["description"][:DESCRIPTION_CHARS]
            return ""

        new = {}
        for c in clusters:
            if c.is_new:
                all_names = [n for gk in c.members for n in c.names[gk]]
                new[_alias_id(*c.center)] = Profile(
                    c.canonical, c.center[0], description(all_names), tuple(all_names)
                )
        # Names this document already fixed by YAML alias: targets, never merged away.
        by_canonical: dict[str, list[AliasDecision]] = defaultdict(list)
        for d in user_decisions:
            by_canonical[d.canonical].append(d)
        local_targets = [
            Profile(canonical, ds[0].entity_type, description(d.alias_name for d in ds))
            for canonical, ds in by_canonical.items()
        ]
        merges = await self.adjudicator.adjudicate(
            new, await self.registry.known_entities(), local_targets, domain
        )
        return {c.center: merges[_alias_id(*c.center)] for c in clusters if _alias_id(*c.center) in merges}

    def _emit_decisions(
        self,
        user_decisions: list[AliasDecision],
        clusters: list[_Cluster],
        merges: dict[tuple[str, str], Merge],
        domain: DomainRegistry,
    ) -> list[AliasDecision]:
        """One decision per key: rule canonical, or the Fase 2 target that overrides it."""
        decisions = list(user_decisions)
        for c in clusters:
            merge = merges.get(c.center)
            canonical = merge.target if merge else c.canonical
            evidence = "long_abbr" if len(c.members) > 1 else "match_key"
            for gk in c.members:
                extracted = gk in c.extracted
                if merge:
                    source, confidence = "llm", merge.confidence
                    # "_implied": a key no extracted name had, so not a node merged away
                    tag = f"llm: {merge.reason[:200]}" if extracted else "llm_implied"
                else:
                    source, confidence = "rule", None
                    tag = evidence if extracted else f"{evidence}_implied"
                for name in c.names[gk]:
                    if not self._vetoed(domain, name, canonical):
                        decisions.append(AliasDecision(
                            _alias_id(*gk), name, gk[0], canonical, source, tag, confidence
                        ))
        return self._one_per_id(decisions)

    async def _apply(
        self,
        nodes: dict[str, list[dict]],
        edges: dict[tuple[str, str], list[dict]],
        names: dict[str, tuple[str, str]],
        domain: DomainRegistry,
    ) -> tuple[dict[str, list[dict]], dict[tuple[str, str], list[dict]]]:
        """Read back what the registry holds now and rewrite the document onto it."""
        stored = await self.registry.get(_alias_id(*gk) for gk in names.values())
        mapping = {}
        for name, gk in names.items():
            canonical = stored.get(_alias_id(*gk), name)
            if canonical != name and not self._vetoed(domain, name, canonical):
                mapping[name] = canonical

        if mapping:
            logger.info(
                f"entity resolution: {len(mapping)} names rewritten, "
                f"{len(nodes)} -> {len(set(nodes) - set(mapping) | set(mapping.values()))} entities"
            )
        return rewrite(nodes, edges, mapping)

    async def canonical_names(self, node_lists: Iterable[dict[str, list[dict]]]) -> dict[str, str]:
        """Rebuild path: registry lookup only — no YAML, no new rule, no write."""
        names = _resolvable_names(node_lists)
        stored = await self.registry.get(_alias_id(*gk) for gk in names.values())
        mapping = {}
        for name, gk in names.items():
            canonical = stored.get(_alias_id(*gk), name)
            if canonical != name:
                mapping[name] = canonical
        return mapping

    # --- aliases on the canonical node ------------------------------------------

    @staticmethod
    def merged_aliases(canonical: str, existing: str | None, records: Iterable[dict]) -> str:
        """Names merged into ``canonical`` (``GRAPH_FIELD_SEP``-joined), for the node and its VDB text.

        ``existing`` is the node's stored ``aliases``; ``records`` carry ``raw_name``
        when ``rewrite`` renamed them. Only names whose words differ are kept:
        ``KOTA MALANG`` adds nothing to ``Kota Malang``, ``ICCN`` does.
        """
        names = [n for n in (existing or "").split(GRAPH_FIELD_SEP) if n]
        for r in records:  # consolidation passes a node's name and its aliases at once
            names += [n for n in (r.get("raw_name") or "").split(GRAPH_FIELD_SEP) if n]
        seen, kept = {entity_key(canonical)}, []
        for name in sorted(set(names)):
            key = entity_key(name)
            if key and key not in seen:
                seen.add(key)
                kept.append(name)
        return GRAPH_FIELD_SEP.join(kept[:MAX_ALIASES])

    @staticmethod
    def vdb_content(name: str, description: str, aliases: str | None = None) -> str:
        """Entity VDB text: unchanged without aliases; with them, aliases come before the
        description so truncation (which cuts the end) never drops them."""
        if not aliases:
            return f"{name}\n{description}"
        return f"{name}\nAlias: {', '.join(aliases.split(GRAPH_FIELD_SEP))}\n{description}"

    @staticmethod
    def _vetoed(domain: DomainRegistry, name: str, canonical: str) -> bool:
        return frozenset((entity_key(name), entity_key(canonical))) in domain.never_merge

    @staticmethod
    def _link_long_abbr(groups: dict[tuple[str, str], list[str]]):
        """Star clusters around a long form: ``X (A)`` and ``A`` join ``X``.

        Returns (group -> center, implied group -> display names), where an
        implied group is a key the pattern names but no extracted entity has.
        An abbreviation naming two different long forms in one document is
        ambiguous and joins neither; ``X (A)`` still joins ``X``.
        """
        pairs = []  # (full group, long group, abbr group, long display, abbr display)
        abbr_targets: dict[tuple[str, str], set] = defaultdict(set)
        for gk, members in sorted(groups.items()):
            for name in members:
                split = split_long_abbr(name)
                if split:
                    long_gk, abbr_gk = (gk[0], match_key(split[0])), (gk[0], match_key(split[1]))
                    pairs.append((gk, long_gk, abbr_gk, *split))
                    abbr_targets[abbr_gk].add(long_gk)

        centers: dict[tuple[str, str], tuple[str, str]] = {}
        implied: dict[tuple[str, str], list[str]] = defaultdict(list)
        hubs: set[tuple[str, str]] = set()
        for full_gk, long_gk, abbr_gk, long_name, abbr_name in pairs:
            if long_gk in centers:  # the long form already joined another cluster: no chains
                continue
            hubs.add(long_gk)
            if long_gk not in groups:
                implied[long_gk].append(long_name)
            joiners = [(full_gk, None)]
            if len(abbr_targets[abbr_gk]) == 1:
                joiners.append((abbr_gk, abbr_name))
            for gk, display in joiners:
                if gk in hubs or centers.get(gk, long_gk) != long_gk:
                    continue
                centers[gk] = long_gk
                if gk not in groups:
                    implied[gk].append(display)
        return centers, dict(implied)

    @staticmethod
    def _one_per_id(decisions: Iterable[AliasDecision]) -> list[AliasDecision]:
        # Variants sharing a match_key share one row: user first, then deterministic order.
        by_id: dict[str, AliasDecision] = {}
        for d in sorted(decisions, key=lambda d: (d.source != "user", d.alias_id, d.alias_name)):
            by_id.setdefault(d.alias_id, d)
        return list(by_id.values())
