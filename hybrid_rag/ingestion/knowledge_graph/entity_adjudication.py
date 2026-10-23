"""Entity resolution Fase 2: candidates → vetoes → LLM adjudication → star merges.

Fase 1 (``entity_resolution``) only merges names whose keys are equal. What it
misses — qualifiers (``Dewan Pakar`` / ``Dewan Pakar ICCN``), typos
(``Cang`` / ``Chang``), acronyms (``Menpora``) — has the same shape as hard
negatives (``Ekonomi Kreatif`` / ``Ekosistem Ekonomi Kreatif``), so a rule cannot
decide them. Here rules only *propose* pairs and veto the ones that are provably
different; an LLM that reads both descriptions decides the rest. Design and
evidence: docs/entity-deduplication/00-ringkasan-dan-rekomendasi.md §6.2 L3–L6.

``EntityAdjudicator.adjudicate`` is the only entry point; ``EntityResolver``
calls it for the clusters of one document that the registry has never seen.
Everything it reads comes from the workspace's own graph and entity VDB.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, ValidationError, model_validator

from hybrid_rag import progress
from hybrid_rag.config import settings
from hybrid_rag.core.prompts import load_prompt
from hybrid_rag.storage.postgres.sql import SQL_TEMPLATES
from hybrid_rag.ingestion.knowledge_graph.entity_resolution import (
    _CONNECTORS,
    _INSTITUTION_PREFIXES,
    _STOPWORDS,
    DomainRegistry,
    entity_key,
    type_key,
)
from hybrid_rag.ingestion.utils import INVALID_OUTPUT_ERRORS

logger = logging.getLogger(__name__)

PROMPTS_PATH = Path(__file__).parent / "entity_resolution.yaml"

FUZZY_MIN = 90  # token_sort_ratio for spelling variants
EMBED_MIN_SIMILARITY = 0.70  # cosine; calibrated on the benchmark (evaluation/entity_dedup_benchmark.py)
PAIRS_PER_CLUSTER = 5
PAIRS_PER_DOCUMENT = 300
PAIRS_PER_CALL = 40
# Calibrated on the benchmark: every false merge the LLM made had confidence <= 0.90.
MIN_CONFIDENCE = 0.90
DESCRIPTION_CHARS = 300

# Cross-type pairs the extractor mixes up (10 of 164 benchmark positives); any
# other type difference is a veto.
_TYPE_GROUPS = (frozenset({"concept", "program"}), frozenset({"organization", "role"}))
_ADMIN_TOKENS = frozenset("kota kabupaten kab provinsi prov kecamatan desa kelurahan kotamadya".split())
_REGIONAL_INSTITUTIONS = frozenset("dinas suku bappeda dprd pemerintah".split())


@dataclass(frozen=True)
class Profile:
    """One entity as the vetoes and the LLM see it."""

    name: str
    etype: str
    description: str = ""
    aliases: tuple[str, ...] = ()  # other names of a new cluster, for never_merge


@dataclass(frozen=True)
class Merge:
    """A new cluster folds into ``target`` (an existing name or another new cluster's)."""

    target: str
    confidence: float
    reason: str


@dataclass
class _Pair:
    source: str  # new cluster key
    other: str  # new cluster key or existing entity name
    other_is_new: bool
    signal: str
    score: float
    pair_id: str = ""


# --- LLM contract ---------------------------------------------------------------


class PairVerdict(BaseModel):
    pair_id: str = Field(description="The id of the pair being judged, copied exactly.")
    same: bool = Field(description="True only when a and b are the same real-world entity.")
    confidence: float = Field(ge=0.0, le=1.0, description="Probability that `same` is right.")
    preferred: Literal["a", "b"] = Field(default="a", description="Side with the better canonical name.")
    reason: str = Field(default="", description="One short sentence in Bahasa Indonesia.")


class AdjudicationOutput(BaseModel):
    verdicts: list[PairVerdict] = Field(description="One verdict per pair.")

    @model_validator(mode="before")
    @classmethod
    def _drop_invalid_verdicts(cls, data):
        # A broken record costs that pair, not the whole batch.
        if isinstance(data, dict) and isinstance(data.get("verdicts"), list):
            kept = []
            for record in data["verdicts"]:
                try:
                    kept.append(PairVerdict.model_validate(record))
                except ValidationError as e:
                    logger.warning(f"entity adjudication: dropping verdict {record!r}: {e.errors()[0]['msg']}")
            data = {**data, "verdicts": kept}
        return data


@cache
def _adjudication_chain() -> Runnable:
    from hybrid_rag.ingestion.knowledge_graph.chain import _extract_llm

    prompt = load_prompt("entity_adjudication", PROMPTS_PATH)
    template = ChatPromptTemplate.from_messages(
        [("system", prompt["system_prompt"]), ("human", prompt["user_prompt"])]
    )
    return template | _extract_llm(AdjudicationOutput)


# One gate for every document and workspace: adjudication shares the extraction model.
_llm_gate = asyncio.Semaphore(max(1, settings.extract_llm_max_async))


class AdjudicationFailed(Exception):
    """The LLM could not judge a batch; the document keeps its Fase 1 result."""


# --- verdicts already paid for ----------------------------------------------------


class PairVerdictStore:
    """``hybrid_rag_entity_pair_verdict``: every LLM verdict of one workspace.

    Consolidation re-reads the whole graph each run; with this, a pair is judged
    once. Verdicts below the merge threshold stay here for a later human review.
    """

    def __init__(self, workspace: str, db):
        self.workspace = workspace
        self.db = db

    @staticmethod
    def pair_id(a: Profile, b: Profile) -> str:
        sides = sorted((f"{a.etype}:{a.name}", f"{b.etype}:{b.name}"))
        return hashlib.sha1("\x1f".join(sides).encode()).hexdigest()

    async def get(self, ids: Iterable[str]) -> dict[str, dict]:
        ids = sorted(set(ids))
        if not ids:
            return {}
        rows = await self.db.query(SQL_TEMPLATES["get_pair_verdicts"], [self.workspace, ids], multirows=True)
        return {row["id"]: dict(row) for row in rows or []}

    async def put(self, rows: list[tuple[str, str, str, bool, float, str, str]]) -> None:
        """Rows of (id, name_a, name_b, same, confidence, preferred name, reason)."""
        if not rows:
            return
        values = [(self.workspace, *row) for row in rows]

        async def _upsert(connection) -> None:
            await connection.executemany(SQL_TEMPLATES["upsert_pair_verdict"], values)

        await self.db._run_with_retry(_upsert)


# --- Adjudicator ------------------------------------------------------------------


@dataclass
class EntityAdjudicator:
    """Fase 2 for one workspace: ``graph`` and ``entity_vdb`` are that workspace's storages."""

    graph: object
    entity_vdb: object = None
    chain: Runnable | None = None  # default: the EXTRACT_* model; tests pass a fake
    verdicts: PairVerdictStore | None = None  # consolidation only: reuse earlier verdicts
    max_pairs: int = PAIRS_PER_DOCUMENT
    calls: int = field(default=0, init=False)  # LLM calls made, for reports
    judged: int = field(default=0, init=False)  # pairs judged in the last adjudicate()
    cached: int = field(default=0, init=False)  # of those, answered from ``verdicts``

    async def adjudicate(
        self,
        new: dict[str, Profile],
        known: Iterable[tuple[str, str]],
        local_targets: Iterable[Profile] = (),
        domain: DomainRegistry | None = None,
        vectors: dict[str, np.ndarray] | None = None,
        rank: dict[str, tuple] | None = None,
    ) -> dict[str, Merge]:
        """Which new clusters merge into what. Keys of ``new`` are cluster keys.

        ``known`` is every (name, type) the workspace registry has decided on;
        ``local_targets`` are names this document already fixed (YAML aliases).
        Both are merge targets only, never merged away. An LLM failure returns
        ``{}`` (Fase 1 result kept) and is logged.

        Consolidation passes ``vectors`` (stored embeddings of ``new``, so nothing
        is embedded) and ``rank`` (between two of ``new``, the higher rank survives).
        """
        self.judged = self.cached = 0
        if not new:
            return {}
        targets = {p.name: p for p in local_targets}
        known_types = {name: etype for name, etype in known if name not in targets}
        pairs = self._propose_lexical(new, {**known_types, **{n: p.etype for n, p in targets.items()}})
        pairs += await self._propose_embedding(new, vectors)

        existing = {p.other for p in pairs if not p.other_is_new and p.other not in targets}
        targets.update(await self._existing_profiles(existing))
        pairs = self._shortlist(pairs, new, targets, domain)
        if not pairs:
            return {}
        self.judged = len(pairs)

        try:
            verdicts = await self._verdicts(pairs, new, targets)
        except AdjudicationFailed as e:
            logger.warning(f"entity adjudication skipped for this document: {e}")
            progress.activity(
                "Entity resolution Fase 2 dilewati untuk dokumen ini: LLM tidak bisa menilai", level="warning"
            )
            return {}
        return self._select(pairs, verdicts, new, rank)

    # --- L3: candidates ---------------------------------------------------------

    def _propose_lexical(self, new: dict[str, Profile], known: dict[str, str]) -> list[_Pair]:
        """Containment, spelling variants, acronyms and admin-prefixed region names, against new and known names.

        One ``cdist`` per compatible-type group (C, not Python loops); acronyms are
        only tried against names that share the first letter.
        """
        from rapidfuzz import fuzz, process

        pool = [(key, p.name, p.etype, True) for key, p in new.items()]
        pool += [(name, name, etype, False) for name, etype in known.items()]
        plain = [_plain(o[1]) for o in pool]
        pairs: list[_Pair] = []

        def add(source_key: str, i: int, signal: str, score: float) -> None:
            key, name, _, is_new = pool[i]
            if key != source_key and name != new[source_key].name:
                pairs.append(_Pair(source_key, key, is_new, signal, float(score)))

        sources_by_group: dict[frozenset, list[str]] = defaultdict(list)
        for key, profile in new.items():
            sources_by_group[_type_group(profile.etype)].append(key)
        for group, keys in sources_by_group.items():
            idx = [i for i, o in enumerate(pool) if o[2] in group and plain[i]]
            if not idx:
                continue
            choices = [plain[i] for i in idx]
            queries = [_plain(new[k].name) for k in keys]
            spelling = process.cdist(queries, choices, scorer=fuzz.token_sort_ratio, score_cutoff=FUZZY_MIN)
            for q, c in zip(*np.nonzero(spelling)):
                add(keys[q], idx[c], "spelling", spelling[q, c])
            subset = process.cdist(queries, choices, scorer=fuzz.token_set_ratio, score_cutoff=100)
            for q, c in zip(*np.nonzero(subset)):
                # one-word subsets ("ICCN" in "Dewan Pakar ICCN") are noise
                if len(queries[q].split()) >= 2 and len(choices[c].split()) >= 2:
                    add(keys[q], idx[c], "containment", 100)

            singles, multis = defaultdict(list), defaultdict(list)
            for c, choice in enumerate(choices):
                words = choice.split()
                (singles if len(words) == 1 else multis)[choice[0]].append((c, words))
            for q, query in enumerate(queries):
                words = query.split()
                if not words:
                    continue
                if len(words) == 1:  # "menpora" against multi-word names starting with m
                    hits = [c for c, w in multis[query[0]] if self.abbreviates(query, w)]
                else:  # "badan pusat statistik" against one-word names starting with b
                    hits = [c for c, w in singles[query[0]] if self.abbreviates(w[0], words)]
                for c in hits:
                    add(keys[q], idx[c], "acronym", 100)

            # "jambi" / "provinsi jambi": the one-word pair the containment rule skips as noise
            by_region_name = defaultdict(list)
            for c, choice in enumerate(choices):
                by_region_name[_without_admin_prefix(choice)].append(c)
            for q, query in enumerate(queries):
                for c in by_region_name.get(_without_admin_prefix(query), ()):
                    if choices[c] != query:
                        add(keys[q], idx[c], "admin_prefix", 100)
        return pairs

    async def _propose_embedding(
        self, new: dict[str, Profile], stored: dict[str, np.ndarray] | None = None
    ) -> list[_Pair]:
        """Nearest stored entities by "name\\ndescription", plus new-vs-new similarity.

        With ``stored`` vectors (consolidation: every entity is already in the VDB)
        only the new-vs-new comparison runs, on those vectors.
        """
        pairs: list[_Pair] = []
        if stored is not None:
            keys = [k for k in new if stored.get(k) is not None]
            if len(keys) < 2:
                return []
            vectors = np.vstack([np.asarray(stored[k], dtype=np.float32) for k in keys])
        else:
            if self.entity_vdb is None:
                return []
            keys = list(new)
            texts = [f"{new[k].name}\n{new[k].description}" for k in keys]
            batch = max(1, getattr(self.entity_vdb, "_max_batch_size", 32) or 32)
            parts = await asyncio.gather(*(
                self.entity_vdb.embedding_func(texts[i:i + batch], context="document")
                for i in range(0, len(texts), batch)
            ))
            vectors = np.vstack(parts)
            nearest = await self.entity_vdb.nearest_entities(
                list(vectors), PAIRS_PER_CLUSTER, 1 - EMBED_MIN_SIMILARITY
            )
            for key, hits in zip(keys, nearest):
                for name, distance in hits:
                    if name != new[key].name:
                        pairs.append(_Pair(key, name, False, "embedding", 1 - distance))
        # ponytail: N×N in memory; block by type or first token past ~20k entities
        unit = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
        similarity = unit @ unit.T
        for i, j in zip(*np.nonzero(similarity >= EMBED_MIN_SIMILARITY)):
            if i < j:
                pairs.append(_Pair(keys[i], keys[j], True, "embedding", float(similarity[i, j])))
        return pairs

    async def _existing_profiles(self, names: set[str]) -> dict[str, Profile]:
        """Descriptions of existing candidates; names without a graph node (deleted) drop out."""
        if not names:
            return {}
        nodes = await self.graph.get_nodes_batch(sorted(names))
        return {
            name: Profile(name, type_key(node.get("entity_type", "")),
                          (node.get("description") or "")[:DESCRIPTION_CHARS])
            for name, node in nodes.items()
        }

    def _shortlist(
        self, pairs: list[_Pair], new: dict[str, Profile], targets: dict[str, Profile],
        domain: DomainRegistry | None,
    ) -> list[_Pair]:
        """Drop vetoed and duplicate pairs; keep the best few per cluster and per document."""
        rank = {"containment": 0, "spelling": 0, "acronym": 0, "admin_prefix": 0, "embedding": 1}
        seen, by_source = set(), defaultdict(list)
        for p in sorted(pairs, key=lambda p: (rank[p.signal], -p.score, p.source, p.other)):
            other = new.get(p.other) if p.other_is_new else targets.get(p.other)
            undirected = tuple(sorted((p.source, p.other))) if p.other_is_new else (p.source, p.other)
            if other is None or undirected in seen or self.is_vetoed(new[p.source], other, domain):
                continue
            seen.add(undirected)
            if len(by_source[p.source]) < PAIRS_PER_CLUSTER:
                by_source[p.source].append(p)
        shortlisted = [p for source in sorted(by_source) for p in by_source[source]]
        if len(shortlisted) > self.max_pairs:
            logger.warning(
                f"entity adjudication: {len(shortlisted)} pairs, judging the first {self.max_pairs}"
            )
            shortlisted = shortlisted[:self.max_pairs]
        for i, p in enumerate(shortlisted, start=1):
            p.pair_id = f"p{i}"
        return shortlisted

    # --- L4: vetoes ---------------------------------------------------------------

    @staticmethod
    def is_vetoed(a: Profile, b: Profile, domain: DomainRegistry | None = None) -> bool:
        """True when the pair is provably different; such pairs never reach the LLM."""
        if a.name == b.name or not _compatible(a.etype, b.etype):
            return True
        ta, tb = _tokens(a.name), _tokens(b.name)
        da, db = _numbers(ta), _numbers(tb)
        if da and db and da != db:  # Pasal 9 / Pasal 10
            return True
        ra, rb = _regions(ta), _regions(tb)
        if ra and rb and ra != rb:  # Kota Malang / Kabupaten Malang / Kota Batu
            return True
        if ta and tb and ta[0] in _INSTITUTION_PREFIXES and tb[0] in _INSTITUTION_PREFIXES and ta[0] != tb[0]:
            return True  # Badan X / Dinas X
        if (ta and tb and ta[0] in _REGIONAL_INSTITUTIONS and tb[0] in _REGIONAL_INSTITUTIONS
                and bool(ra) != bool(rb)):
            return True  # "Dinas Pariwisata" of which region?
        if domain is not None:
            for name in (a.name, *a.aliases):
                for other in (b.name, *b.aliases):
                    if frozenset((entity_key(name), entity_key(other))) in domain.never_merge:
                        return True
        return False

    # --- L5: LLM ------------------------------------------------------------------

    async def _verdicts(
        self, pairs: list[_Pair], new: dict[str, Profile], targets: dict[str, Profile]
    ) -> dict[str, PairVerdict]:
        """Stored verdicts where ``verdicts`` has them, the LLM for the rest (then stored)."""
        if self.verdicts is None:
            return await self._ask_llm(pairs, new, targets)

        def other(p: _Pair) -> Profile:
            return new[p.other] if p.other_is_new else targets[p.other]

        ids = {p.pair_id: self.verdicts.pair_id(new[p.source], other(p)) for p in pairs}
        stored = await self.verdicts.get(ids.values())
        verdicts: dict[str, PairVerdict] = {}
        for p in pairs:
            row = stored.get(ids[p.pair_id])
            if row is not None:
                # REAL column: 0.95 reads back as 0.9499999; round so the threshold still holds
                verdicts[p.pair_id] = PairVerdict(
                    pair_id=p.pair_id, same=row["same"], confidence=round(row["confidence"] or 0.0, 4),
                    preferred="a" if row["preferred"] == new[p.source].name else "b",
                    reason=row["reason"] or "",
                )
        self.cached = len(verdicts)
        fresh = [p for p in pairs if p.pair_id not in verdicts]
        if fresh:
            answered = await self._ask_llm(fresh, new, targets)
            await self.verdicts.put([
                (ids[p.pair_id], new[p.source].name, other(p).name, v.same, v.confidence,
                 new[p.source].name if v.preferred == "a" else other(p).name, v.reason)
                for p in fresh if (v := answered.get(p.pair_id)) is not None
            ])
            verdicts.update(answered)
        return verdicts

    async def _ask_llm(
        self, pairs: list[_Pair], new: dict[str, Profile], targets: dict[str, Profile]
    ) -> dict[str, PairVerdict]:
        """All batches of one document; any failure fails the whole document (deterministic)."""
        def side(p: Profile) -> dict:
            return {"name": p.name, "type": p.etype, "description": p.description}

        payloads = [
            {"id": p.pair_id, "a": side(new[p.source]),
             "b": side(new[p.other] if p.other_is_new else targets[p.other]), "signal": p.signal}
            for p in pairs
        ]
        batches = [payloads[i:i + PAIRS_PER_CALL] for i in range(0, len(payloads), PAIRS_PER_CALL)]
        try:
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(self._judge(batch)) for batch in batches]
        except Exception as e:  # TaskGroup cancels the remaining batches first
            raise AdjudicationFailed(f"{type(e).__name__}: {str(e)[:300]}") from e
        verdicts: dict[str, PairVerdict] = {}
        for task in tasks:
            verdicts.update({v.pair_id: v for v in task.result().verdicts})
        return verdicts

    async def _judge(self, batch: list[dict]) -> AdjudicationOutput:
        chain = self.chain or _adjudication_chain()
        inputs = {"pairs": json.dumps(batch, ensure_ascii=False, indent=1)}
        for attempt in (1, 2):
            async with _llm_gate:
                self.calls += 1
                try:
                    return await chain.ainvoke(inputs)
                except INVALID_OUTPUT_ERRORS as e:
                    logger.warning(f"entity adjudication: invalid output (attempt {attempt}/2): {str(e)[:300]}")
        raise AdjudicationFailed("invalid output twice")

    # --- L6: star merges ------------------------------------------------------------

    def _select(
        self, pairs: list[_Pair], verdicts: dict[str, PairVerdict], new: dict[str, Profile],
        rank: dict[str, tuple] | None = None,
    ) -> dict[str, Merge]:
        """Accepted verdicts as merges, best first, never a chain A→B→C.

        Between two new clusters the higher ``rank`` survives; on a tie (or without
        ``rank``) the LLM's preferred name does.
        """
        accepted = []
        for p in pairs:
            v = verdicts.get(p.pair_id)
            if v is None or not v.same or v.confidence < MIN_CONFIDENCE:
                continue
            source, target = p.source, p.other
            if p.other_is_new:
                ra, rb = (rank or {}).get(p.source), (rank or {}).get(p.other)
                if (ra != rb and ra is not None and rb is not None and ra > rb) or (
                    (ra == rb or ra is None or rb is None) and v.preferred == "a"
                ):
                    source, target = p.other, p.source
            accepted.append((-v.confidence, p.pair_id, source, target, p.other_is_new, v))

        merges: dict[str, Merge] = {}
        targets_used: set[str] = set()
        for _, _, source, target, target_is_new, v in sorted(accepted, key=lambda a: (a[0], a[1])):
            if source in merges or source in targets_used or (target_is_new and target in merges):
                continue
            target_name = new[target].name if target_is_new else target
            merges[source] = Merge(target_name, v.confidence, v.reason)
            targets_used.add(target)
        if merges:
            logger.info(f"entity adjudication: {len(merges)} merges from {len(pairs)} judged pairs")
        return merges

    # --- acronyms -----------------------------------------------------------------

    @staticmethod
    def abbreviates(abbr: str, words: list[str]) -> bool:
        """Could the one-token ``abbr`` abbreviate ``words``? Initials, or syllables in order."""
        if len(words) < 2 or " " in abbr or not 2 <= len(abbr) <= 15 or not abbr.isalpha():
            return False
        initials = "".join(w[0] for w in words)
        content = "".join(w[0] for w in words if w not in _STOPWORDS)
        if abbr in (initials, content):
            return True
        # Syllabic Indonesian acronyms (Kemenparekraf, Menpora): letters in order across
        # at least two words, first letter shared.
        if len(abbr) < 4 or abbr[0] != words[0][0]:
            return False
        used, rest = 0, abbr
        for w in words:
            before = len(rest)
            for c in w:
                if rest and c == rest[0]:
                    rest = rest[1:]
            used += len(rest) < before
        return not rest and used >= 2


# --- helpers -------------------------------------------------------------------------


def _type_group(etype: str) -> frozenset[str]:
    return next((g for g in _TYPE_GROUPS if etype in g), frozenset({etype}))


def _compatible(a: str, b: str) -> bool:
    return b in _type_group(a)


def _tokens(name: str) -> list[str]:
    return [t for t in entity_key(name).split() if t not in _CONNECTORS]


def _plain(name: str) -> str:
    return " ".join(_tokens(name))


def _without_admin_prefix(plain: str) -> str:
    """``provinsi jambi`` → ``jambi``; a name without a leading admin token is returned as is."""
    first, _, rest = plain.partition(" ")
    return rest if first in _ADMIN_TOKENS and rest else plain


def _numbers(tokens: list[str]) -> list[str]:
    return [t for t in tokens if any(c.isdigit() for c in t)]


def _regions(tokens: list[str]) -> frozenset[tuple[str, tuple[str, ...]]]:
    """``kota kabupaten kreatif`` → {(kota, ()), (kabupaten, (kreatif,))}."""
    regions, current, words = [], None, []
    for t in tokens + [None]:
        if t is None or t in _ADMIN_TOKENS:
            if current is not None:
                regions.append((current, tuple(words)))
            current, words = t, []
        elif current is not None:
            words.append(t)
    return frozenset(regions)

