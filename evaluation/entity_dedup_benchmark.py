"""Entity deduplication benchmark on the raw corpus in examples/data/ (no labels yet).

    uv run --extra ingestion python evaluation/entity_dedup_benchmark.py ingest [--entity-registry iccn]
    uv run --extra ingestion python evaluation/entity_dedup_benchmark.py report

ingest  parses every file of examples/data/ and ingests it into one workspace
        (default workspace_entity_test) with entity resolution on.
report  measures what resolution did and what it left:
        - nodes now vs. without resolution (nodes + names the registry rewrote);
        - residual duplicates: nodes sharing an entity_key / match_key (should be 0);
        - near duplicates, rapidfuzz token_set_ratio >= 90 within a type: recall
          left for Fase 2, and hard negatives to keep apart;
        - the target cases (ICCN, Kota/Kabupaten);
        and writes evaluation/results/entity_dedup_<workspace>_{merges,near}.csv.
        Fill the empty `label` column of the merges file (1 = same entity,
        0 = false merge): precision = mean(label), target >= 0.98.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hybrid_rag import providers  # noqa: E402
from hybrid_rag.ingestion import parse_file, workspace_pool  # noqa: E402
import numpy as np  # noqa: E402

from hybrid_rag.ingestion.knowledge_graph import entity_adjudication, entity_resolution  # noqa: E402
from hybrid_rag.ingestion.knowledge_graph.entity_adjudication import (  # noqa: E402
    AdjudicationOutput,
    EntityAdjudicator,
)
from hybrid_rag.ingestion.knowledge_graph.entity_resolution import (  # noqa: E402
    EntityResolver,
    entity_key,
    match_key,
    type_key,
)

RESULTS = ROOT / "evaluation" / "results"
NEAR_DUPLICATE_SCORE = 90
REGISTRY_SQL = """SELECT id, alias_name, entity_type, canonical_entity_id, source, evidence
                  FROM hybrid_rag_entity_alias WHERE workspace=$1 ORDER BY id"""


def configure_providers() -> None:
    """The host hooks of examples/rag.ipynb, with langchain's own retries."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    def get_llm(model_id, *, base_url, api_key, timeout, max_retries, max_tokens=None, **model_kwargs):
        return ChatOpenAI(model=model_id, base_url=base_url, api_key=api_key, timeout=timeout,
                          max_retries=max_retries, max_tokens=max_tokens, **model_kwargs)

    class IndexPairedEmbeddings(OpenAIEmbeddings):
        """Vectors paired by ``index``, not position, on both paths (see ``providers.embed_by_index``)."""

        def embed_documents(self, texts, chunk_size=None, **kwargs):
            self._ensure_sync_client_available()
            create = partial(self.client.create, **self._invocation_params, **kwargs)
            return providers.embed_by_index_sync(create, texts, chunk_size or self.chunk_size)

        async def aembed_documents(self, texts, chunk_size=None, **kwargs):
            create = partial(self.async_client.create, **self._invocation_params, **kwargs)
            return await providers.embed_by_index(create, texts, chunk_size or self.chunk_size)

    def get_embeddings(model_id, *, base_url, api_key, timeout, max_retries):
        return IndexPairedEmbeddings(model=model_id, base_url=base_url, api_key=api_key, timeout=timeout,
                                     max_retries=max_retries, model_kwargs={"encoding_format": "float"},
                                     check_embedding_ctx_length=False)

    providers.configure(get_llm=get_llm, get_vlm=get_llm, get_embeddings=get_embeddings)


def capture_resolutions(path: Path) -> None:
    """Record every resolve() call: raw names in, mapping out, seconds spent.

    The raw names are what the golden set is labelled on and what `evaluate`
    replays offline under other configurations.
    """
    last_mapping: dict[str, str] = {}
    rewrite, resolve = entity_resolution.rewrite, EntityResolver.resolve

    def recording_rewrite(nodes, edges, mapping):
        last_mapping.clear()
        last_mapping.update(mapping)
        return rewrite(nodes, edges, mapping)

    async def timed_resolve(self, nodes, edges, registry_name=None, **kwargs):
        last_mapping.clear()
        start = time.perf_counter()
        out = await resolve(self, nodes, edges, registry_name, **kwargs)
        seconds = time.perf_counter() - start
        entities = []
        for name, records in nodes.items():
            types = Counter(type_key(r.get("entity_type", "")) for r in records)
            entities.append({
                "name": name,
                "type": max(sorted(types), key=types.__getitem__),
                "description": next((r["description"] for r in records if r.get("description")), "")[:300],
                "canonical": last_mapping.get(name, name),
            })
        file_path = next((r.get("file_path") for rs in nodes.values() for r in rs), "")
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"file": file_path, "registry": registry_name, "seconds": seconds,
                                "use_llm": kwargs.get("use_llm", False),
                                "edges": len(edges), "entities": entities}, ensure_ascii=False) + "\n")
        return out

    entity_resolution.rewrite = recording_rewrite
    EntityResolver.resolve = timed_resolve


async def ingest(workspace: str, data_dir: Path, entity_registry: str | None) -> None:
    files = sorted(p for p in data_dir.iterdir() if p.is_file())
    RESULTS.mkdir(parents=True, exist_ok=True)
    capture_resolutions(RESULTS / f"entity_dedup_{workspace}_raw.jsonl")
    failed = []
    async with workspace_pool.lease(workspace) as rag:
        for i, path in enumerate(files, start=1):
            print(f"[{i}/{len(files)}] {path.name}")
            try:
                async with parse_file(path, entity_registry=entity_registry) as content_list:
                    await rag.insert_content_list(
                        content_list, file_path=str(path), entity_registry=entity_registry
                    )
            except Exception as e:  # one bad file must not stop the corpus
                failed.append(path.name)
                print(f"  FAILED: {type(e).__name__}: {e}")
    print(f"ingested {len(files) - len(failed)}/{len(files)} files" + (f", failed: {failed}" if failed else ""))


def _groups(nodes: list[dict], key) -> list[list[str]]:
    by_key = defaultdict(list)
    for n in nodes:
        k = key(n["id"])
        if k:
            by_key[(type_key(n.get("entity_type", "")), k)].append(n["id"])
    return [sorted(v) for v in by_key.values() if len(v) > 1]


def _near_duplicates(nodes: list[dict]) -> list[tuple[str, str, str, float]]:
    from rapidfuzz import fuzz, process

    by_type = defaultdict(list)
    for n in nodes:
        by_type[type_key(n.get("entity_type", ""))].append(n["id"])
    pairs = []
    for etype, names in sorted(by_type.items()):
        # ponytail: O(n^2) per type, fine for one corpus; block first if a type grows past ~20k
        scores = process.cdist(names, names, scorer=fuzz.token_set_ratio,
                               processor=entity_key, score_cutoff=NEAR_DUPLICATE_SCORE)
        for i, j in zip(*scores.nonzero()):
            if i < j and match_key(names[i]) != match_key(names[j]):
                pairs.append((etype, names[i], names[j], float(scores[i, j])))
    return sorted(pairs, key=lambda p: (-p[3], p[0], p[1]))


def _write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


async def report(workspace: str) -> None:
    async with workspace_pool.lease(workspace) as rag:
        nodes = await rag.chunk_entity_relation_graph.get_all_nodes()
        registry = await rag.entity_aliases.db.query(REGISTRY_SQL, [rag.workspace], multirows=True) or []

    # A row whose name differs from its canonical is a name rewritten at least once
    # (variants sharing a match_key share a row, so this is a lower bound).
    merges = [r for r in registry
              if r["alias_name"] != r["canonical_entity_id"] and not r["evidence"].endswith("_implied")]
    print(f"workspace          {workspace}")
    print(f"nodes now          {len(nodes)}")
    print(f"without resolution >= {len(nodes) + len(merges)}  ({len(merges)} names merged away)")
    by_evidence = defaultdict(int)
    for r in merges:
        by_evidence[f"{r['source']}:{r['evidence']}"] += 1
    for evidence, count in sorted(by_evidence.items()):
        print(f"  {evidence:<28} {count}")

    for label, key in (("entity_key", entity_key), ("match_key", match_key)):
        groups = _groups(nodes, key)
        print(f"residual duplicates by {label}: {len(groups)}")
        for g in groups[:10]:
            print(f"  {g}")

    near = _near_duplicates(nodes)
    print(f"near duplicates (token_set >= {NEAR_DUPLICATE_SCORE}, same type): {len(near)}")
    for etype, a, b, score in near[:15]:
        print(f"  {score:5.1f} [{etype}] {a!r} ~ {b!r}")

    print("target probes:")
    for needle in ("iccn", "creative cities", "kota kreatif", "malang", "kemen"):
        hits = sorted(n["id"] for n in nodes if needle in entity_key(n["id"]))
        print(f"  {needle!r}: {hits[:12]}{' ...' if len(hits) > 12 else ''}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    merges_csv = RESULTS / f"entity_dedup_{workspace}_merges.csv"
    near_csv = RESULTS / f"entity_dedup_{workspace}_near.csv"
    _write_csv(merges_csv, ["entity_type", "alias_name", "canonical", "source", "evidence", "label"],
               [(r["entity_type"], r["alias_name"], r["canonical_entity_id"], r["source"], r["evidence"], "")
                for r in merges])
    _write_csv(near_csv, ["entity_type", "a", "b", "token_set", "label"], [(*p, "") for p in near])
    print(f"wrote {merges_csv.relative_to(ROOT)} and {near_csv.relative_to(ROOT)}")


# --- golden set: candidates to label, then precision/recall ------------------------

GOLDEN = ROOT / "evaluation" / "datasets" / "entity_dedup_golden.csv"
FUZZY_SCORE = 88


def _load_raw(workspace: str) -> list[dict]:
    path = RESULTS / f"entity_dedup_{workspace}_raw.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _entities(docs: list[dict]) -> dict[str, dict]:
    """Every distinct raw name (one node each without resolution) with its majority type."""
    out: dict[str, dict] = {}
    for doc in docs:
        for e in doc["entities"]:
            entry = out.setdefault(e["name"], {"types": Counter(), "description": e["description"], "files": set()})
            entry["types"][e["type"]] += 1
            entry["files"].add(Path(doc["file"]).name)
    for entry in out.values():
        entry["type"] = max(sorted(entry["types"]), key=entry["types"].__getitem__)
    return out


def _abbreviates(abbr: str, words: list[str]) -> str | None:
    """Signal name when the one-token `abbr` could abbreviate `words` (the resolver's own test)."""
    if not EntityAdjudicator.abbreviates(abbr, words):
        return None
    initials = {"".join(w[0] for w in words),
                "".join(w[0] for w in words if w not in entity_resolution._STOPWORDS)}
    return "initials" if abbr in initials else "syllabic"


def candidates(workspace: str) -> None:
    """Pairs worth a human look, found without the resolver's own rules."""
    from rapidfuzz import fuzz, process

    docs = _load_raw(workspace)
    entities = _entities(docs)
    names = sorted(entities)
    keys = [entity_key(n) for n in names]
    placed = _placements(docs)
    signals: dict[tuple[str, str], set[str]] = defaultdict(set)

    def add(a: str, b: str, signal: str) -> None:
        if a != b:
            signals[tuple(sorted((a, b)))].add(signal)

    # 1. what the resolver merged (precision needs every one of them)
    by_node = defaultdict(set)
    for name, nodes in placed.items():
        for node in nodes:
            by_node[node].add(name)
    for members in by_node.values():
        for a in members:
            for b in members:
                add(a, b, "system")
    # 2. same key, any type
    for key_fn in (entity_key, match_key):
        by_key = defaultdict(list)
        for n in names:
            if key_fn(n):
                by_key[key_fn(n)].append(n)
        for group in by_key.values():
            for a in group:
                for b in group:
                    add(a, b, key_fn.__name__)
    # 3. fuzzy (token_set only when both sides have 2+ tokens: one-word subsets score 100)
    sort = process.cdist(keys, keys, scorer=fuzz.token_sort_ratio, score_cutoff=FUZZY_SCORE)
    sets = process.cdist(keys, keys, scorer=fuzz.token_set_ratio, score_cutoff=FUZZY_SCORE)
    multi = [len(k.split()) >= 2 for k in keys]
    for i, j in zip(*sort.nonzero()):
        if i < j:
            add(names[i], names[j], "token_sort")
    for i, j in zip(*sets.nonzero()):
        if i < j and multi[i] and multi[j]:
            add(names[i], names[j], "token_set")
    # 4. abbreviations, including "Long (ABBR)" spelled out elsewhere
    expansions = []  # (words, name it came from)
    for n, k in zip(names, keys):
        expansions.append((k.split(), n))
        m = entity_resolution._LONG_ABBR.match(n)
        if m:
            expansions.append((entity_key(m["long"]).split(), n))
            for other in names:
                if match_key(other) in (match_key(m["long"]), match_key(m["abbr"])):
                    add(n, other, "paren")
    for n, k in zip(names, keys):
        if " " in k:
            continue
        for words, other in expansions:
            signal = _abbreviates(k, words)
            if signal:
                add(n, other, signal)

    rows = []
    for (a, b), sig in sorted(signals.items()):
        ea, eb = entities[a], entities[b]
        rows.append([ea["type"], a, eb["type"], b, "+".join(sorted(sig)),
                     int(bool(placed[a] & placed[b])), "", ea["description"], eb["description"],
                     "; ".join(sorted(ea["files"] | eb["files"]))])
    out = RESULTS / f"entity_dedup_{workspace}_candidates.csv"
    _write_csv(out, ["type_a", "a", "type_b", "b", "signals", "system_merged", "label",
                     "description_a", "description_b", "files"], rows)
    print(f"{len(entities)} distinct raw names, {len(rows)} candidate pairs -> {out.relative_to(ROOT)}")
    print("label: 1 = same real-world entity, 0 = different; save as " + str(GOLDEN.relative_to(ROOT)))


def _placements(docs: list[dict]) -> dict[str, set[str]]:
    """name -> the node(s) it was written to in the captured run."""
    placed: dict[str, set[str]] = defaultdict(set)
    for doc in docs:
        for e in doc["entities"]:
            placed[e["name"]].add(e["canonical"])
    return placed


class _MemoryRegistry:
    """hybrid_rag_entity_alias semantics in memory, for offline replays."""

    def __init__(self):
        self.rows: dict[str, tuple[str, str]] = {}

    async def get(self, ids):
        return {i: self.rows[i][0] for i in set(ids) if i in self.rows}

    async def add(self, decisions):
        for d in decisions:
            if d.alias_id not in self.rows or d.source == "user":
                self.rows[d.alias_id] = (d.canonical, d.source)

    async def known_entities(self):
        return sorted({(canonical, i.split(":", 1)[0]) for i, (canonical, _) in self.rows.items()})


class _MemoryGraph:
    """Neo4j stand-in: the nodes written by the documents replayed so far."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}

    async def get_nodes_batch(self, names):
        return {n: self.nodes[n] for n in names if n in self.nodes}


class _MemoryVectors:
    """Entity VDB stand-in: real embeddings, searchable only for earlier documents."""

    _max_batch_size = 32

    def __init__(self, embedding_func):
        self.embedding_func = embedding_func
        self.names: list[str] = []
        self.unit = np.zeros((0, 0), dtype=np.float32)

    async def add(self, texts: dict[str, str]) -> None:
        names = [n for n in texts if n not in set(self.names)]
        if not names:
            return
        batches = [names[i:i + self._max_batch_size] for i in range(0, len(names), self._max_batch_size)]
        vectors = np.vstack(await asyncio.gather(
            *(self.embedding_func([texts[n] for n in b], context="document") for b in batches)))
        vectors = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
        self.unit = vectors if not self.names else np.vstack([self.unit, vectors])
        self.names += names

    async def nearest_entities(self, vectors, top_k, max_distance):
        if not self.names:
            return [[] for _ in vectors]
        query = np.vstack(vectors)
        query = query / np.clip(np.linalg.norm(query, axis=1, keepdims=True), 1e-12, None)
        distance = 1 - query @ self.unit.T
        out = []
        for row in distance:
            best = np.argsort(row)[:top_k]
            out.append([(self.names[i], float(row[i])) for i in best if row[i] < max_distance])
        return out


class _MemoChain:
    """The real adjudication LLM, memoised on disk so re-running `evaluate` is cheap."""

    def __init__(self, path: Path):
        self.path = path
        self.memo = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.chain = entity_adjudication._adjudication_chain()
        self.calls = self.cached = self.pairs = 0

    async def ainvoke(self, inputs):
        key = hashlib.sha1(inputs["pairs"].encode()).hexdigest()
        self.pairs += len(json.loads(inputs["pairs"]))
        if key in self.memo:
            self.cached += 1
            return AdjudicationOutput.model_validate(self.memo[key])
        self.calls += 1
        out = await self.chain.ainvoke(inputs)
        self.memo[key] = out.model_dump()
        self.path.write_text(json.dumps(self.memo, ensure_ascii=False), encoding="utf-8")
        return out


async def _replay(
    docs: list[dict], registry_name: str | None, chain: _MemoChain | None = None
) -> dict[str, set[str]]:
    """Re-run resolution over the captured raw names, documents in ingest order.

    With ``chain``, Fase 2 runs too, against a graph and an entity VDB that hold
    only what the earlier documents wrote (no peeking at later documents).
    """
    registry, graph = _MemoryRegistry(), _MemoryGraph()
    adjudicator = vectors = None
    if chain is not None:
        from hybrid_rag.providers import build_embedding_func

        vectors = _MemoryVectors(build_embedding_func())
        adjudicator = EntityAdjudicator(graph=graph, entity_vdb=vectors, chain=chain)
    resolver, placed, seen = EntityResolver(registry, adjudicator), defaultdict(set), {}
    rewrite = entity_resolution.rewrite

    def recording(nodes, edges, mapping):
        seen.clear()
        seen.update(mapping)
        return rewrite(nodes, edges, mapping)

    entity_resolution.rewrite = recording
    try:
        for doc in docs:
            nodes = {e["name"]: [{"entity_name": e["name"], "entity_type": e["type"],
                                  "description": e["description"]}] for e in doc["entities"]}
            seen.clear()
            await resolver.resolve(nodes, {}, registry_name, use_llm=chain is not None)
            written = {}
            for name, records in nodes.items():
                canonical = seen.get(name, name)
                placed[name].add(canonical)
                graph.nodes.setdefault(canonical, {"entity_type": records[0]["entity_type"],
                                                   "description": records[0]["description"]})
                written.setdefault(canonical, f"{canonical}\n{records[0]['description']}")
            if vectors is not None:
                await vectors.add(written)
    finally:
        entity_resolution.rewrite = rewrite
    return placed


def _scores(labels: list[tuple[str, str, int]], placed: dict[str, set[str]]) -> dict:
    tp = fp = fn = 0
    false_merges, missed = [], []
    for a, b, same in labels:
        merged = bool(placed[a] & placed[b])
        if merged and same:
            tp += 1
        elif merged:
            fp += 1
            false_merges.append((a, b))
        elif same:
            fn += 1
            missed.append((a, b))
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "nodes": len({n for nodes in placed.values() for n in nodes}),
            "false_merges": false_merges, "missed": missed}


async def evaluate(workspace: str, use_llm: bool = True) -> None:
    docs = _load_raw(workspace)
    with GOLDEN.open(encoding="utf-8") as f:
        golden = [r for r in csv.DictReader(f) if r["label"].strip()]
    # "ambiguous" pairs could not be decided from the corpus (e.g. a common name with
    # two roles): scored once without them and once counted as different (worst case).
    clear = [(r["a"], r["b"], int(r["label"])) for r in golden if r["ambiguous"] != "1"]
    worst = [(r["a"], r["b"], 0 if r["ambiguous"] == "1" else int(r["label"])) for r in golden]
    signals = {(r["a"], r["b"]): r["signals"] for r in golden}
    labelled = {frozenset((r["a"], r["b"])) for r in golden}
    positives = sum(s for *_, s in clear)
    entities = _entities(docs)
    print(f"golden pairs: {len(golden)} ({positives} same, {len(clear) - positives} different, "
          f"{len(golden) - len(clear)} ambiguous)")
    print(f"raw names (nodes without resolution): {len(entities)}")

    configs = [("replay: rules only", await _replay(docs, None), None),
               ("replay: rules + iccn.yaml", await _replay(docs, "iccn"), None)]
    if use_llm:
        RESULTS.mkdir(parents=True, exist_ok=True)
        memo = RESULTS / f"entity_dedup_{workspace}_llm_memo.json"
        for label, registry_name in (("replay: rules + LLM", None), ("replay: rules + LLM + iccn", "iccn")):
            chain, start = _MemoChain(memo), time.perf_counter()
            placed = await _replay(docs, registry_name, chain)
            cost = (f"{chain.calls} LLM calls (+{chain.cached} memoised), {chain.pairs} pairs judged, "
                    f"{time.perf_counter() - start:.0f}s")
            configs.append((label, placed, cost))

    print(f"\n{'configuration':<27} {'nodes':>6} {'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'precision':>9} {'recall':>7} {'F1':>6}   precision worst-case")
    for label, placed, _ in configs:
        s, w = _scores(clear, placed), _scores(worst, placed)
        f1 = 2 * s["precision"] * s["recall"] / (s["precision"] + s["recall"]) if s["tp"] else 0.0
        print(f"{label:<27} {s['nodes']:>6} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} "
              f"{s['precision']:>9.3f} {s['recall']:>7.3f} {f1:>6.3f}   {w['precision']:.3f}")
    for label, _, cost in configs:
        if cost:
            print(f"  {label}: {cost}")

    label, placed, _ = configs[-1]
    s = _scores(clear, placed)
    print(f"\n[{label}] false merges ({len(s['false_merges'])}):")
    for a, b in s["false_merges"]:
        print(f"  {a!r} + {b!r}")
    by_signal = Counter(signals[(a, b)] for a, b in s["missed"])
    print(f"[{label}] missed {len(s['missed'])}, by candidate signal: {dict(by_signal.most_common())}")
    for a, b in s["missed"]:
        print(f"  [{signals[(a, b)]}] {a!r} ~ {b!r}")

    # Merges the golden set cannot score: label them before trusting the precision above.
    unlabelled = set()
    for _, placed, cost in configs:
        if not cost:
            continue
        by_node = defaultdict(set)
        for name, nodes in placed.items():
            for node in nodes:
                by_node[node].add(name)
        unlabelled |= {tuple(sorted((a, b))) for members in by_node.values() for a in members
                       for b in members if a != b and frozenset((a, b)) not in labelled}
    out = RESULTS / f"entity_dedup_{workspace}_unlabelled_merges.csv"
    _write_csv(out, ["type_a", "a", "type_b", "b", "label", "description_a", "description_b"],
               [(entities[a]["type"], a, entities[b]["type"], b, "",
                 entities[a]["description"], entities[b]["description"]) for a, b in sorted(unlabelled)])
    print(f"\n{len(unlabelled)} merged pairs (any LLM configuration) are not in the golden set "
          f"-> {out.relative_to(ROOT)}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["ingest", "report", "candidates", "evaluate"])
    parser.add_argument("--workspace", default="workspace_entity_test")
    parser.add_argument("--data", type=Path, default=ROOT / "examples" / "data")
    parser.add_argument("--entity-registry", default=None,
                        help="YAML in knowledge_graph/entity_registry/; used by both parse_file and insert_content_list")
    parser.add_argument("--no-llm", action="store_true", help="evaluate: replay the rules only")
    parser.add_argument("--embed-min", type=float, default=None,
                        help="evaluate: override EMBED_MIN_SIMILARITY for a calibration run")
    parser.add_argument("--min-confidence", type=float, default=None,
                        help="evaluate: override MIN_CONFIDENCE for a calibration run")
    args = parser.parse_args()
    if args.embed_min is not None:
        entity_adjudication.EMBED_MIN_SIMILARITY = args.embed_min
    if args.min_confidence is not None:
        entity_adjudication.MIN_CONFIDENCE = args.min_confidence
    logging.basicConfig(level=logging.INFO if args.command == "ingest" else logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    configure_providers()
    try:
        if args.command == "ingest":
            await ingest(args.workspace, args.data, args.entity_registry)
        elif args.command == "report":
            await report(args.workspace)
        elif args.command == "candidates":
            candidates(args.workspace)
        else:
            await evaluate(args.workspace, use_llm=not args.no_llm)
    finally:
        await workspace_pool.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
