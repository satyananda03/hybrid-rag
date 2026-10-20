"""Compare EXTRACT_LLM_MODEL candidates on a real ingest.

    uv run --extra ingestion python evaluation/extract_model_benchmark.py ingest <model> <workspace> [file ...] [--entity-registry iccn]
    uv run --extra ingestion python evaluation/extract_model_benchmark.py report <workspace> [<workspace> ...]
    uv run --extra ingestion python evaluation/extract_model_benchmark.py judge <ws_a> <ws_b> [--n 24] [--judge MODEL]

ingest  runs the full pipeline into its own workspace with EXTRACT_LLM_MODEL=<model>,
        recording every extraction pass (seconds, tokens, records, invalid output, how many
        entity names occur verbatim in the chunk; translated names count as not verbatim) to evaluation/results/extract_bench_<ws>.jsonl.
report  speed, robustness and graph shape per workspace, side by side.
judge   a third model scores both extractions of the same chunks (order randomised):
        precision, recall (1-5) and a winner. The quality number the counts cannot give.
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation" / "results"
DEFAULT_FILES = [
    "Peraturan Presiden Nomor 37 Tahun 2026 Tentang Rencana Induk Ekonomi Kreatif Tahun 2026-2045.md",
    "Overview ICCN.pdf",
    "10 Prinsip Kota Kreatif.pdf",
    "Susunan-Pengurus-ICCN-(2025-2028)-Cleaned-rev-FIX (1).pdf",
    "02. JAWA - KorKotKab + Jejaring ICCN 2026-2028.xlsx",
]


def _log(workspace: str) -> Path:
    return RESULTS / f"extract_bench_{workspace}.jsonl"


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _setup() -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "evaluation"))
    from entity_dedup_benchmark import configure_providers

    configure_providers()


def _grounded(name: str, text_tokens: set[str]) -> bool:
    """Every word of the name occurs in the chunk: a cheap check against invented entities."""
    words = {w for w in _tokens(name) if len(w) > 2}
    return words <= text_tokens


async def ingest(model: str, workspace: str, files: list[str], entity_registry: str | None = None) -> None:
    # Settings are read at import, so the model must be in the environment first.
    os.environ["EXTRACT_LLM_MODEL"] = model
    _setup()
    from langchain_core.callbacks import get_usage_metadata_callback

    from hybrid_rag.ingestion import parse_file, workspace_pool
    from hybrid_rag.ingestion.knowledge_graph import extraction
    from hybrid_rag.ingestion.rag import IngestionRAG

    RESULTS.mkdir(parents=True, exist_ok=True)
    log = _log(workspace).open("a", encoding="utf-8")
    current = {"file": ""}

    def write(record: dict) -> None:
        log.write(json.dumps({"model": model, "file": current["file"], **record}, ensure_ascii=False) + "\n")
        log.flush()

    extract_or_none = extraction._extract_or_none

    async def timed_extract(chain, inputs, config, chunk_key, pass_name):
        start = time.perf_counter()
        # Per pass, not per file: the same model may also serve contextualize/VLM.
        with get_usage_metadata_callback() as usage:
            out = await extract_or_none(chain, inputs, config, chunk_key, pass_name)
        text = inputs.get("input_text", "")
        entities = [e.model_dump() for e in getattr(out, "entities", None) or []]
        relationships = [r.model_dump() for r in getattr(out, "relationships", None) or []]
        text_tokens = _tokens(text)
        write({"kind": "pass", "pass": pass_name, "chunk": chunk_key,
               "text_hash": hashlib.md5(text.encode()).hexdigest(), "seconds": time.perf_counter() - start,
               "invalid": out is None, "output_tokens": sum(u.get("output_tokens", 0) for u in usage.usage_metadata.values()), "grounded": sum(_grounded(e["name"], text_tokens) for e in entities),
               "entities": entities, "relationships": relationships,
               "text": text if pass_name == "extraction" else ""})
        return out

    extraction._extract_or_none = timed_extract

    process_extract = IngestionRAG._process_extract_entities

    async def timed_stage(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return await process_extract(self, *args, **kwargs)
        finally:
            write({"kind": "stage", "seconds": time.perf_counter() - start})

    IngestionRAG._process_extract_entities = timed_stage

    data = ROOT / "examples" / "data"
    async with workspace_pool.lease(workspace) as rag:
        for name in files:
            current["file"] = name
            start = time.perf_counter()
            error = None
            try:
                async with parse_file(data / name, entity_registry=entity_registry) as content_list:
                    await rag.insert_content_list(content_list, file_path=name, entity_registry=entity_registry)
            except Exception as e:  # one bad file must not stop the run
                error = f"{type(e).__name__}: {e}"[:500]
            write({"kind": "file", "seconds": time.perf_counter() - start, "error": error})
            print(f"{name}: {time.perf_counter() - start:.0f}s" + (f" FAILED {error}" if error else ""), flush=True)


def _records(workspace: str) -> list[dict]:
    return [json.loads(line) for line in _log(workspace).read_text(encoding="utf-8").splitlines()]


async def report(workspaces: list[str]) -> None:
    _setup()
    from hybrid_rag.ingestion import workspace_pool

    rows: dict[str, dict] = {}
    for ws in workspaces:
        recs = _records(ws)
        model = recs[0]["model"]
        passes = [r for r in recs if r["kind"] == "pass"]
        first = [r for r in passes if r["pass"] == "extraction"]
        files = [r for r in recs if r["kind"] == "file"]
        entities = sum(len(r["entities"]) for r in first)
        names = [e["name"] for r in first for e in r["entities"]]
        async with workspace_pool.lease(ws) as rag:
            nodes = await rag.chunk_entity_relation_graph.get_all_nodes()
            edges = await rag.chunk_entity_relation_graph.get_all_edges()
        edges = {frozenset((e["source"], e["target"])) for e in edges}  # the undirected match returns both directions
        linked = {n for e in edges for n in e}
        rows[ws] = {
            "model": model,
            "files ok / total": f"{sum(not f['error'] for f in files)}/{len(files)}",
            "ingest wall s (sum of files)": round(sum(f["seconds"] for f in files)),
            "extract stage wall s": round(sum(r["seconds"] for r in recs if r["kind"] == "stage")),
            "chunks": len(first),
            "median s / extraction call": round(sorted(r["seconds"] for r in first)[len(first) // 2], 1) if first else 0,
            "invalid outputs (all passes)": sum(r["invalid"] for r in passes),
            "repair passes (orphans left)": sum(r["pass"] == "repair" for r in passes),
            "output tokens (all passes)": sum(r.get("output_tokens", 0) for r in passes),
            "entities / chunk (1st pass)": round(entities / max(1, len(first)), 1),
            "relations / chunk (1st pass)": round(sum(len(r["relationships"]) for r in first) / max(1, len(first)), 1),
            "names verbatim in chunk %": round(100 * sum(r["grounded"] for r in first) / max(1, entities), 1),
            "distinct types": len({e["type"] for r in first for e in r["entities"]}),
            "graph nodes": len(nodes),
            "graph edges": len(edges),
            "edges / node": round(len(edges) / max(1, len(nodes)), 2),
            "isolated nodes": sum(n["id"] not in linked for n in nodes),
            "dup names (casefold) in 1st pass": len(names) - len({n.casefold() for n in names}),
        }
    keys = list(next(iter(rows.values())))
    width = max(map(len, keys))
    print(" " * width + "  " + "  ".join(f"{ws:>24}" for ws in rows))
    for k in keys:
        print(f"{k:<{width}}  " + "  ".join(f"{str(rows[ws][k]):>24}" for ws in rows))


JUDGE_PROMPT = """You audit knowledge-graph extraction. Below is a source text and two extractions (A and B) of it.
Judge each against the text only:
- precision 1-5: entities/relations are real, correctly typed, faithful to the text, no invention or noise.
- recall 1-5: the important entities and relations of the text are captured (people, organisations, programs,
  regulations, places, concepts that matter for answering questions about this text).
Then pick the extraction you would rather have in the graph: "A", "B" or "tie". Be strict and brief.

SOURCE TEXT:
{text}

EXTRACTION A:
{a}

EXTRACTION B:
{b}"""


def _render(rec: dict) -> str:
    ents = "\n".join(f"- {e['name']} [{e['type']}]: {e['description']}" for e in rec["entities"])
    rels = "\n".join(f"- {r['source']} -> {r['target']}: {r['description']}" for r in rec["relationships"])
    return f"Entities:\n{ents or '(none)'}\nRelations:\n{rels or '(none)'}"


async def judge(ws_a: str, ws_b: str, n: int, judge_model: str) -> None:
    sys.path.insert(0, str(ROOT))
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field

    from hybrid_rag.config import settings

    class Verdict(BaseModel):
        precision_a: int = Field(ge=1, le=5)
        recall_a: int = Field(ge=1, le=5)
        precision_b: int = Field(ge=1, le=5)
        recall_b: int = Field(ge=1, le=5)
        winner: str = Field(description='"A", "B" or "tie"')
        reason: str = Field(description="one sentence")

    def first_passes(ws):
        return {r["text_hash"]: r for r in _records(ws) if r["kind"] == "pass" and r["pass"] == "extraction" and not r["invalid"]}

    a, b = first_passes(ws_a), first_passes(ws_b)
    shared = sorted(a.keys() & b.keys())
    random.Random(0).shuffle(shared)
    sample = shared[:n]
    llm = ChatOpenAI(model=judge_model, base_url=settings.extract_llm_binding_host,
                     api_key=settings.extract_llm_binding_api_key, timeout=600, max_retries=2,
                     max_tokens=16000).with_structured_output(Verdict, method="function_calling")
    gate = asyncio.Semaphore(8)
    rng = random.Random(1)

    async def one(h):
        swap = rng.random() < 0.5  # position bias: each model is "A" about half the time
        first, second = (b[h], a[h]) if swap else (a[h], b[h])
        async with gate:
            v = await llm.ainvoke(JUDGE_PROMPT.format(text=a[h]["text"], a=_render(first), b=_render(second)))
        p_a, r_a, p_b, r_b = v.precision_a, v.recall_a, v.precision_b, v.recall_b
        winner = {"A": ws_a, "B": ws_b}.get(v.winner.strip().upper(), "tie")
        if swap:
            p_a, r_a, p_b, r_b = p_b, r_b, p_a, r_a
            winner = {ws_a: ws_b, ws_b: ws_a}.get(winner, "tie")
        return {"hash": h, "file": a[h]["file"], "precision": (p_a, p_b), "recall": (r_a, r_b),
                "winner": winner, "reason": v.reason}

    verdicts = await asyncio.gather(*(one(h) for h in sample), return_exceptions=True)
    ok = [v for v in verdicts if isinstance(v, dict)]
    out = RESULTS / f"extract_bench_judge_{ws_a}_vs_{ws_b}.json"
    out.write_text(json.dumps(ok, ensure_ascii=False, indent=1), encoding="utf-8")
    wins = Counter(v["winner"] for v in ok)
    mean = lambda xs: round(sum(xs) / max(1, len(xs)), 2)  # noqa: E731
    print(f"judge {judge_model}: {len(ok)}/{len(sample)} chunks judged ({len(shared)} shared)")
    for i, ws in enumerate((ws_a, ws_b)):
        print(f"  {ws:<28} precision {mean([v['precision'][i] for v in ok])}  recall {mean([v['recall'][i] for v in ok])}  wins {wins[ws]}")
    print(f"  ties {wins['tie']}   -> {out.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest")
    p.add_argument("model")
    p.add_argument("workspace")
    p.add_argument("files", nargs="*", default=DEFAULT_FILES)
    p.add_argument("--entity-registry", default=None,
                   help="YAML in knowledge_graph/entity_registry/; used by both parse_file and insert_content_list")
    p = sub.add_parser("report")
    p.add_argument("workspaces", nargs="+")
    p = sub.add_parser("judge")
    p.add_argument("ws_a")
    p.add_argument("ws_b")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--judge", default="global.xai.grok-4.6")
    args = parser.parse_args()
    if args.cmd == "ingest":
        asyncio.run(ingest(args.model, args.workspace, args.files, args.entity_registry))
    elif args.cmd == "report":
        asyncio.run(report(args.workspaces))
    else:
        asyncio.run(judge(args.ws_a, args.ws_b, args.n, args.judge))


if __name__ == "__main__":
    main()
