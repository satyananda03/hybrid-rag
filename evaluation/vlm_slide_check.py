"""Ingest one slide-deck PDF and keep the VLM transcription of every slide for manual review.

    uv run --extra ingestion python evaluation/vlm_slide_check.py <pdf> <workspace> [--no-insert]

Writes evaluation/results/vlm_slides_<stem>.json (one item per slide: page, text, coverage of the
PDF text layer) before inserting, so the transcripts survive a failed insert.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))


async def main(pdf: Path, workspace: str, insert: bool) -> None:
    import logging

    import pymupdf
    from entity_dedup_benchmark import configure_providers

    from hybrid_rag.ingestion import parse_file, workspace_pool
    from hybrid_rag.ingestion.parser.pdf.slides import _coverage

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    configure_providers()
    with pymupdf.open(pdf) as doc:
        references = [page.get_text(sort=True).strip() for page in doc]

    out = ROOT / "evaluation" / "results" / f"vlm_slides_{pdf.stem}.json"
    async with parse_file(pdf) as content_list:
        slides = [
            {"page": item["page_idx"] + 1, "coverage": round(_coverage(references[item["page_idx"]], item["text"]), 3),
             "text": item["text"]}
            for item in content_list if item.get("type") == "text"
        ]
        out.write_text(json.dumps(slides, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{len(slides)} slide → {out}")
        if insert:
            async with workspace_pool.lease(workspace) as rag:
                await rag.insert_content_list(content_list, file_path=pdf, doc_id=f"doc-{pdf.stem}", save_parent=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pdf", type=Path)
    p.add_argument("workspace")
    p.add_argument("--no-insert", action="store_true")
    a = p.parse_args()
    asyncio.run(main(a.pdf.resolve(), a.workspace, not a.no_insert))
