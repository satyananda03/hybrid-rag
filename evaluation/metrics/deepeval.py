# import sys
# import asyncio
# from pathlib import Path
# import pandas as pd
# from deepeval import evaluate
# from deepeval.evaluate.configs import DisplayConfig, ErrorConfig
# from deepeval.metrics import (
#     AnswerRelevancyMetric,
#     ContextualPrecisionMetric,
#     ContextualRecallMetric,
#     FaithfulnessMetric,
# )
# from deepeval.models import LiteLLMModel
# from deepeval.test_case import LLMTestCase

# from app.core.config import settings
# from app.rag_pipeline.retrieve import evaluate_rag_pipeline
# from app.evals.datasets.loader import load_dataset
# OUT_DIR = Path(__file__).resolve().parent / "experiments"
# RESULTS = OUT_DIR / "deepeval_results.csv"

# JUDGE = LiteLLMModel(
#     model=f"openai/{settings.judge_model_id}",
#     base_url=settings.judge_base_url,
#     api_key=settings.judge_api_key,
#     temperature=0.0,
# )

# def build_metrics():
#     kwargs = dict(threshold=0.0, model=JUDGE, include_reason=True)
#     return [
#         AnswerRelevancyMetric(**kwargs),
#         FaithfulnessMetric(**kwargs),
#         ContextualPrecisionMetric(**kwargs),
#         ContextualRecallMetric(**kwargs),
#     ]

# async def run_pipeline(rows: list[dict]) -> list[dict]:
#     async def one(row):
#         try:
#             r = await evaluate_rag_pipeline(row["query"])
#             return {**row, "answer": r["answer"], "context": r["context"]}
#         except Exception as e:
#             print(f"  x #{row['no']} gagal: {type(e).__name__}: {e}")
#             return None

#     done = await asyncio.gather(*(one(r) for r in rows))
#     return [d for d in done if d]

# def to_dataframe(test_results, generated: list[dict]) -> pd.DataFrame:
#     by_query = {g["query"]: g for g in generated}
#     rows = []
    
#     for tr in test_results:
#         row = {
#             "no": by_query.get(tr.input, {}).get("no"),
#             "query": tr.input,
#             "expected_output": tr.expected_output,
#             "answer": tr.actual_output,
#             "context": "\n---\n".join(tr.retrieval_context or []),
#         }
#         for m in tr.metrics_data or []:
#             key = m.name.lower().replace(" ", "_").replace("[", "").replace("]", "")
#             row[f"{key}_score"] = m.score
#             row[f"{key}_reason"] = m.error or m.reason
#         rows.append(row)
        
#     return pd.DataFrame(rows).sort_values("no", key=lambda s: s.astype(int))

# def summarize(df: pd.DataFrame) -> None:
#     score_cols = [c for c in df.columns if c.endswith("_score")]
#     print(f"\nRINGKASAN  ({len(df)} soal)")
#     for c in score_cols:
#         print(f"  {c.removesuffix('_score'):<22} mean {df[c].mean():.3f}  median {df[c].median():.3f}")

# def run(csv_path: str):
#     rows = load_dataset(csv_path)
#     generated = asyncio.run(run_pipeline(rows))
#     if not generated:
#         raise SystemExit("Tidak ada output RAG yang dihasilkan")
#     cases = [
#         LLMTestCase(
#             input=g["query"],
#             actual_output=g["answer"],
#             expected_output=g["expected_output"],
#             retrieval_context=g["context"],
#         )
#         for g in generated
#     ]
#     print("\nMengevaluasi hasil dengan DeepEval...")
#     result = evaluate(
#         test_cases=cases,
#         metrics=build_metrics(),
#         display_config=DisplayConfig(print_results=False, inspect_after_run=False),
#         error_config=ErrorConfig(ignore_errors=True, skip_on_missing_params=True),
#     )

#     df = to_dataframe(result.test_results, generated)
#     OUT_DIR.mkdir(parents=True, exist_ok=True)
#     df.to_csv(RESULTS, index=False)
#     print(f"Hasil evaluasi disimpan di: {RESULTS}")
#     summarize(df)

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         sys.exit(1)
#     csv_file_path = sys.argv[1]
#     run(csv_file_path)